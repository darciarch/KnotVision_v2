"""structure: layout (symmetry, border + period, medallion, 2-D repeat), gradient mask, coarse label
map at grid resolution, class detection with margin, thin_line_frac, internal geometry (border /
ground repeat / knot grid field), TPS correction, region map.

Output: layout.json, regions.npy, coarse_labels.npy, gradient_mask.npy, gradient_regions.json,
        rectified*.png (rewritten only if TPS applied), job.json (resolved_params, final design_mode)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

from . import DESIGN_CLASSES, PX_PER_KNOT
from .color import delta_e2000, rgb_to_lab
from .job import Job, dump_json, read_json
from .metrics import canny_L, component_area_map, connected_components, is_thin, noise_stats

DETECT_CONF = 0.6          # detection threshold for symmetry / period / repeat / medallion
GRAD_MIN_DEFAULT = 0.04    # L per knot, gradient mask lower bound
GRAD_MAX_DEFAULT = 1.5     # L per knot, upper bound (above this it is an edge, not a ramp)
REGION_GROUND, REGION_BORDER, REGION_MEDALLION, REGION_OTHER, REGION_CORNER = 0, 1, 2, 3, 4


def _read_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _resize_long(img: np.ndarray, long_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_px / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- 1. symmetry

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def detect_symmetry(L: np.ndarray) -> dict:
    ch = _corr(L, L[:, ::-1])
    cv = _corr(L, L[::-1, :])
    h_ok, v_ok = ch >= DETECT_CONF, cv >= DETECT_CONF
    kind = "hv" if (h_ok and v_ok) else "h" if h_ok else "v" if v_ok else "none"
    conf = {"hv": min(ch, cv), "h": ch, "v": cv, "none": max(ch, cv)}[kind]
    return {"kind": kind, "conf": float(conf), "corr_h": float(ch), "corr_v": float(cv)}


# ----------------------------------------------------------------------------- 2./3. border

def _edge_profile(lab: np.ndarray, edge: str) -> np.ndarray:
    """Per-row (top/bottom) or per-column (left/right) signal from the edge inward:
    [mean L, mean a, mean b, std L]."""
    if edge in ("top", "bottom"):
        m = lab.mean(axis=1)
        s = lab[..., 0].std(axis=1)
    else:
        m = lab.mean(axis=0)
        s = lab[..., 0].std(axis=0)
    sig = np.concatenate([m, s[:, None]], axis=1)
    if edge in ("bottom", "right"):
        sig = sig[::-1]
    return sig


def _best_change(sig: np.ndarray, lo: int, hi: int) -> tuple[Optional[int], float]:
    """Largest mean shift between sig[0:t] and sig[t:2t] for t in [lo, hi]. Returns (t, score)."""
    best_t, best = None, 0.0
    n = len(sig)
    for t in range(lo, min(hi, n // 2) + 1):
        a = sig[:t].mean(0)
        b = sig[t:2 * t].mean(0)
        within = sig[:t].std(0).mean() + sig[t:2 * t].std(0).mean() + 1e-6
        score = np.linalg.norm(a - b) / within
        if score > best:
            best, best_t = score, t
    return best_t, float(best)


def detect_border(lab512: np.ndarray, frac_min: float, frac_max: float) -> dict:
    h, w = lab512.shape[:2]
    short = min(h, w)
    out = {}
    for edge in ("top", "bottom", "left", "right"):
        sig = _edge_profile(lab512, edge)
        lo = max(3, int(short * frac_min))
        hi = max(lo, int(short * frac_max))
        t, score = _best_change(sig, lo, hi)
        out[edge] = {"thickness_px": t, "score": score, "detected": bool(t is not None and score >= 1.5)}
    return out


def _autocorr_1d(x: np.ndarray) -> np.ndarray:
    """Zero-padded autocorrelation with the (n - k) / n overlap bias removed."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-9)
    k = np.arange(n)
    return ac * n / np.maximum(n - k, 1)


def border_period(L512: np.ndarray, border: dict, min_period: int = 6) -> dict:
    """1-D autocorrelation along each border strip (corner blocks excluded)."""
    h, w = L512.shape
    res = {}
    th = {e: border[e]["thickness_px"] or 0 for e in border}
    strips = {
        "top": L512[:th["top"], th["left"]:w - th["right"]] if th["top"] else None,
        "bottom": L512[h - th["bottom"]:, th["left"]:w - th["right"]] if th["bottom"] else None,
        "left": L512[th["top"]:h - th["bottom"], :th["left"]].T if th["left"] else None,
        "right": L512[th["top"]:h - th["bottom"], w - th["right"]:].T if th["right"] else None,
    }
    for e, s in strips.items():
        if s is None or s.size == 0 or s.shape[1] < 4 * min_period:
            res[e] = {"period_px": None, "conf": 0.0}
            continue
        sig = s.mean(axis=0)
        ac = _autocorr_1d(sig)
        n = len(ac)
        rng = ac[min_period:n // 2]
        if len(rng) == 0:
            res[e] = {"period_px": None, "conf": 0.0}
            continue
        # first local maximum above threshold
        k = int(np.argmax(rng)) + min_period
        conf = float(ac[k])
        res[e] = {"period_px": int(k) if conf >= DETECT_CONF else None, "conf": conf}
    return res


# ----------------------------------------------------------------------------- 4. medallion

def detect_medallion(L512: np.ndarray, edges512: np.ndarray) -> dict:
    h, w = L512.shape
    y0, y1 = int(h * 0.3), int(h * 0.7)
    x0, x1 = int(w * 0.3), int(w * 0.7)
    center = edges512[y0:y1, x0:x1].astype(np.uint8)
    if center.sum() == 0:
        return {"detected": False, "conf": 0.0, "bbox": None}
    closed = cv2.morphologyEx(center, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    filled = ndi.binary_fill_holes(closed > 0)
    lab, n = ndi.label(filled)
    if n == 0:
        return {"detected": False, "conf": 0.0, "bbox": None}
    sizes = ndi.sum(filled, lab, index=np.arange(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    m = lab == i
    area_frac = m.mean()
    ys, xs = np.nonzero(m)
    bbox = [int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max()), int(y0 + ys.max())]
    # contrast of the region vs its surroundings
    inner = L512[y0:y1, x0:x1][m].mean()
    outer = L512[y0:y1, x0:x1][~m].mean() if (~m).any() else inner
    conf = float(min(1.0, abs(inner - outer) / 20.0) * min(1.0, area_frac / 0.2))
    return {"detected": bool(conf >= DETECT_CONF), "conf": conf, "bbox": bbox}


# ----------------------------------------------------------------------------- 5. 2-D repeat

def detect_repeat(L: np.ndarray, min_period: int, max_period: int) -> dict:
    """2-D autocorrelation -> two strongest independent peaks (v1, v2) in pixels of `L`."""
    h, w = L.shape
    L = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.0)   # pixel noise / pile texture
    x = L - L.mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fft2(x * win)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(f) ** 2).real)
    cy, cx = h // 2, w // 2
    ac = ac / (ac[cy, cx] + 1e-9)
    # remove the window's own taper (overlap bias) so that peaks at large lags keep their height
    fw = np.fft.fft2(win)
    acw = np.fft.fftshift(np.fft.ifft2(np.abs(fw) ** 2).real)
    acw = acw / (acw[cy, cx] + 1e-9)
    ac = np.where(acw > 0.2, ac / np.maximum(acw, 0.2), 0.0)
    ac[cy - min_period + 1:cy + min_period, cx - min_period + 1:cx + min_period] = 0
    # suppress far lags
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    ac[r > max_period] = 0
    ac[:cy, :] = 0  # half-plane only (peaks are symmetric)
    peaks = []
    tmp = ac.copy()
    for _ in range(6):
        k = np.unravel_index(np.argmax(tmp), tmp.shape)
        val = float(tmp[k])
        if val < DETECT_CONF:
            break
        v = (int(k[0] - cy), int(k[1] - cx))
        peaks.append((v, val))
        tmp[max(0, k[0] - min_period // 2):k[0] + min_period // 2 + 1,
            max(0, k[1] - min_period // 2):k[1] + min_period // 2 + 1] = 0
    if not peaks:
        return {"v1": None, "v2": None, "conf": 0.0}
    v1, c1 = peaks[0]
    v2, c2 = None, 0.0
    for v, c in peaks[1:]:
        cross = abs(v1[0] * v[1] - v1[1] * v[0])
        if cross > 0.3 * (np.hypot(*v1) * np.hypot(*v)):
            v2, c2 = v, c
            break
    return {"v1": list(v1), "v2": list(v2) if v2 else None, "conf": float(c1 if v2 is None else min(c1, c2))}


def lattice_nodes(v1, v2, H, W, origin=(0, 0)) -> np.ndarray:
    if v1 is None:
        return np.zeros((0, 2))
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float) if v2 is not None else np.array([-v1[1], v1[0]])
    nodes = []
    n = int(2 * max(H, W) / max(1.0, min(np.hypot(*v1), np.hypot(*v2)))) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            p = np.array(origin) + i * v1 + j * v2
            if 0 <= p[0] < H and 0 <= p[1] < W:
                nodes.append(p)
    return np.array(nodes) if nodes else np.zeros((0, 2))


# ----------------------------------------------------------------------------- 5b. gradient mask

def gradient_mask(rect_rgb: np.ndarray, W: int, H: int, grad_min: float, grad_max: float,
                  block_knots: int = 32, edge_density_max: float = 0.02) -> tuple[np.ndarray, list]:
    """Blocks of 32x32 knots with low edge density AND a low-frequency (σ = 8 knots) one-directional
    ramp (|∇L| in [grad_min, grad_max] L/knot, direction consistency >= 0.8). Blocks are merged,
    closed with an 8-knot element, regions < 64 knots dropped. Returns (mask (H, W) bool, regions)."""
    lab = rgb_to_lab(rect_rgb)
    # to grid resolution
    Lg = cv2.resize(lab[..., 0].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    ag = cv2.resize(lab[..., 1].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(lab[..., 2].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    # dither texture (1-knot checkers) must not count as edges: 1-knot blur before Canny
    soft = cv2.GaussianBlur(rect_rgb, (0, 0), float(PX_PER_KNOT))
    edges = cv2.resize(canny_L(soft).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    # a step edge smoothed with sigma = 8 knots looks like a ramp; exclude a 3-knot band around edges
    # and smooth with a normalised convolution so that only genuinely smooth areas contribute
    edge_band = cv2.dilate((edges > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    valid = (~edge_band).astype(np.float32)
    def _nconv(sig):
        num = cv2.GaussianBlur(Lg * valid, (0, 0), sig)
        den = cv2.GaussianBlur(valid, (0, 0), sig)
        return np.where(den > 0.05, num / np.maximum(den, 1e-6), Lg)
    Ls = _nconv(8)    # ramp direction / magnitude / plane fit (dither steps averaged out)
    gx = cv2.Sobel(Ls, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    mask = np.zeros((H, W), dtype=bool)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for y in range(0, H, block_knots):
        for x in range(0, W, block_knots):
            ys, xs = slice(y, min(H, y + block_knots)), slice(x, min(W, x + block_knots))
            if edges[ys, xs].mean() > edge_density_max:
                continue
            v = valid[ys, xs] > 0
            if v.sum() < 0.5 * v.size:
                continue
            m = mag[ys, xs][v]
            mm = m.mean()
            if not (grad_min <= mm <= grad_max):
                continue
            vx, vy = gx[ys, xs][v].mean(), gy[ys, xs][v].mean()
            coherence = np.hypot(vx, vy) / (mm + 1e-6)
            if coherence < 0.8:
                continue
            # the raw L inside the block (edge band excluded) must be explained by a plane
            Lb = Ls[ys, xs][v]
            A = np.column_stack([xx[ys, xs][v], yy[ys, xs][v], np.ones(int(v.sum()), np.float32)])
            coef, *_ = np.linalg.lstsq(A, Lb, rcond=None)
            resid = Lb - A @ coef
            var = float(Lb.var())
            r2 = 1.0 - float(resid.var()) / var if var > 1e-6 else 0.0
            if r2 >= 0.6 and abs(coef[0]) + abs(coef[1]) >= grad_min:
                mask[ys, xs] = True
    if mask.any():
        k = np.ones((8, 8), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        lab_m, n = ndi.label(mask)
        sizes = ndi.sum(mask, lab_m, index=np.arange(1, n + 1))
        keep = np.zeros_like(mask)
        regions = []
        for i, s in enumerate(sizes, start=1):
            if s < 64:
                continue
            rm = lab_m == i
            keep |= rm
            d = np.array([gy[rm].mean(), gx[rm].mean()])
            d = d / (np.linalg.norm(d) + 1e-9)
            ys, xs = np.nonzero(rm)
            proj = ys * d[0] + xs * d[1]
            lo, hi = np.percentile(proj, 5), np.percentile(proj, 95)
            sel_lo, sel_hi = proj <= lo + 1e-6, proj >= hi - 1e-6
            lab_lo = [float(Lg[rm][sel_lo].mean()), float(ag[rm][sel_lo].mean()), float(bg[rm][sel_lo].mean())]
            lab_hi = [float(Lg[rm][sel_hi].mean()), float(ag[rm][sel_hi].mean()), float(bg[rm][sel_hi].mean())]
            regions.append({"id": len(regions), "area_knots": int(s), "direction": d.tolist(),
                            "lab_lo": lab_lo, "lab_hi": lab_hi,
                            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
        mask = keep
    else:
        regions = []
    return mask, regions


# ----------------------------------------------------------------------------- 6. coarse labels

def coarse_labels(rect_rgb: np.ndarray, W: int, H: int, k: int = 16, merge_de: float = 8.0):
    small = _resize_long(rect_rgb, 400)
    lab = rgb_to_lab(small).reshape(-1, 3)
    km = KMeans(n_clusters=min(k, len(lab)), random_state=0, n_init=4).fit(lab)
    centers = km.cluster_centers_
    # merge centres closer than merge_de (greedy, lowest index survives)
    alive = list(range(len(centers)))
    merged = True
    while merged and len(alive) > 1:
        merged = False
        best = None
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                d = float(delta_e2000(centers[alive[i]], centers[alive[j]]))
                if d < merge_de and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            _, i, j = best
            ci, cj = alive[i], alive[j]
            wi = (km.labels_ == ci).sum()
            wj = (km.labels_ == cj).sum()
            centers[ci] = (centers[ci] * wi + centers[cj] * wj) / max(1, wi + wj)
            km.labels_[km.labels_ == cj] = ci
            alive.pop(j)
            merged = True
    # clusters with < 1% share are edge-transition colours (pile blur mixing two yarns), not yarns:
    # fold them into the nearest surviving centre (lowest index on ties) before counting
    shares = np.array([(km.labels_ == c).sum() for c in alive], dtype=np.float64) / max(1, len(km.labels_))
    keep = [c for c, s_ in zip(alive, shares) if s_ >= 0.01] or [alive[int(np.argmax(shares))]]
    centers = centers[keep]
    grid_rgb = cv2.resize(rect_rgb, (W, H), interpolation=cv2.INTER_AREA)
    glab = rgb_to_lab(grid_rgb).reshape(-1, 3)
    d = delta_e2000(glab[:, None, :], centers[None, :, :])
    labels = np.argmin(d, axis=1).astype(np.int32).reshape(H, W)
    return labels, centers


def thin_line_frac(labels: np.ndarray, line_min_knots: int, line_max_width: int,
                   centers: Optional[np.ndarray] = None) -> float:
    """Knot share of is_thin components. A thin component whose colour lies between the colours of its
    two dominant neighbours (a pile-blur transition band between two yarns) is not a line and is skipped."""
    total = 0
    struct = ndi.generate_binary_structure(2, 1)
    for c in connected_components(labels):
        if c.area < line_min_knots:
            continue
        fm = c.full_mask(labels.shape)
        if not is_thin(fm, line_min_knots, line_max_width):
            continue
        if centers is not None:
            ring = ndi.binary_dilation(fm, structure=struct, iterations=1) & ~fm
            nb = labels[ring]
            nb = nb[nb != c.index]
            if len(nb):
                vals, counts = np.unique(nb, return_counts=True)
                order = np.argsort(-counts)
                if len(vals) >= 2 and counts[order[1]] >= 0.25 * counts[order[0]]:
                    a, b = centers[vals[order[0]]], centers[vals[order[1]]]
                    m = centers[c.index]
                    ab = b - a
                    L2 = float(ab @ ab) + 1e-9
                    t = float(((m - a) @ ab) / L2)
                    dist = float(np.linalg.norm(m - (a + t * ab)))
                    if 0.0 <= t <= 1.0 and dist < 0.35 * np.sqrt(L2):
                        continue  # transition band, not a drawn line
        total += c.area
    return float(total / labels.size)


def line_frac(edges512: np.ndarray) -> float:
    e = edges512.astype(np.uint8) * 255
    if e.sum() == 0:
        return 0.0
    h, w = e.shape
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=30, minLineLength=int(min(h, w) * 0.08), maxLineGap=4)
    if lines is None:
        return 0.0
    cover = np.zeros_like(e)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        cv2.line(cover, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
    return float(((e > 0) & (cover > 0)).sum() / (e > 0).sum())


# ----------------------------------------------------------------------------- class detection

def _ratio_ge(value: float, thr: float) -> float:
    return value / thr if thr > 0 else (np.inf if value > 0 else 0.0)


def _ratio_lt(value: float, thr: float) -> float:
    return thr / value if value > 0 else np.inf


def _ratio_bool(conf: float, thr: float = DETECT_CONF) -> float:
    return conf / thr


def classify(f: dict, P: dict) -> dict:
    """Ordered rules; each condition -> ratio to its threshold; fire_i = min(ratios); first fire_i >= 1 wins.
    class_margin = fire_i - 1 for rules 1-6, 1 - max(fire) for the fallback rule 7."""
    ed_min = float(P.get("edge_density_min", 0.02))
    has_structure = max(f["symmetry_conf"], f["border_period_conf"], f["ground_repeat_conf"]) >= DETECT_CONF
    rules = [
        ("pictorial", [_ratio_ge(f["gradient_frac"], 0.6),
                       (1.0 / max(1e-6, _ratio_bool(max(f["symmetry_conf"], f["border_period_conf"], f["ground_repeat_conf"])))),
                       _ratio_ge(f["edge_density"], ed_min)]),
        ("distressed", [_ratio_ge(f["noise_frac"], float(P.get("noise_class_frac", 0.08))),
                        0.0 if f["pile_blur"] else np.inf]),
        ("gradient", [_ratio_ge(f["gradient_frac"], float(P.get("gradient_class_frac", 0.25)))]),
        ("minimal", [_ratio_lt(f["n_colors_est"], 3.5), _ratio_lt(f["edge_density"], ed_min)]),
        ("classic", [max(_ratio_bool(f["symmetry_conf"]) if f["symmetry"] == "hv" else 0.0,
                         min(1.0 if f["border_accepted"] else 0.0, _ratio_bool(f["medallion_conf"])))]),
        ("geometric", [max(_ratio_bool(f["ground_repeat_conf"]), _ratio_ge(f["line_frac"], 0.6))]),
    ]
    fires = []
    for name, ratios in rules:
        fire = float(min(ratios)) if ratios else 0.0
        fires.append((name, fire))
    for name, fire in fires:
        if fire >= 1.0:
            return {"class_auto": name, "class_margin": float(min(fire - 1.0, 10.0)), "fires": dict(fires)}
    best = max(f_ for _, f_ in fires)
    return {"class_auto": "classic", "class_margin": float(1.0 - best), "fires": dict(fires)}


# ----------------------------------------------------------------------------- 7. internal geometry

def border_inner_edge_dev(lab_full: np.ndarray, border: dict, scale: float) -> Optional[float]:
    """Refit the inner border edge per column (top/bottom) / row (left/right) at full resolution and
    measure the max deviation from a straight line, in knots."""
    h, w = lab_full.shape[:2]
    devs = []
    L = lab_full[..., 0].astype(np.float32)
    for e in ("top", "bottom", "left", "right"):
        t = border[e]["thickness_px"]
        if not t:
            continue
        t_full = int(t * scale)
        rng = slice(max(1, int(t_full * 0.6)), min(int(t_full * 1.4), (h if e in ("top", "bottom") else w) // 2))
        if rng.stop - rng.start < 4:
            continue
        if e == "top":
            g = np.abs(np.diff(L, axis=0))[rng, :]
            pos = np.argmax(g, axis=0) + rng.start
        elif e == "bottom":
            g = np.abs(np.diff(L[::-1], axis=0))[rng, :]
            pos = np.argmax(g, axis=0) + rng.start
        elif e == "left":
            g = np.abs(np.diff(L, axis=1))[:, rng]
            pos = np.argmax(g, axis=1) + rng.start
        else:
            g = np.abs(np.diff(L[:, ::-1], axis=1))[:, rng]
            pos = np.argmax(g, axis=1) + rng.start
        x = np.arange(len(pos))
        step = max(1, len(pos) // 200)
        xs, ps = x[::step], pos[::step].astype(np.float64)
        a, b = np.polyfit(xs, ps, 1)
        resid = ps - (a * xs + b)
        # robust: 95th percentile of abs residual to ignore motif hits
        devs.append(float(np.percentile(np.abs(resid), 95)) / PX_PER_KNOT)
    return max(devs) if devs else None


def knot_grid_field(contrast_rgb: np.ndarray, block: int = 64) -> dict:
    """2-D autocorrelation in 64x64 px blocks; expected peaks at (4,0) and (0,4) px. Peak >= 0.4 in
    >= 70% of blocks -> visible. Returns spacing (px), per-block displacement field (px)."""
    L = rgb_to_lab(contrast_rgb)[..., 0].astype(np.float32)
    L = L - cv2.GaussianBlur(L, (0, 0), 3.0)  # high-pass: keep the pile ripple, drop motifs / shading
    h, w = L.shape
    ny, nx = h // block, w // block
    if ny < 2 or nx < 2:
        return {"visible": False, "spacing_px": None, "mismatch_pct": None, "disp": None, "peak_frac": 0.0}
    peaks, spacings, phases = [], [], []
    for by in range(ny):
        row_p, row_s, row_ph = [], [], []
        for bx in range(nx):
            blk = L[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            x = blk - blk.mean()
            if x.std() < 1e-3:
                row_p.append(0.0); row_s.append((np.nan, np.nan)); row_ph.append((np.nan, np.nan))
                continue
            f = np.fft.fft2(x)
            ac = np.fft.ifft2(np.abs(f) ** 2).real
            ac /= ac[0, 0] + 1e-9
            # a local maximum at lag 3..5 (strictly above both neighbours) in each axis
            rowac, colac = ac[0, :8], ac[:8, 0]

            def local_peak(a):
                best, best_v = None, -1.0
                for lag in (3, 4, 5):
                    if a[lag] > a[lag - 1] and a[lag] > a[lag + 1] and a[lag] > best_v:
                        best, best_v = lag, float(a[lag])
                return best, best_v
            lx, px = local_peak(rowac)
            ly, py = local_peak(colac)
            if lx is None or ly is None:
                row_p.append(0.0); row_s.append((np.nan, np.nan)); row_ph.append((np.nan, np.nan))
                continue
            row_p.append(min(px, py))
            row_s.append((float(lx), float(ly)))
            # phase of the fundamental at period 4 -> sub-pixel grid offset
            kx = block // 4
            phx = np.angle(f[0, kx]) / (2 * np.pi) * 4.0
            phy = np.angle(f[kx, 0]) / (2 * np.pi) * 4.0
            row_ph.append((phx, phy))
        peaks.append(row_p); spacings.append(row_s); phases.append(row_ph)
    peaks = np.array(peaks)
    ok = peaks >= 0.4
    frac = float(ok.mean())
    if frac < 0.7:
        return {"visible": False, "spacing_px": None, "mismatch_pct": None, "disp": None, "peak_frac": frac}
    sp = np.array(spacings)
    spacing = float(np.nanmedian(sp[ok]))
    ph = np.array(phases)
    ref = np.nanmedian(ph[ok], axis=0)
    disp = ph - ref
    disp = (disp + 2.0) % 4.0 - 2.0  # unwrap into [-2, 2) px
    disp[~ok] = np.nan
    return {"visible": True, "spacing_px": spacing, "mismatch_pct": abs(spacing - 4.0) / 4.0,
            "disp": disp, "peak_frac": frac}


def tps_warp(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    tps = cv2.createThinPlateSplineShapeTransformer()
    s = src_pts.reshape(1, -1, 2).astype(np.float32)
    d = dst_pts.reshape(1, -1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    # OpenCV warpImage applies the inverse mapping: estimate target -> source
    tps.estimateTransformation(d, s, matches)
    return tps.warpImage(img)


# ----------------------------------------------------------------------------- 9. regions

def region_map(shape512, border: dict, medallion: dict, W: int, H: int, labels: np.ndarray) -> np.ndarray:
    h, w = shape512
    r512 = np.full((h, w), REGION_GROUND, dtype=np.uint8)
    th = {e: border[e]["thickness_px"] or 0 for e in border}
    if all(border[e]["detected"] for e in border):
        r512[:th["top"], :] = REGION_BORDER
        r512[h - th["bottom"]:, :] = REGION_BORDER
        r512[:, :th["left"]] = REGION_BORDER
        r512[:, w - th["right"]:] = REGION_BORDER
        r512[:th["top"], :th["left"]] = REGION_CORNER
        r512[:th["top"], w - th["right"]:] = REGION_CORNER
        r512[h - th["bottom"]:, :th["left"]] = REGION_CORNER
        r512[h - th["bottom"]:, w - th["right"]:] = REGION_CORNER
    if medallion.get("detected") and medallion.get("bbox"):
        x0, y0, x1, y1 = medallion["bbox"]
        sub = r512[y0:y1 + 1, x0:x1 + 1]
        sub[sub == REGION_GROUND] = REGION_MEDALLION
    rg = cv2.resize(r512, (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.empty((H, W), dtype=np.uint8)
    # whole components go to the region they overlap most
    for c in connected_components(labels):
        fm = c.full_mask(labels.shape)
        vals = rg[fm]
        out[fm] = np.bincount(vals, minlength=5).argmax()
    return out


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("structure")
    rd = job.stage_dir("rectify")
    rect_png, con_png = rd / "rectified.png", rd / "rectified_contrast.png"
    key = job.cache_key("structure", [rect_png, con_png], exclude_prefix="edit_")
    if job.cached(sd, key) and (sd / "layout.json").exists():
        job.timings["structure"] = 0.0
        return job.state

    P = job.params
    grid = job.state["grid"]
    W, H = grid["W"], grid["H"]
    rect = _read_rgb(rect_png)
    con = _read_rgb(con_png)
    layout = _analyse(job, rect, con, W, H, sd)

    # 8. TPS (only when non_planar_internal and a displacement source exists)
    ig = layout["internal_geometry"]
    ig["tps_applied"] = False
    if job.state.get("non_planar_internal") and layout.get("_tps_pts") is not None:
        src, dst = layout.pop("_tps_pts")
        try:
            rect2 = tps_warp(rect, src, dst)
            con2 = tps_warp(con, src, dst)
            before = ig["max_dev_knots_before"]
            layout2 = _analyse(job, rect2, con2, W, H, sd, quiet=True)
            after = layout2["internal_geometry"]["max_dev_knots_before"]
            if after is not None and before and after <= 0.5 * before:
                rect, con = rect2, con2
                cv2.imwrite(str(rect_png), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(con_png), cv2.cvtColor(con, cv2.COLOR_RGB2BGR))
                layout2.pop("_tps_pts", None)
                layout = layout2
                layout["internal_geometry"]["tps_applied"] = True
                layout["internal_geometry"]["max_dev_knots_before"] = before
                layout["internal_geometry"]["max_dev_knots_after"] = after
                job.state["non_planar_internal"] = bool(after > float(P.get("planarity_warn", 0.02)) * max(W, H))
            else:
                job.warn("tps_no_gain")
        except cv2.error:
            job.warn("tps_no_gain")
    layout.pop("_tps_pts", None)
    if layout["internal_geometry"].get("max_dev_knots_after") is None:
        layout["internal_geometry"]["max_dev_knots_after"] = layout["internal_geometry"]["max_dev_knots_before"]

    dump_json(sd / "layout.json", layout)
    job.state["layout"] = {k: v for k, v in layout.items() if k not in ("lattice_nodes",)}
    job.state["internal_geometry"] = layout["internal_geometry"]
    job.state.setdefault("stages_ran", []).append("structure")
    job.timings["structure"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state


def _analyse(job: Job, rect: np.ndarray, con: np.ndarray, W: int, H: int, sd: Path, quiet: bool = False) -> dict:
    P = job.params
    grid = job.state["grid"]
    con512 = _resize_long(con, 512)
    scale = con.shape[1] / con512.shape[1]
    lab512 = rgb_to_lab(con512)
    L512 = lab512[..., 0].astype(np.float32)
    edges512 = canny_L(con512)
    ed = float(edges512.mean())

    # 1. symmetry
    sym = detect_symmetry(L512)
    # 2./3. border + period
    border = detect_border(lab512, float(P.get("border_frac_min", 0.03)), float(P.get("border_frac_max", 0.30)))
    period = border_period(L512, border)
    short = min(L512.shape)
    all_det = all(border[e]["detected"] for e in border)
    period_conf = max(p["conf"] for p in period.values()) if all_det else 0.0
    th_ok = all_det and all(float(P["border_frac_min"]) <= (border[e]["thickness_px"] or 0) / short <= float(P["border_frac_max"]) for e in border)
    border_accepted = bool(all_det and period_conf >= DETECT_CONF and th_ok)
    frame_hint = None if border_accepted else "panel"
    if not border_accepted:
        for e in border:
            border[e]["detected"] = False
    # 4. medallion
    med = detect_medallion(L512, edges512)
    # 5. 2-D repeat on the ground
    th = {e: (border[e]["thickness_px"] or 0) if border_accepted else 0 for e in border}
    ground = L512[th["top"]:L512.shape[0] - th["bottom"], th["left"]:L512.shape[1] - th["right"]]
    min_p = max(6, int(8 * (512 / max(W, H)) / 1))  # >= ~8 knots
    rep = detect_repeat(ground, min_period=min_p, max_period=min(ground.shape) // 2) if min(ground.shape) > 4 * min_p else {"v1": None, "v2": None, "conf": 0.0}
    knots_per_px512 = max(W, H) / max(L512.shape)
    rep_knots = {"v1": [v * knots_per_px512 for v in rep["v1"]] if rep["v1"] else None,
                 "v2": [v * knots_per_px512 for v in rep["v2"]] if rep["v2"] else None,
                 "conf": rep["conf"]}
    nodes = lattice_nodes(rep_knots["v1"], rep_knots["v2"], H, W) if rep_knots["v1"] else np.zeros((0, 2))
    # 5b. gradient mask
    gmask, gregions = gradient_mask(rect, W, H, float(P.get("grad_min", GRAD_MIN_DEFAULT)),
                                    float(P.get("grad_max", GRAD_MAX_DEFAULT)))
    gradient_frac = float(gmask.mean())
    # 6. coarse labels + class
    labels, centers = coarse_labels(rect, W, H)
    ns = noise_stats(labels, ~gmask, int(P.get("noise_cc_min", 2)), int(P.get("noise_cc_max", 6)),
                     int(P.get("min_region_knots", 2)))
    tlf = thin_line_frac(labels, int(P.get("line_min_knots", 6)), int(P.get("line_max_width", 1)), centers)
    lf = line_frac(edges512)
    feats = {"edge_density": ed, "n_colors_est": int(len(centers)), "gradient_frac": gradient_frac,
             "noise_frac": ns["noise_frac"], "noise_1knot_frac": ns["noise_1knot_frac"], "line_frac": lf,
             "thin_line_frac": tlf, "symmetry": sym["kind"], "symmetry_conf": sym["conf"],
             "border_accepted": border_accepted, "border_period_conf": period_conf,
             "ground_repeat_conf": rep["conf"], "medallion_conf": med["conf"],
             "pile_blur": job.has_warning("pile_blur")}
    cls = classify(feats, P)
    if not quiet:
        _apply_class(job, cls, feats)
    # 7. internal geometry
    lab_full = rgb_to_lab(con)
    sources, devs = [], []
    tps_src, tps_dst = [], []
    if border_accepted:
        d = border_inner_edge_dev(lab_full, border, scale)
        if d is not None:
            sources.append("border"); devs.append(d)
    if rep_knots["v1"] and len(nodes):
        sources.append("ground_repeat"); devs.append(0.0)  # ideal lattice is what we fitted; deviation measured by conf
    kg = {"visible": False, "spacing_px": None, "mismatch_pct": None}
    eff = grid.get("photo_px_per_knot_eff") or 0.0
    if eff >= 2.0 and not job.has_warning("pile_blur"):
        kgf = knot_grid_field(con)
        kg = {"visible": kgf["visible"], "spacing_px": kgf["spacing_px"], "mismatch_pct": kgf["mismatch_pct"]}
        if kgf["visible"]:
            sources.append("knot_grid")
            disp = kgf["disp"]
            mag = np.nan_to_num(np.hypot(disp[..., 0], disp[..., 1]))
            devs.append(float(mag.max()) / PX_PER_KNOT)
            if kgf["mismatch_pct"] > float(P.get("grid_mismatch_tol", 0.05)) and not quiet:
                job.warn("grid_mismatch")
            grid["measured_reed_per_m"] = float(grid["reed_per_m"] * 4.0 / kgf["spacing_px"])
            # TPS control points: 8x8 samples from the field
            ny, nx = disp.shape[:2]
            for iy in np.linspace(0, ny - 1, 8).astype(int):
                for ix in np.linspace(0, nx - 1, 8).astype(int):
                    if np.isnan(disp[iy, ix, 0]):
                        continue
                    cx, cy = (ix + 0.5) * 64, (iy + 0.5) * 64
                    tps_src.append([cx + disp[iy, ix, 0], cy + disp[iy, ix, 1]])
                    tps_dst.append([cx, cy])
    grid["knot_grid"] = kg
    max_dev = max(devs) if devs else None
    ig = {"source": sources, "tps_applied": False, "max_dev_knots_before": max_dev, "max_dev_knots_after": None}
    if not sources:
        ig["status"] = "unmeasurable"
        if not quiet:
            job.warn("internal_geometry_unmeasurable")
        job.state["non_planar_internal"] = False
    else:
        ig["status"] = "measured"
        thr = float(P.get("planarity_warn", 0.02)) * max(W, H)
        npi = bool(max_dev is not None and max_dev > thr)
        job.state["non_planar_internal"] = npi
        if npi and not quiet:
            job.warn("non_planar_internal")
    # 9. regions
    regions = region_map(L512.shape, border, med, W, H, labels)

    layout = {
        "symmetry": sym, "border": border, "border_accepted": border_accepted, "frame_hint": frame_hint,
        "border_period": period, "medallion": med, "ground_repeat": rep_knots,
        "lattice_nodes": nodes.tolist(), "gradient_frac": gradient_frac, "class_features": feats,
        "internal_geometry": ig, "scale_512": scale,
        "_tps_pts": (np.array(tps_src), np.array(tps_dst)) if len(tps_src) >= 16 else None,
    }
    if not quiet:
        np.save(sd / "coarse_labels.npy", labels)
        np.save(sd / "gradient_mask.npy", gmask)
        np.save(sd / "regions.npy", regions)
        dump_json(sd / "gradient_regions.json", gregions)
        np.save(sd / "coarse_centers.npy", centers)
    return layout


def _apply_class(job: Job, cls: dict, feats: dict):
    P = job.params
    auto = cls["class_auto"]
    margin = cls["class_margin"]
    job.state["class_auto"] = auto
    job.state["class_margin"] = margin
    job.state["class_features"] = {k: feats[k] for k in ("edge_density", "n_colors_est", "gradient_frac",
                                                          "noise_frac", "noise_1knot_frac", "line_frac", "thin_line_frac")}
    job.state["class_conflict"] = False
    mixed = feats["gradient_frac"] >= float(P.get("gradient_class_frac", 0.25)) and \
        feats["noise_frac"] >= float(P.get("noise_class_frac", 0.08))
    job.state["class_mixed"] = bool(mixed)
    if mixed:
        job.warn("class_mixed")
    if job.class_source == "auto":
        if auto == "pictorial":
            job.stop("class_unsupported", "auto-detected pictorial")
            return
        if margin < float(P.get("class_margin_min", 0.2)):
            job.warn("class_uncertain")
        job.design_class = auto
        job.state["design_class"] = auto
        job.resolve(auto)
    else:
        if auto != job.design_class and margin >= float(P.get("class_margin_min", 0.2)):
            job.state["class_conflict"] = True
            job.warn("class_conflict")
    job.params["design_class"] = job.design_class
    mode = job.apply_routing(feats["thin_line_frac"])
    job.state["design_mode"] = mode
    job.state["design_mode_requested"] = job.params.get("design_mode_requested")
    job.state["design_mode_class_default"] = job.params.get("design_mode_class_default")
