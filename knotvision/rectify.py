"""rectify: carpet mask (SAM 3.1 or manual corners), coarse quad + fringe, partial view, corners,
planarity, geometric resolution, homography to 4W x 4H, white balance, illumination flattening,
edge sharpness / effective resolution gates.

Output: rectified.png (colour-correct), rectified_contrast.png (CLAHE L, for structure/edges only),
        mask.png, corners.json, job.json updated (grid, fringe, illumination, wb, warnings)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from . import PX_PER_KNOT
from .color import lab_to_rgb, rgb_to_lab
from .job import MODELS_DIR, Job, JobError, dump_json
from .metrics import guided

SAM_PROMPTS = ("carpet", "rug", "carpet on the floor")


# ----------------------------------------------------------------------------- SAM 3.1 wrapper

def sam_segment(photo_rgb: np.ndarray, prompts=SAM_PROMPTS, min_score: float = 0.3) -> np.ndarray:
    """Text-prompted SAM 3.1 carpet mask. Selection: score x area, next prompt when score < min_score.

    Raises JobError("sam_unavailable") when the package or the checkpoint is missing or inference fails.
    """
    ckpt_dir = MODELS_DIR / "sam3.1"
    if not ckpt_dir.exists():
        raise JobError("sam_unavailable", f"checkpoint dir missing: {ckpt_dir}")
    try:
        import torch  # noqa: F401
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    except Exception as e:  # pragma: no cover - needs the real package
        raise JobError("sam_unavailable", f"sam3 import failed: {e}")
    try:  # pragma: no cover - needs weights + GPU
        model = build_sam3_image_model(checkpoint_path=str(next(ckpt_dir.glob("*.pt"))))
        proc = Sam3Processor(model)
        from PIL import Image
        state = proc.set_image(Image.fromarray(photo_rgb))
        best, best_val = None, -1.0
        for p in prompts:
            out = proc.set_text_prompt(state=state, prompt=p)
            masks, scores = out["masks"], out["scores"]
            for m, s in zip(masks, scores):
                s = float(s)
                if s < min_score:
                    continue
                m = np.asarray(m).astype(bool)
                val = s * m.mean()
                if val > best_val:
                    best, best_val = m, val
            if best is not None:
                break
        if best is None:
            raise JobError("needs_corners", "SAM found no carpet")
        return best
    except JobError:
        raise
    except Exception as e:
        raise JobError("sam_unavailable", f"sam3 inference failed: {e}")


# ----------------------------------------------------------------------------- geometry helpers

def order_corners(pts: np.ndarray) -> np.ndarray:
    """-> [top-left, top-right, bottom-right, bottom-left] (x, y)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(1)
    d = pts[:, 0] - pts[:, 1]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmax(d)], pts[np.argmin(d)]
    return np.array([tl, tr, br, bl])


def quad_mask(shape, corners: np.ndarray) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(m, np.round(corners).astype(np.int32), 1)
    return m.astype(bool)


def largest_component_filled(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    return ndi.binary_fill_holes(m)


def coarse_quad(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return order_corners(box)


def corners_from_mask(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """approxPolyDP (eps = 2% of perimeter) -> 4 corners; else HoughLinesP intersections; else
    minAreaRect + corners_weak. Returns (corners, weak)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.04, 0.015, 0.01):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx[:, 0, :]), False
    hough = _corners_from_hough(mask)
    if hough is not None:
        return hough, False
    return coarse_quad(mask), True


def _line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _corners_from_hough(mask: np.ndarray) -> Optional[np.ndarray]:
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
    h, w = mask.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=int(min(h, w) * 0.1),
                            minLineLength=int(min(h, w) * 0.2), maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None
    lines = np.asarray(lines).reshape(-1, 4)
    ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
    horiz = lines[(ang < 30) | (ang > 150)]
    vert = lines[(ang > 60) & (ang < 120)]
    if len(horiz) < 2 or len(vert) < 2:
        return None
    hy = (horiz[:, 1] + horiz[:, 3]) / 2
    vx = (vert[:, 0] + vert[:, 2]) / 2
    top, bot = horiz[np.argmin(hy)], horiz[np.argmax(hy)]
    left, right = vert[np.argmin(vx)], vert[np.argmax(vx)]
    pts = [_line_intersection(top, left), _line_intersection(top, right),
           _line_intersection(bot, right), _line_intersection(bot, left)]
    if any(p is None for p in pts):
        return None
    pts = np.array(pts)
    if (pts[:, 0] < -w * 0.2).any() or (pts[:, 0] > w * 1.2).any() or (pts[:, 1] < -h * 0.2).any() or (pts[:, 1] > h * 1.2).any():
        return None
    return order_corners(pts)


def planarity_deviation(mask: np.ndarray, corners: np.ndarray) -> float:
    """Max perpendicular distance from the mask contour to the quad edges / that edge's length."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    worst = 0.0
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab) + 1e-9
        t = ((c - a) @ ab) / (L * L)
        on = (t > 0.05) & (t < 0.95)
        if not on.any():
            continue
        proj = a + t[:, None] * ab
        d = np.linalg.norm(c - proj, axis=1)
        # only points that are near this edge (closest edge assignment)
        near = d < 0.25 * L
        if (on & near).any():
            worst = max(worst, float(d[on & near].max() / L))
    return worst


# ----------------------------------------------------------------------------- fringe

def _cusum_change(x: np.ndarray, k: float = 0.5) -> Optional[int]:
    """Index of the sharpest mean shift in a 1-D signal (two-sided CUSUM, returns argmax)."""
    if len(x) < 8:
        return None
    x = (x - x.mean()) / (x.std() + 1e-9)
    best, best_i = 0.0, None
    for i in range(3, len(x) - 3):
        d = abs(x[:i].mean() - x[i:].mean())
        if d > best:
            best, best_i = d, i
    return best_i if best > k else None


def detect_fringe(rect_rgb: np.ndarray, fringe_max_frac: float) -> tuple[list, list, list]:
    """On the coarse-rectified image: per edge, a 1-D profile (saturation mean and 5 px local
    variance); a low-saturation / high-variance strip followed by a sharp transition is a fringe.
    Returns (detected[4], strip_px[4], notes). Edge order: top, bottom, left, right."""
    hsv = cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    gray = cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    var = cv2.blur(gray ** 2, (5, 5)) - cv2.blur(gray, (5, 5)) ** 2
    H, W = S.shape
    detected, strips, notes = [], [], []
    for edge in ("top", "bottom", "left", "right"):
        if edge in ("top", "bottom"):
            sat = S.mean(1)
            vv = var.mean(1)
            n = H
        else:
            sat = S.mean(0)
            vv = var.mean(0)
            n = W
        if edge in ("bottom", "right"):
            sat, vv = sat[::-1], vv[::-1]
        max_w = int(n * fringe_max_frac)
        min_w = max(3, int(n * 0.02))
        seg_s, seg_v = sat[:max_w], vv[:max_w]
        i = _cusum_change(seg_s)
        ok = False
        width = 0
        if i is not None and min_w <= i <= max_w:
            inner_s, outer_s = seg_s[i:].mean(), seg_s[:i].mean()
            inner_v, outer_v = seg_v[i:].mean() + 1e-6, seg_v[:i].mean() + 1e-6
            if outer_s < inner_s * 0.8 and outer_v > inner_v * 1.2:
                ok, width = True, int(i)
            elif outer_s < inner_s * 0.9 and outer_v > inner_v:
                notes.append(f"fringe_uncertain:{edge}")
        detected.append(ok)
        strips.append(width)
    return detected, strips, notes


# ----------------------------------------------------------------------------- photometry

def gray_world(rgb: np.ndarray, mask: np.ndarray, clamp: tuple = (0.75, 1.35)) -> tuple[np.ndarray, list, float]:
    """Gray-world gains, clamped: a strongly coloured rug violates the gray-world assumption and an
    unclamped gain (e.g. x2.6 on green for a pink rug) destroys colour fidelity. Clamping is reported
    through wb.confidence (0 when a gain hit the clamp)."""
    f = rgb.astype(np.float32)
    means = np.array([f[..., c][mask].mean() for c in range(3)]) + 1e-6
    gains = means.mean() / means
    gains = np.clip(gains, clamp[0], clamp[1])
    out = f * gains[None, None, :]
    clipped = float((out > 255).any(axis=2)[mask].mean())
    return np.clip(out, 0, 255).astype(np.uint8), gains.tolist(), clipped


def _poly_terms(x, y, deg):
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=-1)


def flatten_illumination(rgb: np.ndarray, deg: int = 2) -> tuple[np.ndarray, dict]:
    """Multiplicative low-frequency flattening of Lab L (log space, 2nd degree bivariate polynomial,
    Huber loss, fitted outside dilated edges). Mean L is preserved; a/b untouched."""
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    H, W = L.shape
    g = guided(rgb, radius=8, eps=0.01)
    gl = np.clip(rgb_to_lab(g)[..., 0] * 2.55, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gl, 40, 100)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
    fit_mask = ~edges
    # subsample for the fit
    ys, xs = np.mgrid[0:H, 0:W]
    yn, xn = ys / max(1, H - 1) - 0.5, xs / max(1, W - 1) - 0.5
    step = max(1, int(np.sqrt(H * W / 20000)))
    sel = fit_mask[::step, ::step].ravel()
    logL = np.log(np.maximum(L, 1.0))
    A = _poly_terms(xn[::step, ::step].ravel()[sel], yn[::step, ::step].ravel()[sel], deg)
    b = logL[::step, ::step].ravel()[sel]
    if len(b) < A.shape[1] * 3:
        return rgb, {"range_L": 0.0, "residual": 0.0, "flattened": False}
    res = least_squares(lambda c: A @ c - b, x0=np.zeros(A.shape[1]), loss="huber", f_scale=0.1)
    coef = res.x
    field = (_poly_terms(xn.ravel(), yn.ravel(), deg) @ coef).reshape(H, W)
    field = field - field.mean()  # preserve mean L
    L_flat = np.exp(logL - field)
    L_flat = np.clip(L_flat, 0, 100)
    illum_range = float(np.exp(field).max() / np.exp(field).min())
    range_L = float((np.exp(field.max()) - np.exp(field.min())) * L.mean())
    out = lab.copy()
    out[..., 0] = L_flat
    # residual: gradient of a second fit on the flattened image
    logL2 = np.log(np.maximum(L_flat, 1.0))
    b2 = logL2[::step, ::step].ravel()[sel]
    res2 = least_squares(lambda c: A @ c - b2, x0=np.zeros(A.shape[1]), loss="huber", f_scale=0.1)
    field2 = (_poly_terms(xn.ravel(), yn.ravel(), deg) @ res2.x).reshape(H, W)
    residual = float((np.exp(field2.max()) - np.exp(field2.min())) * L_flat.mean())
    return lab_to_rgb(out), {"range_L": range_L, "residual": residual, "flattened": True,
                             "illum_ratio": illum_range}


def contrast_image(rgb: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def edge_width_px(rgb: np.ndarray, max_samples: int = 4000) -> float:
    """Median 10%-90% transition width (px) along the gradient direction at Canny edge points."""
    L = rgb_to_lab(rgb)[..., 0].astype(np.float32)
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(L8, (3, 3), 0), 40, 100)
    ys, xs = np.nonzero(edges)
    if len(ys) == 0:
        return 1.0
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    rng = np.random.RandomState(0)
    if len(ys) > max_samples:
        sel = rng.choice(len(ys), max_samples, replace=False)
        ys, xs = ys[sel], xs[sel]
    H, W = L.shape
    r = 6 * PX_PER_KNOT   # +-6 knots: wide enough for a pile_blur transition (> 2 knots)
    ts = np.arange(-r, r + 1, dtype=np.float32)
    g = np.stack([gx[ys, xs], gy[ys, xs]], axis=1).astype(np.float32)
    n = np.linalg.norm(g, axis=1)
    ok = n > 1e-3
    if not ok.any():
        return 1.0
    d = g[ok] / n[ok, None]
    px = xs[ok, None].astype(np.float32) + ts[None, :] * d[:, 0:1]
    py = ys[ok, None].astype(np.float32) + ts[None, :] * d[:, 1:2]
    inside = (px.min(1) >= 0) & (py.min(1) >= 0) & (px.max(1) <= W - 1) & (py.max(1) <= H - 1)
    if not inside.any():
        return 1.0
    prof = cv2.remap(L, px[inside], py[inside], cv2.INTER_LINEAR)  # (N, 2r+1)
    lo, hi = prof.min(1), prof.max(1)
    keep = (hi - lo) >= 3
    prof, lo, hi = prof[keep], lo[keep], hi[keep]
    a = lo + 0.1 * (hi - lo)
    b = lo + 0.9 * (hi - lo)
    mid = (prof > a[:, None]) & (prof < b[:, None])
    n_s = mid.shape[1]
    c = n_s // 2
    # width of the transition run that contains (or is nearest to) the centre sample
    widths = np.zeros(len(mid), dtype=np.float32)
    for i in range(len(mid)):
        row = mid[i]
        if not row.any():
            widths[i] = 1.0
            continue
        if row[c]:
            j = c
        else:
            nz = np.nonzero(row)[0]
            j = int(nz[np.argmin(np.abs(nz - c))])
        l = j
        while l - 1 >= 0 and row[l - 1]:
            l -= 1
        r_ = j
        while r_ + 1 < n_s and row[r_ + 1]:
            r_ += 1
        widths[i] = float(r_ - l + 1)
    widths = widths.tolist()
    return float(np.median(widths)) if widths else 1.0


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("rectify")
    photo_png = job.stage_dir("intake") / "photo.png"
    key = job.cache_key("rectify", [photo_png], extra={"corners": job.corners_override})
    if job.cached(sd, key) and (sd / "rectified.png").exists():
        job.timings["rectify"] = 0.0
        return job.state

    P = job.params
    grid = job.state["grid"]
    W, H = grid["W"], grid["H"]
    photo = cv2.cvtColor(cv2.imread(str(photo_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = photo.shape[:2]

    # 1. corner source
    manual = job.corners_override is not None
    if manual:
        corners = order_corners(np.asarray(job.corners_override, dtype=np.float64))
        mask = quad_mask(photo.shape, corners)
    else:
        try:
            mask = sam_segment(photo)
        except JobError as e:
            if e.code == "sam_unavailable":
                job.warn("sam_unavailable")
            job.stop("needs_corners", e.message)
            return job.state
        # 2. largest component + fill holes
        mask = largest_component_filled(mask)
    if mask.mean() < 0.20:
        job.stop("needs_corners", f"mask area {mask.mean():.2f} < 0.20")
        return job.state

    # 3. coarse quad + temporary coarse rectification
    cq = coarse_quad(mask) if not manual else corners
    cw = int(max(np.linalg.norm(cq[1] - cq[0]), np.linalg.norm(cq[2] - cq[3])))
    ch = int(max(np.linalg.norm(cq[3] - cq[0]), np.linalg.norm(cq[2] - cq[1])))
    cw, ch = max(cw, 16), max(ch, 16)
    Hc = cv2.getPerspectiveTransform(cq.astype(np.float32),
                                     np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32))
    coarse = cv2.warpPerspective(photo, Hc, (cw, ch))

    # 4. fringe
    fringe = {"detected": [False] * 4, "strip_px": [0] * 4}
    if P.get("fringe_detect", True) and not manual:
        det, strips, notes = detect_fringe(coarse, float(P.get("fringe_max_frac", 0.15)))
        short_is_horizontal = cw >= ch  # short sides are top/bottom when the rug is wider than tall
        short_edges = (0, 1) if short_is_horizontal else (2, 3)
        long_edges = (2, 3) if short_is_horizontal else (0, 1)
        if any(det[i] for i in short_edges):
            # crop the coarse mask to the pile edge and bring it back to the original frame
            cm = np.ones((ch, cw), np.uint8)
            if det[0]: cm[:strips[0], :] = 0
            if det[1]: cm[ch - strips[1]:, :] = 0
            if det[2]: cm[:, :strips[2]] = 0
            if det[3]: cm[:, cw - strips[3]:] = 0
            back = cv2.warpPerspective(cm, np.linalg.inv(Hc), (pw, ph), flags=cv2.INTER_NEAREST) > 0
            mask = mask & back
            fringe = {"detected": det, "strip_px": strips}
            if any(det[i] for i in long_edges):
                job.warn("fringe_long_side")
        for n in notes:
            job.warn("fringe_uncertain")
    job.state["fringe"] = fringe

    # 5. partial view
    border = np.zeros_like(mask)
    border[:3, :] = border[-3:, :] = True
    border[:, :3] = border[:, -3:] = True
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cs, key=cv2.contourArea)[:, 0, :]
    touch = border[contour[:, 1], contour[:, 0]].mean() if len(contour) else 0.0
    partial = bool(touch > 0.10)
    job.state["partial_view"] = partial
    if partial:
        job.warn("partial_view")
        if not P.get("allow_partial", False):
            job.stop("partial_view_stop", f"mask touches image border ({touch:.0%} of perimeter)")
            return job.state

    # 6. corners
    if manual:
        weak = False
    else:
        corners, weak = corners_from_mask(mask)
        if weak:
            job.warn("corners_weak")

    # 7. planarity (outer)
    pdev = planarity_deviation(mask, corners) if not manual else 0.0
    job.state["non_planar"] = bool(pdev > float(P.get("planarity_warn", 0.02)))
    if job.state["non_planar"]:
        job.warn("non_planar")
    if pdev > float(P.get("planarity_stop", 0.05)):
        job.stop("non_planar_stop", f"planarity_dev={pdev:.3f}")
        return job.state

    # 8. geometric resolution
    src_w = min(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    src_h = min(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))
    ppk = float(min(src_w / W, src_h / H))
    grid["photo_px_per_knot"] = ppk
    grid["upsampled"] = bool(ppk < PX_PER_KNOT)

    # 9. homography to 4W x 4H
    dst = np.array([[0, 0], [PX_PER_KNOT * W, 0], [PX_PER_KNOT * W, PX_PER_KNOT * H], [0, PX_PER_KNOT * H]], np.float32)
    Hm = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    interp = cv2.INTER_AREA if ppk >= PX_PER_KNOT else cv2.INTER_CUBIC
    rect = cv2.warpPerspective(photo, Hm, (PX_PER_KNOT * W, PX_PER_KNOT * H), flags=interp)
    full = np.ones(rect.shape[:2], dtype=bool)

    # 10. white balance
    rect, gains, clipped = gray_world(rect, full)
    clamped = bool(min(gains) <= 0.751 or max(gains) >= 1.349)
    job.state["wb"] = {"gains": gains, "clipped_pct": clipped, "clamped": clamped,
                       "confidence": 0.0 if clamped else float(1.0 - min(1.0, abs(np.log(max(gains) / min(gains))) * 2))}
    if clamped:
        job.warn("wb_clamped")

    # 11. illumination flattening
    rect_flat, illum = flatten_illumination(rect, deg=int(P.get("illum_poly_deg", 2)))
    if illum["range_L"] > float(P.get("illum_warn", 15.0)):
        job.warn("strong_illumination")
    job.state["illumination"] = illum
    rect_contrast = contrast_image(rect_flat)

    cv2.imwrite(str(sd / "rectified.png"), cv2.cvtColor(rect_flat, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sd / "rectified_contrast.png"), cv2.cvtColor(rect_contrast, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sd / "mask.png"), mask.astype(np.uint8) * 255)
    dump_json(sd / "corners.json", {"corners": corners.tolist(), "weak": weak, "manual": manual,
                                    "H": Hm.tolist(), "planarity_dev": pdev})

    # 12. edge sharpness / effective resolution
    ew_px = edge_width_px(rect_flat)
    ew_knots = ew_px / PX_PER_KNOT
    eff = ppk / max(1.0, ew_knots)
    grid["edge_width_knots"] = float(ew_knots)
    grid["photo_px_per_knot_eff"] = float(eff)
    if ew_knots > float(P.get("pile_blur_knots", 2.0)):
        job.warn("pile_blur")
    if eff < float(P.get("photo_px_stop", 0.8)):
        if P.get("allow_low_res", False):
            job.warn("forced_low_res")
        else:
            job.stop("low_res_stop", f"photo_px_per_knot_eff={eff:.2f}")
            return job.state
    if eff < float(P.get("photo_px_warn", 1.5)):
        job.warn("low_photo_resolution")

    job.state["grid"] = grid
    job.state.setdefault("stages_ran", []).append("rectify")
    job.timings["rectify"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
