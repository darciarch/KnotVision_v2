"""rectify: carpet mask (SAM 3.1 or manual corners), coarse quad + fringe, partial view, corners,
planarity, geometric resolution, homography to 4W x 4H, white balance, illumination flattening,
edge sharpness / effective resolution gates.

Output: rectified.png (colour-correct), rectified_contrast.png (CLAHE L, for structure/edges only),
        mask.png, corners.json, job.json updated (grid, fringe, illumination, wb, warnings)

Spec deviations:
- the outermost knot ring of the rectified image is replaced by its inner neighbour ring: at the pile
  edge the photo blends rug and background (blur, shadow), and that blend would otherwise become a
  one-knot frame in the design and a spurious thin line / colour in every downstream measurement.
- edge_width_knots is measured on colour (Lab) edges, on axis-aligned edges when enough exist (a
  diagonal staircase is wider than the optical blur), with sub-sample 10 % / 90 % crossings, and
  the median is taken over resolved profiles only (both ends on a plateau >= 1.5 knots): an
  unresolved motif hump under-reports the blur. Canny thresholds relax (40/100 -> 20/50 -> 10/25) until >= 200
  edge points exist, otherwise a heavily blurred photo has no edges to measure.
"""
from __future__ import annotations

import contextlib
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
from .metrics import canny_from_gradients, guided, lab_gradients

SAM_PROMPTS = ("carpet", "rug", "carpet on the floor")
PLATEAU_KNOTS = 1.5   # a resolved edge keeps a flat plateau this long on both sides (edge_width_px)


# ----------------------------------------------------------------------------- SAM 3.1 wrapper

_SAM_CACHE: dict = {}


def _sam_cpu_patches(torch):
    """sam3 0.1.0 assumes a CUDA device at inference time in two places; on a CPU-only machine both are
    neutralised once per process (idempotent):

    * the ViT MLP goes through perflib.fused.addmm_act, which casts to bfloat16 unconditionally (the
      surrounding CUDA autocast makes the next Linear accept that). Without autocast the fp32 fc2 rejects
      the bf16 input, so the fused op is replaced by the plain fp32 Linear + activation;
    * the geometry encoder calls Tensor.pin_memory(), which raises without an accelerator; pinning is a
      host-to-device transfer optimisation and a no-op is exact on the CPU.
    """
    if getattr(torch, "_knotvision_sam_cpu", False):
        return
    F = torch.nn.functional
    try:
        from sam3.model import vitdet  # type: ignore

        def addmm_act(activation, linear, mat1):
            y = F.linear(mat1, linear.weight, linear.bias)
            if activation in (F.relu, torch.nn.ReLU):
                return F.relu(y)
            if activation in (F.gelu, torch.nn.GELU):
                return F.gelu(y)
            raise ValueError(f"Unexpected activation {activation}")

        vitdet.addmm_act = addmm_act
    except Exception:  # pragma: no cover
        pass
    _orig_pin = torch.Tensor.pin_memory

    def pin_memory(self, *a, **k):
        if self.device.type == "cpu" and not torch.cuda.is_available():
            return self
        return _orig_pin(self, *a, **k)

    torch.Tensor.pin_memory = pin_memory  # type: ignore[method-assign]
    torch._knotvision_sam_cpu = True


@contextlib.contextmanager
def _sam_cpu_shim(torch, device: str):
    """sam3 0.1.0 builds two constant caches with a hard-coded device="cuda"
    (PositionEmbeddingSine precompute, decoder box-RPB coords). On a CPU-only machine those
    factory calls are redirected to the CPU for the duration of the model build; nothing else changes."""
    if device == "cuda":
        yield
        return
    orig = {name: getattr(torch, name) for name in ("zeros", "arange")}

    def _wrap(fn):
        def inner(*a, **k):
            if k.get("device") == "cuda":
                k["device"] = "cpu"
            return fn(*a, **k)
        return inner

    for name, fn in orig.items():
        setattr(torch, name, _wrap(fn))
    try:
        yield
    finally:
        for name, fn in orig.items():
            setattr(torch, name, fn)


def _sam_processor(min_score: float):
    """Builds (once per process) the SAM 3.1 image model + processor on the available device."""
    ckpt_dir = MODELS_DIR / "sam3.1"
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        raise JobError("sam_unavailable", f"no checkpoint (*.pt) in {ckpt_dir}")
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    except Exception as e:  # pragma: no cover - needs the real package
        raise JobError("sam_unavailable", f"sam3 import failed: {e}")
    key = (str(ckpts[0]), float(min_score))
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _sam_cpu_patches(torch)
    # load_from_HF=False: the checkpoint is local; never reach for the network here
    with _sam_cpu_shim(torch, device):
        model = build_sam3_image_model(checkpoint_path=str(ckpts[0]), device=device, load_from_HF=False)
    proc = Sam3Processor(model, device=device, confidence_threshold=min_score)
    _SAM_CACHE.clear()
    _SAM_CACHE[key] = proc
    return proc


def sam_segment(photo_rgb: np.ndarray, prompts=SAM_PROMPTS, min_score: float = 0.3) -> np.ndarray:
    """Text-prompted SAM 3.1 carpet mask. Selection: score x area, next prompt when score < min_score.

    Raises JobError("sam_unavailable") when the package or the checkpoint is missing or inference fails,
    JobError("needs_corners") when no prompt yields a mask above `min_score`.
    """
    proc = _sam_processor(min_score)
    try:  # pragma: no cover - needs weights
        import torch
        from PIL import Image
        with torch.inference_mode():
            state = proc.set_image(Image.fromarray(np.ascontiguousarray(photo_rgb)))
            best, best_val = None, -1.0
            for p in prompts:
                out = proc.set_text_prompt(prompt=p, state=state)
                masks = out["masks"].detach().cpu().numpy().reshape(-1, photo_rgb.shape[0], photo_rgb.shape[1])
                scores = out["scores"].detach().cpu().numpy().reshape(-1)
                for m, s in zip(masks, scores):
                    s = float(s)
                    if s < min_score:
                        continue
                    m = m.astype(bool)
                    val = s * m.mean()
                    if val > best_val:
                        best, best_val = m, val
                if best is not None:
                    break
                proc.reset_all_prompts(state)
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
    # a large dark (or light) region still bends the fit through Huber's linear tail: two rounds of
    # sigma clipping keep the fit on the dominant tone (the illumination is what the majority shares)
    for _ in range(2):
        resid = A @ coef - b
        mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826 + 1e-6
        keep = np.abs(resid) < 3.0 * mad
        if keep.sum() < A.shape[1] * 3 or keep.all():
            break
        res = least_squares(lambda c: A[keep] @ c - b[keep], x0=coef, loss="huber", f_scale=0.1)
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


def edge_width_px(rgb: np.ndarray, max_samples: int = 4000, noise_tol: float = 0.75, min_resolved: int = 20) -> float:
    """Median 10%-90% transition width (px) along the gradient direction at Canny edge points.

    Edges are colour edges: Canny runs on the Lab gradient (per pixel the channel with the strongest
    gradient, OpenCV's multi-channel rule) so that two colours of equal lightness (a common two-colour
    rug) still have measurable edges instead of only pile ripple. Each profile is the Lab profile
    projected onto the local colour-change direction, oriented to rise through its centre; only the
    monotone run containing the centre sample is measured (a neighbouring edge or the pile ripple
    ends the run). A run counts only when both ends reach a plateau (>= 2 samples within 10 % of the
    range and >= PLATEAU_KNOTS long): an unresolved small motif is a hump whose rise stops early
    and would report a width *narrower* than the blur. The statistic is the median of the resolved
    runs (spec: median).
    """
    # gradients of the lightly smoothed image for edge detection; profiles are sampled from `lab`
    lab, gx3, gy3 = lab_gradients(rgb)
    H, W = lab.shape[:2]
    # a heavily blurred photo has no edge above the default Canny thresholds: relax them until a
    # few hundred edge points exist (their width is what tells the blur)
    ys = xs = np.zeros(0, np.int64)
    for lo_t, hi_t in ((40, 100), (20, 50), (10, 25)):
        ys, xs = np.nonzero(canny_from_gradients(gx3, gy3, lo_t, hi_t))
        if len(ys) >= 200:
            break
    if len(ys) == 0:
        return 1.0
    # spatial gradient direction: the channel with the strongest gradient at the edge point
    gxe, gye = gx3[ys, xs], gy3[ys, xs]                       # (N, 3)
    mag = np.hypot(gxe, gye)
    ch = np.argmax(mag, axis=1)
    g = np.stack([gxe[np.arange(len(ys)), ch], gye[np.arange(len(ys)), ch]], axis=1)
    n = np.linalg.norm(g, axis=1)
    ok = n > 1e-3
    if not ok.any():
        return 1.0
    ys, xs, g, n = ys[ok], xs[ok], g[ok], n[ok]
    d = g / n[:, None]
    gxe, gye = gxe[ok], gye[ok]
    # a diagonal edge is a knot staircase whose profile is wider than the optical blur: measure on
    # axis-aligned edges (gradient within ~15 degrees of an axis) when there are enough of them
    axis = (np.abs(d[:, 0]) > 0.966) | (np.abs(d[:, 1]) > 0.966)
    if axis.sum() >= max(50, 0.05 * len(ys)):
        ys, xs, d, gxe, gye = ys[axis], xs[axis], d[axis], gxe[axis], gye[axis]
    rng = np.random.RandomState(0)
    if len(ys) > max_samples:
        sel = np.sort(rng.choice(len(ys), max_samples, replace=False))
        ys, xs, d, gxe, gye = ys[sel], xs[sel], d[sel], gxe[sel], gye[sel]
    # colour-change direction u (unit Lab vector): the directional derivative of Lab along d
    u = gxe * d[:, 0:1] + gye * d[:, 1:2]                     # (N, 3)
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-6)
    r = 6 * PX_PER_KNOT   # +-6 knots: wide enough for a pile_blur transition (> 2 knots)
    ts = np.arange(-r, r + 1, dtype=np.float32)
    px = xs[:, None].astype(np.float32) + ts[None, :] * d[:, 0:1]
    py = ys[:, None].astype(np.float32) + ts[None, :] * d[:, 1:2]
    inside = (px.min(1) >= 0) & (py.min(1) >= 0) & (px.max(1) <= W - 1) & (py.max(1) <= H - 1)
    if not inside.any():
        return 1.0
    px, py, u = px[inside], py[inside], u[inside]
    prof = np.zeros(px.shape, np.float32)                     # (N, 2r+1), sampled along +gradient
    for c in range(3):
        prof += cv2.remap(np.ascontiguousarray(lab[..., c]), px, py, cv2.INTER_LINEAR) * u[:, c:c + 1]
    # light smoothing along the profile (sigma 1 px) so that camera / JPEG noise does not end the
    # monotone run early on a blurred edge; its own width is removed in quadrature below
    prof = cv2.GaussianBlur(prof, (5, 1), 1.0)
    n_s = prof.shape[1]
    c = n_s // 2
    widths, resolved = [], []
    need = max(2, int(round(PLATEAU_KNOTS * PX_PER_KNOT)))
    for row in prof:
        # monotone (non-decreasing, noise_tol allowance) run through the centre
        l = c
        while l - 1 >= 0 and row[l - 1] <= row[l] + noise_tol:
            l -= 1
        rr = c
        while rr + 1 < n_s and row[rr + 1] >= row[rr] - noise_tol:
            rr += 1
        seg = row[l:rr + 1]
        lo, hi = float(seg.min()), float(seg.max())
        if hi - lo < 6.0:   # a real edge, not noise
            continue
        a, b = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
        # sub-sample crossings of the 10 % and 90 % levels (no +1 sample bias on sharp edges)
        ia = int(np.argmax(seg >= a))
        ib = int(np.argmax(seg >= b))
        ta = ia - 1 + (a - seg[ia - 1]) / max(1e-6, seg[ia] - seg[ia - 1]) if ia > 0 else 0.0
        tb = ib - 1 + (b - seg[ib - 1]) / max(1e-6, seg[ib] - seg[ib - 1]) if ib > 0 else 0.0
        w = max(0.5, float(tb - ta))
        widths.append(w)
        # a resolved step keeps a plateau of >= PLATEAU_KNOTS on both sides; the top of a Gaussian
        # hump (an unresolved motif) stays within 10 % of its peak for only ~0.9 sigma
        if (seg <= a).sum() >= need and (seg >= b).sum() >= need:
            resolved.append(w)
    if len(resolved) >= min_resolved:
        w = float(np.median(resolved))
    else:
        w = float(np.median(widths)) if widths else 1.0
    # remove the profile smoothing (10-90 % width of a sigma-1 Gaussian = 2.56 px) in quadrature
    return float(max(1.0, np.sqrt(max(0.0, w * w - 2.56 ** 2))))


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
    # outermost knot ring = pile edge blended with the background: take the inner neighbour ring
    k = PX_PER_KNOT
    rect[:k] = rect[k:2 * k]
    rect[-k:] = rect[-2 * k:-k]
    rect[:, :k] = rect[:, k:2 * k]
    rect[:, -k:] = rect[:, -2 * k:-k]
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
