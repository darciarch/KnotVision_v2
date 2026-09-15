"""Shared measurements. Single home for is_thin, edge_fidelity, component matching (cc_*),
line_jaggedness, repeat_mismatch, banding, noise_fidelity, thin_line_recall, edit added/missing.

Conventions: index maps are (H, W) int arrays at grid resolution (one cell = one knot).
Masks are boolean (H, W); "inside the gradient mask" cells are excluded where the spec says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .color import rgb_to_lab


# ----------------------------------------------------------------------------- components

@dataclass
class Component:
    label: int
    index: int
    area: int
    bbox: tuple           # (y0, y1, x0, x1), exclusive upper
    mask: np.ndarray      # bool (bbox-local)

    def full_mask(self, shape) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        y0, y1, x0, x1 = self.bbox
        m[y0:y1, x0:x1] = self.mask
        return m


def connected_components(indexed: np.ndarray, valid: Optional[np.ndarray] = None,
                         connectivity: int = 1) -> list[Component]:
    """4-connected (connectivity=1) components per index value. `valid` excludes cells."""
    comps = []
    struct = ndi.generate_binary_structure(2, connectivity)
    vals = np.unique(indexed[valid] if valid is not None else indexed)
    for v in vals:
        m = indexed == v
        if valid is not None:
            m &= valid
        lab, n = ndi.label(m, structure=struct)
        if n == 0:
            continue
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            sub = lab[sl] == i
            comps.append(Component(label=i, index=int(v), area=int(sub.sum()),
                                   bbox=(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop), mask=sub))
    return comps


def component_area_map(indexed: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Per-cell area of the component the cell belongs to (same index, 4-connected)."""
    out = np.zeros(indexed.shape, dtype=np.int32)
    struct = ndi.generate_binary_structure(2, connectivity)
    for v in np.unique(indexed):
        m = indexed == v
        lab, n = ndi.label(m, structure=struct)
        if n == 0:
            continue
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1)).astype(np.int32)
        out[m] = sizes[lab[m] - 1]
    return out


# ----------------------------------------------------------------------------- is_thin

def is_thin(mask: np.ndarray, line_min_knots: int = 6, line_max_width: int = 1) -> bool:
    """Skeleton length >= line_min_knots AND max local width <= line_max_width.

    Width rule: a component is at most w knots wide iff it contains no fully foreground
    (w+1) x (w+1) block (exact for axis-aligned runs, tight for diagonals). Single home of this rule
    (production, structure, tools/motif_stats.py all call it).
    """
    m = np.asarray(mask, dtype=bool)
    if m.sum() < line_min_knots:
        return False
    k = int(line_max_width) + 1
    if ndi.binary_erosion(m, structure=np.ones((k, k), bool), border_value=0).any():
        return False
    sk = skeletonize(m)
    return bool(int(sk.sum()) >= line_min_knots)


# ----------------------------------------------------------------------------- edges

def _lab_L(img_rgb: np.ndarray) -> np.ndarray:
    return rgb_to_lab(img_rgb).astype(np.float32)[..., 0]


def guided(img: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    """Guided filter (cv2.ximgproc) with the image as its own guide. Works on uint8 RGB or float."""
    src = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    out = cv2.ximgproc.guidedFilter(guide=src, src=src, radius=radius, eps=eps)
    if img.dtype == np.uint8:
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def canny_L(img_rgb: np.ndarray, radius: int = 8, lo: int = 40, hi: int = 100) -> np.ndarray:
    """guided filter (r=8) + Lab L Canny -> bool edge map."""
    g = guided(img_rgb, radius=radius, eps=0.01)
    L = _lab_L(g)
    L8 = np.clip(L * 255.0 / 100.0, 0, 255).astype(np.uint8)
    return cv2.Canny(L8, lo, hi) > 0


def edge_density(img_rgb: np.ndarray) -> float:
    e = canny_L(img_rgb)
    return float(e.mean())


def edge_fidelity(a_rgb: np.ndarray, b_rgb: np.ndarray, tol_px: int = 2) -> float:
    """Edge IoU with `tol_px` tolerance between two RGB images of the same size.

    IoU = |A ∩ dilate(B)| ∪ ... implemented symmetrically: matched = A∩dil(B) + B∩dil(A),
    union = |A| + |B|. Single home (qc.rerender_edge_iou, design_edit, train/align).
    """
    if a_rgb.shape[:2] != b_rgb.shape[:2]:
        b_rgb = cv2.resize(b_rgb, (a_rgb.shape[1], a_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    ea = canny_L(a_rgb)
    eb = canny_L(b_rgb)
    if ea.sum() + eb.sum() == 0:
        return 1.0
    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), k) > 0
    db = cv2.dilate(eb.astype(np.uint8), k) > 0
    matched = (ea & db).sum() + (eb & da).sum()
    return float(matched / (ea.sum() + eb.sum()))


def boundary_map(indexed: np.ndarray) -> np.ndarray:
    """Cells whose right or bottom neighbour has a different index (grid-resolution boundary)."""
    b = np.zeros(indexed.shape, dtype=bool)
    b[:, :-1] |= indexed[:, :-1] != indexed[:, 1:]
    b[:-1, :] |= indexed[:-1, :] != indexed[1:, :]
    return b


def edge_iou_indexed(a: np.ndarray, b: np.ndarray, tol_knots: int = 1) -> float:
    """Boundary IoU between two index maps with 1-knot tolerance (eval `edge_iou`)."""
    ea, eb = boundary_map(a), boundary_map(b)
    if ea.sum() + eb.sum() == 0:
        return 1.0
    k = np.ones((2 * tol_knots + 1, 2 * tol_knots + 1), np.uint8)
    da = cv2.dilate(ea.astype(np.uint8), k) > 0
    db = cv2.dilate(eb.astype(np.uint8), k) > 0
    return float(((ea & db).sum() + (eb & da).sum()) / (ea.sum() + eb.sum()))


# ----------------------------------------------------------------------------- jaggedness

def _boundary_chains(indexed: np.ndarray, valid: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Boundary polylines (cv2.findContours on each index mask, CHAIN_APPROX_NONE)."""
    chains = []
    for v in np.unique(indexed):
        m = (indexed == v)
        if valid is not None:
            m &= valid
        if m.sum() < 4:
            continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) >= 8:
                chains.append(c[:, 0, :])
    return chains


def _run_lengths(chain: np.ndarray) -> np.ndarray:
    """Run lengths of constant step direction along a chain."""
    d = np.diff(chain, axis=0)
    if len(d) == 0:
        return np.array([], dtype=int)
    code = d[:, 0] * 3 + d[:, 1]
    runs, cur = [], 1
    for i in range(1, len(code)):
        if code[i] == code[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.asarray(runs)


def line_jaggedness(indexed: np.ndarray, valid: Optional[np.ndarray] = None, window: int = 16) -> float:
    """Run-length entropy of boundary chains in 16-knot windows, averaged over all windows.

    A straight / clean-stepped boundary has few distinct run lengths per window (low entropy);
    a jagged one has many. Normalised by log(window) -> [0, 1].
    """
    ents, weights = [], []
    for ch in _boundary_chains(indexed, valid):
        for s in range(0, max(1, len(ch) - window + 1), window):
            seg = ch[s:s + window + 1]
            rl = _run_lengths(seg)
            if len(rl) == 0:
                continue
            _, counts = np.unique(rl, return_counts=True)
            p = counts / counts.sum()
            ents.append(float(-(p * np.log(p)).sum() / np.log(window)))
            weights.append(len(seg))
    if not ents:
        return 0.0
    return float(np.average(ents, weights=weights))


# ----------------------------------------------------------------------------- component matching

def _match_components(before: list[Component], after: np.ndarray, shape, iou_thr: float = 0.3) -> list[bool]:
    """For each `before` component: is there an `after` component with the same index, IoU >= thr?
    Implemented as: within bbox, the same-index cells of `after` restricted to the connected
    piece(s) overlapping the component — cheap approximation that is exact for the pass/fail."""
    matched = []
    struct = ndi.generate_binary_structure(2, 1)
    for c in before:
        y0, y1, x0, x1 = c.bbox
        pad = 2
        Y0, Y1, X0, X1 = max(0, y0 - pad), min(shape[0], y1 + pad), max(0, x0 - pad), min(shape[1], x1 + pad)
        cm = c.full_mask(shape)[Y0:Y1, X0:X1]
        am = after[Y0:Y1, X0:X1] == c.index
        if not am.any():
            matched.append(False)
            continue
        lab, n = ndi.label(am, structure=struct)
        best = 0.0
        for i in range(1, n + 1):
            piece = lab == i
            inter = (piece & cm).sum()
            if inter == 0:
                continue
            union = (piece | cm).sum()
            best = max(best, inter / union)
        matched.append(best >= iou_thr)
    return matched


def cc_added_missing(output: np.ndarray, truth: np.ndarray, motif_min_knots: int,
                     exclude: Optional[np.ndarray] = None, line_min_knots: int = 6,
                     line_max_width: int = 1, iou_thr: float = 0.3) -> dict:
    """Eval metrics against a reference index map (truth or the quantize baseline).

    cc_added_area_pct  : knot share of output components with no match in truth.
    cc_added_motif     : number of such components with area > motif_min_knots and not is_thin.
    cc_missing_pct     : knot share of truth components with no match in output.
    added_pct/missing_pct (aliases) are what design_edit / production use vs ref_indexed, with the
    extra rule "area > motif_min_knots" applied to both directions.
    """
    valid = np.ones(output.shape, dtype=bool) if exclude is None else ~exclude
    total = max(1, int(valid.sum()))
    out_c = connected_components(output, valid)
    tru_c = connected_components(truth, valid)
    out_ok = _match_components(out_c, truth, output.shape, iou_thr)
    tru_ok = _match_components(tru_c, output, output.shape, iou_thr)
    added_area = sum(c.area for c, ok in zip(out_c, out_ok) if not ok)
    added_motif = sum(1 for c, ok in zip(out_c, out_ok)
                      if not ok and c.area > motif_min_knots
                      and not is_thin(c.full_mask(output.shape), line_min_knots, line_max_width))
    missing_area = sum(c.area for c, ok in zip(tru_c, tru_ok) if not ok)
    added_big = sum(c.area for c, ok in zip(out_c, out_ok) if not ok and c.area > motif_min_knots)
    missing_big = sum(c.area for c, ok in zip(tru_c, tru_ok) if not ok and c.area > motif_min_knots)
    return {
        "cc_added_area_pct": added_area / total,
        "cc_added_motif": int(added_motif),
        "cc_missing_pct": missing_area / total,
        "added_pct": added_big / total,
        "missing_pct": missing_big / total,
    }


def edit_added_missing(indexed_cand: np.ndarray, ref_indexed: np.ndarray, motif_min_knots: int,
                       gradient_mask: Optional[np.ndarray] = None) -> tuple[float, float]:
    """edit_added_pct, edit_missing_pct vs the quantize baseline (same index, IoU < 0.3,
    area > motif_min_knots, outside the gradient mask)."""
    r = cc_added_missing(indexed_cand, ref_indexed, motif_min_knots, exclude=gradient_mask)
    return float(r["added_pct"]), float(r["missing_pct"])


def thin_line_recall(output: np.ndarray, truth: np.ndarray, line_min_knots: int = 6,
                     line_max_width: int = 1) -> Optional[float]:
    """Share of truth is_thin components that survive (same index, skeleton overlap >= 0.7)."""
    tot, kept = 0, 0
    for c in connected_components(truth):
        fm = c.full_mask(truth.shape)
        if not is_thin(fm, line_min_knots, line_max_width):
            continue
        tot += 1
        sk = skeletonize(fm)
        if sk.sum() == 0:
            continue
        near = cv2.dilate((output == c.index).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if (sk & near).sum() / sk.sum() >= 0.7:
            kept += 1
    return None if tot == 0 else kept / tot


# ----------------------------------------------------------------------------- noise

def noise_stats(indexed: np.ndarray, valid: Optional[np.ndarray], cc_min: int, cc_max: int,
                min_region_knots: int) -> dict:
    """noise_frac (area in [cc_min, cc_max]), noise_1knot_frac (area == 1), noise_knots_frac (area <= min_region)."""
    area = component_area_map(indexed)
    v = np.ones(indexed.shape, dtype=bool) if valid is None else valid
    total = max(1, int(v.sum()))
    return {
        "noise_frac": float(((area >= cc_min) & (area <= cc_max) & v).sum() / total),
        "noise_1knot_frac": float(((area == 1) & v).sum() / total),
        "noise_knots_frac": float(((area <= max(0, min_region_knots)) & (area >= 1) & v).sum() / total)
        if min_region_knots > 0 else 0.0,
    }


def noise_fidelity(output: np.ndarray, truth: np.ndarray, noise_mask: np.ndarray, cc_max: int = 6) -> dict:
    """Inside truth noise_mask: index agreement x min(r, 1/r) where r = small-component count ratio."""
    if noise_mask is None or noise_mask.sum() == 0:
        return {"noise_fidelity": None, "noise_index_acc": None, "noise_cc_ratio": None}
    acc = float((output[noise_mask] == truth[noise_mask]).mean())

    def small_cc(ix):
        area = component_area_map(ix)
        n = 0
        for c in connected_components(ix, noise_mask):
            if c.area <= cc_max:
                n += 1
        return n

    n_out, n_tru = small_cc(output), small_cc(truth)
    r = (n_out / n_tru) if n_tru > 0 else (np.inf if n_out > 0 else 1.0)
    f = acc * (min(r, 1 / r) if r > 0 and np.isfinite(r) else 0.0)
    return {"noise_fidelity": float(f), "noise_index_acc": acc, "noise_cc_ratio": float(r) if np.isfinite(r) else None}


# ----------------------------------------------------------------------------- repeat / symmetry

def repeat_mismatch(indexed: np.ndarray, v1: tuple, v2: Optional[tuple], region: Optional[np.ndarray] = None) -> Optional[float]:
    """Share of cells (inside region) whose translated copy by v1 (and v2) has a different index."""
    H, W = indexed.shape
    total, bad = 0, 0
    for v in (v1, v2):
        if v is None:
            continue
        dy, dx = int(round(v[0])), int(round(v[1]))
        if dy == 0 and dx == 0:
            continue
        ys = slice(max(0, -dy), min(H, H - dy))
        xs = slice(max(0, -dx), min(W, W - dx))
        a = indexed[ys, xs]
        b = indexed[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx]
        m = np.ones(a.shape, dtype=bool) if region is None else region[ys, xs]
        total += int(m.sum())
        bad += int(((a != b) & m).sum())
    return None if total == 0 else bad / total


def symmetry_mismatch(indexed: np.ndarray, kind: str, region: Optional[np.ndarray] = None) -> Optional[float]:
    if kind in (None, "none"):
        return None
    m = np.ones(indexed.shape, dtype=bool) if region is None else region
    vals = []
    if kind in ("h", "hv"):
        vals.append(((indexed != indexed[:, ::-1]) & m).sum() / max(1, m.sum()))
    if kind in ("v", "hv"):
        vals.append(((indexed != indexed[::-1, :]) & m).sum() / max(1, m.sum()))
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------------- banding / dither

def banding(rerender_L: np.ndarray, rectified_L: np.ndarray, mask: np.ndarray, sigma_knots: float = 8.0,
            px_per_knot: float = 1.0) -> Optional[float]:
    """Low-frequency (σ = 8 knots) RMS of the L difference inside the gradient mask."""
    if mask is None or mask.sum() == 0:
        return None
    d = (rerender_L.astype(np.float32) - rectified_L.astype(np.float32))
    d[~mask] = 0.0
    s = sigma_knots * px_per_knot
    num = cv2.GaussianBlur(d, (0, 0), s)
    den = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), s)
    lf = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return float(np.sqrt((lf[mask] ** 2).mean()))


def dither_err(output: np.ndarray, truth: np.ndarray, mask: np.ndarray, block: int = 4) -> Optional[float]:
    """Mean abs difference of 4x4 block index-mixing ratios inside the mask."""
    if mask is None or mask.sum() == 0:
        return None
    H, W = output.shape
    errs = []
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            m = mask[y:y + block, x:x + block]
            if m.mean() < 0.5:
                continue
            o = output[y:y + block, x:x + block]
            t = truth[y:y + block, x:x + block]
            vals = np.union1d(o, t)
            ho = np.array([(o == v).mean() for v in vals])
            ht = np.array([(t == v).mean() for v in vals])
            errs.append(np.abs(ho - ht).sum() / 2)
    return float(np.mean(errs)) if errs else None


# ----------------------------------------------------------------------------- misc

def spearman(a, b) -> Optional[float]:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).correlation)


def derive_edit_gates(edit_added, cc_added_area, edit_missing, cc_missing, corr_min: float = 0.6,
                      added_ok: float = 0.002, missing_ok: float = 0.01) -> dict:
    """eval/run.py rule: if Spearman >= corr_min, gate = p95 of edit_* over jobs whose cc_* is ok."""
    ra = spearman(edit_added, cc_added_area)
    rm = spearman(edit_missing, cc_missing)
    out = {"corr_added": ra, "corr_missing": rm, "edit_gate_source": "assumed",
           "edit_added_max": None, "edit_missing_max": None}
    if ra is not None and rm is not None and ra >= corr_min and rm >= corr_min:
        ea = np.asarray(edit_added)[np.asarray(cc_added_area) <= added_ok]
        em = np.asarray(edit_missing)[np.asarray(cc_missing) <= missing_ok]
        if len(ea) and len(em):
            out.update({"edit_gate_source": "eval",
                        "edit_added_max": float(np.percentile(ea, 95)),
                        "edit_missing_max": float(np.percentile(em, 95))})
    return out
