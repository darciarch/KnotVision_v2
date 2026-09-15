"""production: quantization to the palette at grid resolution, noise measurement, edit added/missing
vs ref_indexed, region cleanup (guard), is_thin, coverage merge (guard), reporting.

Input: design_rgb.png | design_lab.npy + boundary.npy, design_lab_smooth.npy (edit/gradient),
       palette.json, gradient_mask.npy, ref_indexed.npy (edit), job.json
Output: indexed_raw.npy, design_lab_smooth.npy (if gradient), production_stats.json, stages/guard.json
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import PX_PER_KNOT
from .color import delta_e2000, nearest_palette_index, rgb_to_lab
from .guard import assert_no_motif_loss
from .job import Job, dump_json, read_json
from .metrics import component_area_map, connected_components, edit_added_missing, guided, is_thin, noise_stats


# ----------------------------------------------------------------------------- quantization

def mode_pool(idx_px: np.ndarray, dist_px: np.ndarray, W: int, H: int, k: int, block: int = PX_PER_KNOT) -> np.ndarray:
    """4x4 mode pooling. Ties: the index of the pixel nearest the block centre... then lowest index.
    Implemented as: count per index; among max-count indices prefer the one with the smallest deltaE
    at the centre pixels; final tie -> lowest index (argmax picks the first)."""
    blocks = idx_px.reshape(H, block, W, block).transpose(0, 2, 1, 3).reshape(H, W, block * block)
    dblocks = dist_px.reshape(H, block, W, block).transpose(0, 2, 1, 3).reshape(H, W, block * block)
    counts = np.zeros((H, W, k), dtype=np.int32)
    for j in range(block * block):
        np.add.at(counts, (np.arange(H)[:, None], np.arange(W)[None, :], blocks[..., j]), 1)
    maxc = counts.max(axis=2, keepdims=True)
    is_max = counts == maxc
    n_max = is_max.sum(axis=2)
    out = counts.argmax(axis=2).astype(np.int32)
    tie = n_max > 1
    if tie.any():
        # centre pixels: the 4 central cells of the 4x4 block
        centre = [5, 6, 9, 10] if block == 4 else list(range(block * block))
        ys, xs = np.nonzero(tie)
        for y, x in zip(ys, xs):
            cands = np.nonzero(is_max[y, x])[0]
            best, best_d = cands[0], np.inf
            for c in cands:
                sel = [j for j in centre if blocks[y, x, j] == c]
                d = dblocks[y, x, sel].min() if sel else np.inf
                if d < best_d:
                    best, best_d = c, d
            out[y, x] = best
    return out


def quick_quantize(rgb: np.ndarray, palette_lab: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """quantize/edit path: nearest entry per pixel (deltaE2000) + 4x4 mode pooling.
    Returns (indexed (H, W), per-knot deltaE of the pooled index vs the knot's mean Lab)."""
    if rgb.shape[0] != H * PX_PER_KNOT or rgb.shape[1] != W * PX_PER_KNOT:
        rgb = cv2.resize(rgb, (W * PX_PER_KNOT, H * PX_PER_KNOT), interpolation=cv2.INTER_AREA)
    lab = rgb_to_lab(rgb)
    idx, dist = nearest_palette_index(lab.reshape(-1, 3), palette_lab)
    idx = idx.reshape(rgb.shape[:2])
    dist = dist.reshape(rgb.shape[:2])
    indexed = mode_pool(idx, dist, W, H, len(palette_lab))
    knot_lab = cv2.resize(lab.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    de = delta_e2000(knot_lab.reshape(-1, 3), palette_lab[indexed.reshape(-1)]).reshape(H, W)
    return indexed, de


def segment_quantize(design_lab: np.ndarray, boundary: np.ndarray, palette_lab: np.ndarray,
                     boundary_thr: float, snap_r: int, gmask: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Two passes. Pass 1: boundary <= thr -> nearest entry. Pass 2: boundary knots -> nearest of the
    pass-1 entries within snap_r (never creates a third entry). Inside the gradient mask: nearest entry."""
    H, W = boundary.shape
    idx, _ = nearest_palette_index(design_lab.reshape(-1, 3), palette_lab)
    idx = idx.reshape(H, W)
    interior = boundary <= boundary_thr
    if gmask is not None:
        interior |= gmask
    out = idx.copy()
    ys, xs = np.nonzero(~interior)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - snap_r), min(H, y + snap_r + 1)
        x0, x1 = max(0, x - snap_r), min(W, x + snap_r + 1)
        cand = np.unique(idx[y0:y1, x0:x1][interior[y0:y1, x0:x1]])
        if len(cand) == 0:
            continue
        d = delta_e2000(design_lab[y, x][None, :], palette_lab[cand])
        out[y, x] = cand[int(np.argmin(d))]
    de = delta_e2000(design_lab.reshape(-1, 3), palette_lab[out.reshape(-1)]).reshape(H, W)
    return out, de


# ----------------------------------------------------------------------------- cleanup / merge

def _neighbor_by_boundary(indexed: np.ndarray, mask: np.ndarray) -> Optional[int]:
    """Index of the neighbour sharing the longest boundary with `mask` (lowest index on ties)."""
    dil = ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(2, 1)) & ~mask
    if not dil.any():
        return None
    vals = indexed[dil]
    counts = np.bincount(vals)
    return int(np.argmax(counts))


def cleanup(indexed: np.ndarray, min_region_knots: int, exempt: Optional[np.ndarray] = None,
            rounds: int = 3) -> tuple[np.ndarray, float]:
    """Components with area <= min_region_knots are merged into the neighbour sharing the longest
    boundary. `exempt` cells are never changed. Returns (indexed', changed_pct)."""
    if min_region_knots <= 0:
        return indexed.copy(), 0.0
    out = indexed.copy()
    for _ in range(rounds):
        area = component_area_map(out)
        small = (area <= min_region_knots)
        if exempt is not None:
            small &= ~exempt
        if not small.any():
            break
        changed_any = False
        for c in connected_components(out, small):
            if c.area > min_region_knots:
                continue
            fm = c.full_mask(out.shape)
            nb = _neighbor_by_boundary(out, fm)
            if nb is None or nb == c.index:
                continue
            out[fm] = nb
            changed_any = True
        if not changed_any:
            break
    return out, float((out != indexed).mean())


def coverage_merge(indexed: np.ndarray, palette_lab: np.ndarray, min_coverage_pct: float,
                   entries: list, params: dict, exempt_cells: Optional[np.ndarray] = None) -> tuple[np.ndarray, list]:
    """Colours with coverage < min_coverage_pct (percent) are merged into the nearest entry.
    Exempt: colours made of is_thin components; motif-colour criterion; gradient endpoints."""
    if min_coverage_pct <= 0:
        return indexed.copy(), []
    out = indexed.copy()
    H, W = out.shape
    total = out.size
    merged = []
    lmk, lmw = int(params.get("line_min_knots", 6)), int(params.get("line_max_width", 1))
    mmk, mcc = int(params.get("motif_min_knots", 10)), int(params.get("motif_color_min_cc", 5))
    for e in entries:
        i = e["index"]
        m = out == i
        cov = m.sum() / total * 100
        if cov == 0 or cov >= min_coverage_pct:
            continue
        if e.get("gradient_endpoint"):
            continue
        comps = connected_components(out, m)
        comps = [c for c in comps if c.index == i]
        if any(is_thin(c.full_mask(out.shape), lmk, lmw) for c in comps):
            continue
        if len(comps) >= mcc or any(c.area >= mmk for c in comps):
            continue
        others = [j for j in range(len(palette_lab)) if j != i and (out == j).any()]
        if not others:
            continue
        d = delta_e2000(palette_lab[i][None, :], palette_lab[others])
        target = others[int(np.argmin(d))]
        cells = m if exempt_cells is None else (m & ~exempt_cells)
        out[cells] = target
        merged.append({"from": i, "to": target, "coverage_pct": float(cov)})
    return out, merged


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("production")
    dd, pd, st = job.stage_dir("design"), job.stage_dir("palette"), job.stage_dir("structure")
    mode = job.state.get("design_mode", job.params.get("design_mode", "quantize"))
    inputs = [pd / "palette.json", st / "gradient_mask.npy", dd / "design_rgb.png"]
    for f in ("design_lab.npy", "boundary.npy", "design_lab_smooth.npy"):
        if (dd / f).exists():
            inputs.append(dd / f)
    ref_path = dd / "baseline" / "ref_indexed.npy"
    if mode == "edit" and ref_path.exists():
        inputs.append(ref_path)
    key = job.cache_key("production", inputs, extra={"mode": mode})
    if job.cached(sd, key) and (sd / "indexed_raw.npy").exists():
        job.timings["production"] = 0.0
        return job.state

    P = job.params
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    pal = read_json(pd / "palette.json")
    entries = pal["entries"]
    palette_lab = np.array([e["lab"] for e in entries], dtype=np.float64)
    gmask = np.load(st / "gradient_mask.npy")
    has_gradient = bool(gmask.any())
    guard_json = job.stages_dir / "guard.json"
    # reset guard entries for production (rerun safety)
    from .guard import reset
    reset(guard_json, keep_stages=())

    # 2. quantization
    if mode == "segment" and (dd / "design_lab.npy").exists():
        design_lab = np.load(dd / "design_lab.npy")
        boundary = np.load(dd / "boundary.npy")
        indexed, de = segment_quantize(design_lab, boundary, palette_lab, float(P.get("boundary_thr", 0.5)),
                                       int(P.get("boundary_snap_r", 2)), gmask if has_gradient else None)
        lab_smooth = design_lab
        exempt_boundary = boundary > 0.5
    else:
        rgb = cv2.cvtColor(cv2.imread(str(dd / "design_rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        indexed, de = quick_quantize(rgb, palette_lab, W, H)
        if (dd / "design_lab_smooth.npy").exists():
            lab_smooth = np.load(dd / "design_lab_smooth.npy")
        else:
            g = guided(rgb, radius=8, eps=0.01)
            lab_smooth = cv2.resize(rgb_to_lab(g).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        exempt_boundary = None
    if has_gradient:
        np.save(sd / "design_lab_smooth.npy", lab_smooth.astype(np.float32))

    # 3. noise before cleanup (+ edit added/missing, final values)
    valid = ~gmask
    ns = noise_stats(indexed, valid, int(P.get("noise_cc_min", 2)), int(P.get("noise_cc_max", 6)),
                     int(P.get("min_region_knots", 2)))
    stats = {"mode": mode, "noise_knots_pct": ns["noise_knots_frac"], "noise_1knot_pct": ns["noise_1knot_frac"],
             "noise_frac": ns["noise_frac"]}
    if job.design_class == "distressed":
        stats["noise_knots_pct"] = None
    if mode == "edit" and ref_path.exists():
        ref = np.load(ref_path)
        a, m = edit_added_missing(indexed, ref, int(P.get("motif_min_knots", 10)), gmask if has_gradient else None)
        stats["edit_added_pct"], stats["edit_missing_pct"] = a, m
        job.state.setdefault("edit", {})
        if job.state["edit"]:
            job.state["edit"]["added_pct"], job.state["edit"]["missing_pct"] = a, m

    # 4. cleanup (guard)
    mrk = int(P.get("min_region_knots", 2))
    exempt = gmask.copy() if has_gradient else np.zeros_like(gmask)
    if exempt_boundary is not None:
        exempt |= exempt_boundary
    gp = dict(P)
    gp["design_class"] = job.design_class
    cleaned, changed = cleanup(indexed, mrk, exempt)
    cleaned = assert_no_motif_loss(indexed, cleaned, gp, "production_cleanup", gmask if has_gradient else None, guard_json)
    stats["cleanup_changed_pct"] = None if job.design_class == "distressed" else float(changed)

    # 6. coverage merge (guard)
    merged, merges = coverage_merge(cleaned, palette_lab, float(P.get("min_coverage_pct", 0.5)), entries, P,
                                    exempt_cells=gmask if has_gradient else None)
    merged = assert_no_motif_loss(cleaned, merged, gp, "production_merge", gmask if has_gradient else None, guard_json)
    stats["coverage_merges"] = merges

    # 7. report
    per_entry = []
    for e in entries:
        m = merged == e["index"]
        per_entry.append({"index": e["index"], "knots": int(m.sum()), "coverage_pct": float(m.mean() * 100),
                          "quantization_deltaE": float(de[m].mean()) if m.any() else None})
    stats["entries"] = per_entry
    stats["palette_deltaE_p50"] = float(np.percentile(de, 50))
    stats["palette_deltaE_p95"] = float(np.percentile(de, 95))
    stats["palette_model_mismatch"] = bool(stats["palette_deltaE_p95"] > float(P.get("palette_model_mismatch_warn", 12.0)))
    if stats["palette_model_mismatch"]:
        job.warn("palette_model_mismatch")
    if any((pe["quantization_deltaE"] or 0) > float(P.get("deltaE_warn", 3.0)) for pe in per_entry):
        job.warn("palette_far")
    stats["palette_used"] = int(len(np.unique(merged)))
    job.state["colors"]["palette_used"] = stats["palette_used"]

    np.save(sd / "indexed_raw.npy", merged.astype(np.int32))
    np.save(sd / "indexed_prequant_deltaE.npy", de.astype(np.float32))
    dump_json(sd / "production_stats.json", stats)
    job.state["production"] = {k: v for k, v in stats.items() if k not in ("entries", "coverage_merges")}
    job.state.setdefault("stages_ran", []).append("production")
    job.timings["production"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
