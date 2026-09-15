"""dither: analytic ordered dither inside the gradient mask. Mode-independent, deterministic.

1. Per gradient region: ramp between the two endpoint indices (multi-segment if intermediate palette
   entries lie on the ramp in Lab).
2. Each knot's design_lab_smooth value is projected onto the ramp -> neighbouring pair (a, b), ratio t.
3. t is rounded to `levels`; the ordered matrix (config/dither/<name>.yaml) picks a or b.
4. Knots on the mask edge take the nearest endpoint (no dither spill into flat areas).
Output: indexed.npy (rewritten), report {applied, mask_pct, levels, pairs}
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from .color import delta_e2000
from .job import CONFIG_DIR, Job, dump_json, load_yaml, read_json


def load_pattern(name: Optional[str]) -> dict:
    if name is None:
        p = CONFIG_DIR / "dither" / "ordered4.yaml"
    else:
        p = Path(name) if str(name).endswith(".yaml") else CONFIG_DIR / "dither" / f"{name}.yaml"
    d = load_yaml(p)
    d["matrix"] = np.array(d["matrix"], dtype=np.float64)
    n = d["matrix"].size
    d["threshold"] = (d["matrix"] + 0.5) / n  # in (0, 1)
    return d


def ramp_indices(lab_lo: np.ndarray, lab_hi: np.ndarray, palette_lab: np.ndarray, endpoint_ids: tuple) -> list[int]:
    """[i0, ..., ik]: endpoint indices plus intermediate entries whose projection lies strictly inside
    the ramp and that are close to it (deltaE to the ramp line < 6). Ordered by projection."""
    i0, i1 = endpoint_ids
    d = lab_hi - lab_lo
    L2 = float(d @ d) + 1e-9
    inter = []
    for j in range(len(palette_lab)):
        if j in (i0, i1):
            continue
        t = float(((palette_lab[j] - lab_lo) @ d) / L2)
        if 0.1 < t < 0.9:
            proj = lab_lo + t * d
            if float(delta_e2000(palette_lab[j], proj)) < 6.0:
                inter.append((t, j))
    inter.sort()
    return [i0] + [j for _, j in inter] + [i1]


def apply_dither(indexed: np.ndarray, lab_smooth: np.ndarray, gmask: np.ndarray, regions: list,
                 palette: dict, pattern: dict, levels: int) -> tuple[np.ndarray, dict]:
    H, W = indexed.shape
    out = indexed.copy()
    entries = palette["entries"]
    palette_lab = np.array([e["lab"] for e in entries], dtype=np.float64)
    thr = pattern["threshold"]
    mh, mw = thr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    T = thr[yy % mh, xx % mw]
    pairs = set()
    # region labels of the mask
    lab_m, n = ndi.label(gmask)
    # mask edge: cells whose 4-neighbourhood leaves the mask
    inner = ndi.binary_erosion(gmask, structure=ndi.generate_binary_structure(2, 1), border_value=0)
    edge = gmask & ~inner
    for ri in range(1, n + 1):
        rm = lab_m == ri
        # find the region description that overlaps most (by bbox) / fallback: nearest endpoints by Lab
        ends = _endpoints_for(rm, regions, entries, palette_lab, lab_smooth)
        if ends is None:
            continue
        i0, i1 = ends
        chain = ramp_indices(palette_lab[i0], palette_lab[i1], palette_lab, (i0, i1))
        # project each knot onto the polyline chain
        ys, xs = np.nonzero(rm)
        vals = lab_smooth[ys, xs].astype(np.float64)
        best_t = np.zeros(len(ys))
        best_seg = np.zeros(len(ys), dtype=int)
        best_d = np.full(len(ys), np.inf)
        for s in range(len(chain) - 1):
            a, b = palette_lab[chain[s]], palette_lab[chain[s + 1]]
            d = b - a
            L2 = float(d @ d) + 1e-9
            t = np.clip(((vals - a) @ d) / L2, 0.0, 1.0)
            proj = a + t[:, None] * d
            dist = np.linalg.norm(vals - proj, axis=1)
            better = dist < best_d
            best_d[better], best_t[better], best_seg[better] = dist[better], t[better], s
        tq = np.round(best_t * levels) / levels
        pick_b = tq > T[ys, xs]
        ia = np.array([chain[s] for s in best_seg])
        ib = np.array([chain[s + 1] for s in best_seg])
        res = np.where(pick_b, ib, ia)
        # mask edge -> nearest endpoint of the ramp segment (no spill)
        e = edge[ys, xs]
        res[e] = np.where(best_t[e] >= 0.5, ib[e], ia[e])
        out[ys, xs] = res
        for s in range(len(chain) - 1):
            pairs.add((int(chain[s]), int(chain[s + 1])))
    return out, {"pairs": sorted(pairs)}


def _endpoints_for(rm: np.ndarray, regions: list, entries: list, palette_lab: np.ndarray, lab_smooth: np.ndarray):
    """Endpoint palette indices for a mask component: gradient_endpoint entries of the region whose
    bbox overlaps most; else the two entries nearest the 5th / 95th percentile Lab along the ramp."""
    ys, xs = np.nonzero(rm)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    best, best_ov = None, 0
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        ov = max(0, min(x1, bbox[2]) - max(x0, bbox[0])) * max(0, min(y1, bbox[3]) - max(y0, bbox[1]))
        if ov > best_ov:
            best, best_ov = r, ov
    if best is not None:
        lo = [e["index"] for e in entries if e.get("gradient_endpoint") and list(e["gradient_endpoint"]) == [best["id"], "lab_lo"]]
        hi = [e["index"] for e in entries if e.get("gradient_endpoint") and list(e["gradient_endpoint"]) == [best["id"], "lab_hi"]]
        if lo and hi and lo[0] != hi[0]:
            return lo[0], hi[0]
        lab_lo, lab_hi = np.array(best["lab_lo"]), np.array(best["lab_hi"])
    else:
        vals = lab_smooth[ys, xs]
        L = vals[:, 0]
        lab_lo = vals[L <= np.percentile(L, 5)].mean(0)
        lab_hi = vals[L >= np.percentile(L, 95)].mean(0)
    i0 = int(np.argmin(delta_e2000(lab_lo[None], palette_lab)))
    i1 = int(np.argmin(delta_e2000(lab_hi[None], palette_lab)))
    if i0 == i1:
        return None
    return i0, i1


def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("dither")
    rg, st, pd, pr = job.stage_dir("regularize"), job.stage_dir("structure"), job.stage_dir("palette"), job.stage_dir("production")
    inputs = [rg / "indexed.npy", st / "gradient_mask.npy", st / "gradient_regions.json", pd / "palette.json"]
    if (pr / "design_lab_smooth.npy").exists():
        inputs.append(pr / "design_lab_smooth.npy")
    key = job.cache_key("dither", inputs)
    if job.cached(sd, key) and (sd / "indexed.npy").exists():
        job.timings["dither"] = 0.0
        return job.state

    P = job.params
    indexed = np.load(rg / "indexed.npy")
    gmask = np.load(st / "gradient_mask.npy")
    setting = job.override.get("dither") or P.get("dither", "auto")
    report = {"applied": False, "mask_pct": float(gmask.mean()), "levels": int(P.get("dither_levels", 4)), "pairs": []}
    if setting == "off" or not gmask.any() or not (pr / "design_lab_smooth.npy").exists():
        np.save(sd / "indexed.npy", indexed)
    else:
        pattern_name = P.get("dither_pattern")
        if job.override.get("dither_pattern_override"):
            pattern_name = job.override["dither_pattern_override"]
        if pattern_name is None:
            job.warn("dither_pattern_default")
        pattern = load_pattern(pattern_name)
        levels = int(pattern.get("levels", P.get("dither_levels", 4)))
        out, info = apply_dither(indexed, np.load(pr / "design_lab_smooth.npy"), gmask,
                                 read_json(st / "gradient_regions.json"), read_json(pd / "palette.json"), pattern, levels)
        np.save(sd / "indexed.npy", out.astype(np.int32))
        report.update({"applied": True, "levels": levels, "pairs": info["pairs"], "pattern": pattern.get("name", str(pattern_name))})
    dump_json(sd / "dither_stats.json", report)
    job.state["dither"] = report
    job.state.setdefault("stages_ran", []).append("dither")
    job.timings["dither"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
