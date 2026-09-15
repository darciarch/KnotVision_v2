#!/usr/bin/env python
"""Real-pair alignment: truth.tiff -> palette RGB -> rectified space -> knotvision.align (same function
the edit loop uses) -> H_refined.json, H_local.json, align_mask.npy, gradient_mask.npy (truth dither
regions, tools/dither_stats rule), noise_mask.npy (truth noise regions, tools/noise_stats rule),
pair-level layout_changed decision, train/align_report.csv with a class breakdown.

Drop policy (spec): >= 40 kept pairs with >= 60 % mask fill -> normal; < 40 -> block threshold 6 px;
still < 40 -> segment trains on synthetic only, real pairs go to val/test.

Usage: python train/align.py [--pairs data/real/pairs] [--runs runs] [--report train/align_report.csv]
API:   align_pair(pair_dir, rectified_png, indexed_shape, profile) -> dict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from knotvision import PX_PER_KNOT  # noqa: E402
from knotvision.align import align_to_rectified  # noqa: E402
from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.job import DATA_DIR, RUNS_DIR, dump_json, read_json  # noqa: E402

MIN_REAL_SEGMENT = 40


def truth_render(indexed: np.ndarray, palette_rgb: np.ndarray, shape_px) -> np.ndarray:
    rgb = palette_rgb[indexed]
    return cv2.resize(rgb, (shape_px[1], shape_px[0]), interpolation=cv2.INTER_NEAREST)


def align_pair(pair: Path, rectified: np.ndarray, block_px: float = 4.0, fid_min: float = 0.3,
               noise_cc_max: int = 6, noise_thr: float = 0.05) -> dict:
    from dither_stats import dither_regions
    from noise_stats import noise_regions
    truth, pal = read_indexed_tiff(pair / "truth.tiff")
    H, W = truth.shape
    render = truth_render(truth, pal, rectified.shape[:2])
    r = align_to_rectified(render, rectified, mask_px=block_px, mask_fid=fid_min)
    grid = r["align_mask"].shape[0]
    # blocks with align_px > 2 px: truth boundaries' 1-knot neighbourhood removed from the mask
    mask_knots = np.zeros((H, W), bool)
    bh, bw = H // grid, W // grid
    disp = np.full((grid, grid), np.inf)
    if r["converged"]:
        # recompute the block displacement in the same grid for the > 2 px rule
        from knotvision.align import local_affine, _L8
        _, disp = local_affine(_L8(r["warped"]), _L8(rectified), grid=grid)
    for by in range(grid):
        for bx in range(grid):
            ys, xs = slice(by * bh, (by + 1) * bh if by < grid - 1 else H), slice(bx * bw, (bx + 1) * bw if bx < grid - 1 else W)
            if r["align_mask"][by, bx]:
                m = np.ones((ys.stop - ys.start, xs.stop - xs.start), bool)
                if disp[by, bx] > 2.0:
                    from knotvision.metrics import boundary_map
                    b = boundary_map(truth[ys, xs])
                    b = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    m &= ~b
                mask_knots[ys, xs] = m
    gmask = dither_regions(truth)
    nmask = noise_regions(truth, noise_cc_max, noise_thr)
    fill = float(r["align_mask"].mean())
    layout_changed = bool((not r["converged"]) or (r["align_px"]["median"] > 2 * block_px and fill < 0.5))
    np.save(pair / "align_mask.npy", mask_knots)
    np.save(pair / "gradient_mask.npy", gmask)
    np.save(pair / "noise_mask.npy", nmask)
    dump_json(pair / "H_refined.json", {"H": r["H_refined"]})
    dump_json(pair / "H_local.json", {"H_local": r["H_local"], "grid": grid})
    cv2.imwrite(str(pair / "flat_aligned.png"), cv2.cvtColor(r["warped"], cv2.COLOR_RGB2BGR))
    return {"converged": r["converged"], "align_px_median": r["align_px"]["median"], "align_px_p95": r["align_px"]["p95"],
            "mask_fill": fill, "layout_changed": layout_changed, "gradient_frac": float(gmask.mean()), "noise_frac": float(nmask.mean())}


def rectified_for_pair(pair: Path, runs_dir: Path) -> np.ndarray | None:
    """The pair's rectified.png from its pipeline run (train/prepare.py runs intake..structure first)."""
    for cand in (runs_dir / f"prep_{pair.name}", runs_dir / pair.name):
        p = cand / "stages" / "01_rectify" / "rectified.png"
        if p.exists():
            return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DATA_DIR / "real" / "pairs"))
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--report", default=str(ROOT / "train" / "align_report.csv"))
    ap.add_argument("--block-px", type=float, default=4.0)
    a = ap.parse_args(argv)
    pairs = sorted(p for p in Path(a.pairs).iterdir() if (p / "truth.tiff").exists())
    rows = []
    for p in pairs:
        rect = rectified_for_pair(p, Path(a.runs))
        if rect is None:
            rows.append({"pair": p.name, "design_class": read_json(p / "meta.json").get("design_class"), "decision": "no_rectified"})
            continue
        res = align_pair(p, rect, block_px=a.block_px)
        meta = read_json(p / "meta.json")
        rows.append({"pair": p.name, "design_class": meta.get("design_class"), **res,
                     "decision": "drop:layout_changed" if res["layout_changed"] else "keep"})
    kept = [r for r in rows if r.get("decision") == "keep"]
    fill = np.mean([r["mask_fill"] for r in kept]) if kept else 0.0
    policy = "normal" if len(kept) >= MIN_REAL_SEGMENT and fill >= 0.6 else \
        ("retry_block_6px" if a.block_px < 6 else "segment_synthetic_only_real_val_test")
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "design_class", "converged", "align_px_median", "align_px_p95", "mask_fill", "layout_changed",
            "gradient_frac", "noise_frac", "decision"]
    with open(a.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    by_class = {}
    for r in rows:
        by_class.setdefault(r.get("design_class"), []).append(r)
    print(f"pairs: {len(rows)} kept: {len(kept)} mean fill: {fill:.2f} policy: {policy}")
    for c, rs in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        k = sum(1 for r in rs if r.get("decision") == "keep")
        print(f"  {c}: {k}/{len(rs)} kept")
    print(f"report -> {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
