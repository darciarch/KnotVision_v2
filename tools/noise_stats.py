#!/usr/bin/env python
"""Noise scale from the archive's distressed designs: distribution of small-component areas inside
noisy regions -> noise_cc_min / noise_cc_max (p10 / p90 of areas of components <= 16 knots inside
noise regions), single-knot share -> distressed min_region_knots (0 when intentional 1-knot noise
exists, else 1). Writes the profile fields and, optionally, config/classes/distressed.yaml.

Also exposes noise_regions(indexed, cc_max, thr) — the rule train/align.py uses for truth noise_mask:
16x16-knot windows where the density of components with area <= cc_max is >= thr.

Usage: python tools/noise_stats.py [--archive DIR] [--profile wilton_default] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive import ROOT, iter_designs, merge_same_rgb, write_profile_fields  # noqa: E402

from knotvision.job import CONFIG_DIR, load_yaml  # noqa: E402
from knotvision.metrics import component_area_map  # noqa: E402

WIN = 16


def noise_regions(indexed: np.ndarray, cc_max: int = 6, thr: float = 0.05, win: int = WIN) -> np.ndarray:
    """Knot mask: windows whose share of knots belonging to components with area <= cc_max is >= thr."""
    H, W = indexed.shape
    area = component_area_map(indexed)
    small = (area <= cc_max).astype(np.float32)
    dens = ndi.uniform_filter(small, size=win, mode="constant")
    return dens >= thr


def compute(archive_dir: Path | None, cc_max_probe: int = 16, thr: float = 0.05) -> dict:
    areas, ones, total_noise = [], 0, 0
    n = 0
    for _id, idx, rgb in iter_designs(archive_dir):
        idx, _ = merge_same_rgb(idx, rgb)
        region = noise_regions(idx, cc_max_probe, thr)
        if not region.any():
            continue
        n += 1
        area = component_area_map(idx)
        small = (area <= cc_max_probe) & region
        # count components (not knots): sample each component once via its label
        lab, k = ndi.label(small)
        if k == 0:
            continue
        sizes = ndi.sum(small, lab, index=np.arange(1, k + 1)).astype(int)
        areas.extend(sizes.tolist())
        ones += int((sizes == 1).sum())
        total_noise += int(len(sizes))
    if not areas:
        return {"n_designs": n, "noise_cc_min": None, "noise_cc_max": None, "one_knot_share": None}
    a = np.array(areas)
    multi = a[a >= 2]
    cc_min = int(max(2, np.percentile(multi, 10))) if len(multi) else 2
    cc_max = int(max(cc_min + 1, np.percentile(multi, 90))) if len(multi) else 6
    share1 = ones / max(1, total_noise)
    return {"n_designs": n, "n_components": int(len(a)), "noise_cc_min": cc_min, "noise_cc_max": cc_max,
            "one_knot_share": float(share1), "distressed_min_region_knots": 0 if share1 >= 0.2 else 1}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive")
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    r = compute(Path(a.archive) if a.archive else None)
    print(r)
    if r["noise_cc_min"] is None:
        print("no noisy archive designs found; profile keeps the assumed noise scale")
        return 1
    if not a.dry_run:
        write_profile_fields({"noise_cc_min": r["noise_cc_min"], "noise_cc_max": r["noise_cc_max"], "noise_scale_source": "archive"}, a.profile)
        cp = CONFIG_DIR / "classes" / "distressed.yaml"
        cfg = load_yaml(cp)
        cfg["min_region_knots"] = r["distressed_min_region_knots"]
        cp.write_text("# config/classes/distressed.yaml (min_region_knots set by tools/noise_stats.py)\n" + yaml.safe_dump(cfg, sort_keys=False))
        print("profile and distressed.yaml updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
