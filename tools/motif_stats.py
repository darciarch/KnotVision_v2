#!/usr/bin/env python
"""motif_min_knots from the archive: 5th percentile of the area of non-thin components in the archive
truth designs. Writes motif_min_knots + motif_min_knots_source: archive to the loom profile.

Usage: python tools/motif_stats.py [--archive data/archive/designs] [--profile wilton_default] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive import ROOT, iter_designs, merge_same_rgb, write_profile_fields  # noqa: E402

from knotvision.job import CONFIG_DIR, load_yaml  # noqa: E402
from knotvision.metrics import connected_components, is_thin  # noqa: E402


def motif_areas(indexed: np.ndarray, line_min_knots: int = 6, line_max_width: int = 1, min_area: int = 2) -> list[int]:
    """Areas of components that are not thin lines and not single/paired knots (noise)."""
    areas = []
    for c in connected_components(indexed):
        if c.area < min_area:
            continue
        if is_thin(c.full_mask(indexed.shape), line_min_knots, line_max_width):
            continue
        areas.append(c.area)
    return areas


def compute(archive_dir: Path | None, line_min_knots: int, line_max_width: int, pct: float = 5.0) -> dict:
    all_areas = []
    n = 0
    for _id, idx, rgb in iter_designs(archive_dir):
        idx, _ = merge_same_rgb(idx, rgb)
        all_areas.extend(motif_areas(idx, line_min_knots, line_max_width))
        n += 1
    if not all_areas:
        return {"n_designs": n, "motif_min_knots": None}
    a = np.array(all_areas)
    return {"n_designs": n, "n_components": int(len(a)), "motif_min_knots": int(max(2, np.percentile(a, pct))),
            "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive")
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    r = compute(Path(a.archive) if a.archive else None, int(P.get("line_min_knots", 6)), int(P.get("line_max_width", 1)))
    print(r)
    if r["motif_min_knots"] is None:
        print("no archive designs found; profile keeps motif_min_knots (assumed)")
        return 1
    if not a.dry_run:
        write_profile_fields({"motif_min_knots": r["motif_min_knots"], "motif_min_knots_source": "archive"}, a.profile)
        print(f"profile updated: motif_min_knots = {r['motif_min_knots']} (archive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
