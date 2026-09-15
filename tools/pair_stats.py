#!/usr/bin/env python
"""Real-world distributions from real pairs -> data/synth/dist.yaml (source: pairs). Without pairs,
the px/knot distribution comes from data/photo_survey.csv (source: survey) and the rest stays default.

Usage: python tools/pair_stats.py [--pairs data/real/pairs] [--survey data/photo_survey.csv] [--out data/synth/dist.yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knotvision.job import DATA_DIR  # noqa: E402

DEFAULT = {"source": "default", "n_colors": {"min": 2, "max": 12}, "width_cm": {"min": 60, "max": 300},
           "height_cm": {"min": 90, "max": 400}, "reed_per_m": {"min": 300, "max": 500},
           "wefts_per_m": {"min": 300, "max": 500}, "photo_px_per_knot": {"p10": 1.2, "p50": 1.9, "p90": 3.5},
           "machine_colors": [5, 6, 8, 12]}


def _rng(vals, lo_p=5, hi_p=95):
    v = np.asarray(vals, float)
    return {"min": float(np.percentile(v, lo_p)), "max": float(np.percentile(v, hi_p))}


def from_pairs(pairs_dir: Path) -> dict | None:
    metas = []
    for p in sorted(pairs_dir.iterdir()) if pairs_dir.exists() else []:
        m = p / "meta.json"
        if m.exists():
            metas.append(json.loads(m.read_text()))
    if len(metas) < 5:
        return None
    d = dict(DEFAULT)
    d["source"] = "pairs"
    cols = [m["truth_colors"] for m in metas if m.get("truth_colors")]
    if cols:
        d["n_colors"] = {"min": int(min(cols)), "max": int(max(cols))}
    d["width_cm"] = _rng([m["width_cm"] for m in metas])
    d["height_cm"] = _rng([m["height_cm"] for m in metas])
    reeds = [m["reed_per_m"] for m in metas if m.get("reed_per_m")]
    wefts = [m["wefts_per_m"] for m in metas if m.get("wefts_per_m")]
    if reeds:
        d["reed_per_m"] = _rng(reeds, 0, 100)
    if wefts:
        d["wefts_per_m"] = _rng(wefts, 0, 100)
    ppk = [m["px_per_knot_geo"] for m in metas if m.get("px_per_knot_geo")]
    if ppk:
        d["photo_px_per_knot"] = {"p10": float(np.percentile(ppk, 10)), "p50": float(np.percentile(ppk, 50)), "p90": float(np.percentile(ppk, 90))}
    mc = sorted({int(m["machine_colors"]) for m in metas if m.get("machine_colors")})
    if mc:
        d["machine_colors"] = mc
    d["n_pairs"] = len(metas)
    return d


def from_survey(survey_csv: Path) -> dict | None:
    if not survey_csv.exists():
        return None
    with open(survey_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    ppk = [float(r["px_per_knot_geo"]) for r in rows if r.get("px_per_knot_geo")]
    if not ppk:
        return None
    d = dict(DEFAULT)
    d["source"] = "survey"
    d["photo_px_per_knot"] = {"p10": float(np.percentile(ppk, 10)), "p50": float(np.percentile(ppk, 50)), "p90": float(np.percentile(ppk, 90))}
    d["width_cm"] = _rng([float(r["width_cm"]) for r in rows])
    d["height_cm"] = _rng([float(r["height_cm"]) for r in rows])
    d["n_photos"] = len(rows)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DATA_DIR / "real" / "pairs"))
    ap.add_argument("--survey", default=str(DATA_DIR / "photo_survey.csv"))
    ap.add_argument("--out", default=str(DATA_DIR / "synth" / "dist.yaml"))
    a = ap.parse_args(argv)
    d = from_pairs(Path(a.pairs)) or from_survey(Path(a.survey)) or dict(DEFAULT)
    Path(a.out).write_text("# Real-world distributions used by the synthetic generators (tools/pair_stats.py).\n"
                           "# source: default | survey | pairs\n" + yaml.safe_dump(d, sort_keys=False))
    print(f"source={d['source']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
