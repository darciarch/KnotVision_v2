#!/usr/bin/env python
"""Photo survey (step 0): distribution of px/knot over the customer photos already at hand.

Input CSV (one row per photo):  path, width_cm, height_cm [, reed_per_m [, wefts_per_m]]
    reed/wefts default to --reed / --wefts (or the loom profile's reed_per_m; square assumption).
Output: data/photo_survey.csv (path, long_px, width_cm, height_cm, reed, px_per_knot_geo) + a decision:
    >= 70 % of photos with px_per_knot_geo >= photo_px_warn  -> customer flow is enough
    30-70 %                                                  -> customer flow + re-photographing
    <  30 %                                                  -> re-photograph for pair collection

Usage: python tools/photo_survey.py survey_input.csv [--reed 400] [--out data/photo_survey.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision.job import CONFIG_DIR, DATA_DIR, load_yaml  # noqa: E402


def survey_rows(rows: list[dict], reed_default: float, wefts_default: float | None = None) -> list[dict]:
    out = []
    for r in rows:
        p = Path(r["path"])
        if "long_px" in r and r["long_px"]:
            long_px = int(r["long_px"])
        else:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                long_px = int(max(im.size))
        reed = float(r.get("reed_per_m") or reed_default)
        wefts = float(r.get("wefts_per_m") or wefts_default or reed)
        W = float(r["width_cm"]) / 100 * reed
        H = float(r["height_cm"]) / 100 * wefts
        out.append({"path": str(p), "long_px": long_px, "width_cm": float(r["width_cm"]), "height_cm": float(r["height_cm"]),
                    "reed": reed, "px_per_knot_geo": long_px / max(W, H)})
    return out


def decide(px: np.ndarray, warn: float, stop: float) -> dict:
    px = np.asarray(px, dtype=float)
    frac_warn = float((px >= warn).mean()) if len(px) else 0.0
    frac_stop = float((px >= stop).mean()) if len(px) else 0.0
    if frac_warn >= 0.7:
        decision, source = "customer_flow", "customer photos are sufficient; collect pairs from the customer flow"
    elif frac_warn >= 0.3:
        decision, source = "mixed", "customer flow + factory/designer re-photographs existing rugs (>= 12 MP, top-down, full frame)"
    else:
        decision, source = "rephotograph", "pair collection entirely by re-photographing; track low_photo_resolution rate in the capacity report"
    return {"n": int(len(px)), "median": float(np.median(px)) if len(px) else None,
            "p10": float(np.percentile(px, 10)) if len(px) else None,
            "frac_ge_warn": frac_warn, "frac_ge_stop": frac_stop, "decision": decision, "note": source}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV: path,width_cm,height_cm[,reed_per_m[,wefts_per_m]]")
    ap.add_argument("--reed", type=float)
    ap.add_argument("--wefts", type=float)
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--out", default=str(DATA_DIR / "photo_survey.csv"))
    a = ap.parse_args(argv)
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    reed = a.reed or P.get("reed_per_m") or 400.0
    with open(a.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 30:
        print(f"warning: {len(rows)} photos; the spec asks for >= 30", file=sys.stderr)
    out = survey_rows(rows, reed, a.wefts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "long_px", "width_cm", "height_cm", "reed", "px_per_knot_geo"])
        w.writeheader()
        w.writerows(out)
    d = decide([r["px_per_knot_geo"] for r in out], float(P.get("photo_px_warn", 1.5)), float(P.get("photo_px_stop", 0.8)))
    for k, v in d.items():
        print(f"{k}: {v}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
