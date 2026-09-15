#!/usr/bin/env python
"""Real pair intake: validate a submission at collection time and copy it to data/real/pairs/<id>/.

Required: photo (jpg/png), truth.tiff (the FINAL production file), design_class, width_cm, height_cm,
density (per the loom's density_definition), machine_colors. px_per_knot_geo = long_px / max(W, H):
    < photo_px_stop -> rejected;  < photo_px_warn -> accepted with low_res_pair: true (train only).
Long-side pixel count alone never rejects.

Usage: python tools/pair_intake.py --id ID --photo P --truth T --class C --width CM --height CM --density D --colors N
       [--reed R] [--consent yes|no] [--out data/real/pairs]
API:   validate_pair(meta, photo_path, truth_path, profile) -> (ok, reasons, computed)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.intake import grid_from_density  # noqa: E402
from knotvision.job import CONFIG_DIR, DATA_DIR, JobError, load_yaml  # noqa: E402

REQUIRED = ("design_class", "width_cm", "height_cm", "density", "machine_colors")


def validate_pair(meta: dict, photo_path: Path | None, truth_path: Path | None, profile: dict) -> tuple[bool, list, dict]:
    reasons, computed = [], {}
    for k in REQUIRED:
        if meta.get(k) in (None, ""):
            reasons.append(f"missing:{k}")
    if meta.get("design_class") not in ("classic", "geometric", "gradient", "distressed", "minimal"):
        reasons.append("bad:design_class")
    if not photo_path or not Path(photo_path).exists():
        reasons.append("missing:photo")
    if not truth_path or not Path(truth_path).exists():
        reasons.append("missing:truth")
    if reasons:
        return False, reasons, computed
    try:
        g = grid_from_density(float(meta["width_cm"]), float(meta["height_cm"]), meta["density"],
                              meta.get("reed_per_m") or profile.get("reed_per_m"), profile.get("density_definition", "knots_m2"))
    except JobError as e:
        return False, [f"grid:{e.code}"], computed
    computed.update({"W": g["W"], "H": g["H"], "reed_per_m": g["reed_per_m"], "wefts_per_m": g["wefts_per_m"]})
    with Image.open(photo_path) as im:
        im = ImageOps.exif_transpose(im)
        long_px = max(im.size)
    computed["photo_native_px"] = int(long_px)
    ppk = long_px / max(g["W"], g["H"])
    computed["px_per_knot_geo"] = float(ppk)
    idx, rgb = read_indexed_tiff(truth_path)
    computed["truth_shape"] = list(idx.shape)
    computed["truth_colors"] = int(len(set(idx.ravel().tolist())))
    if abs(idx.shape[1] - g["W"]) > max(2, 0.02 * g["W"]) or abs(idx.shape[0] - g["H"]) > max(2, 0.02 * g["H"]):
        reasons.append(f"truth_grid_mismatch:{idx.shape[1]}x{idx.shape[0]}_vs_{g['W']}x{g['H']}")
    if computed["truth_colors"] > int(meta["machine_colors"]):
        reasons.append("truth_over_machine_colors")
    if ppk < float(profile.get("photo_px_stop", 0.8)):
        reasons.append(f"low_res_reject:{ppk:.2f}")
    computed["low_res_pair"] = bool(ppk < float(profile.get("photo_px_warn", 1.5)))
    return not reasons, reasons, computed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--class", dest="design_class", required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--colors", type=int, required=True)
    ap.add_argument("--reed", type=float)
    ap.add_argument("--consent", default="yes")
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--out", default=str(DATA_DIR / "real" / "pairs"))
    a = ap.parse_args(argv)
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    density = [float(x) for x in a.density.lower().split("x")] if "x" in a.density else float(a.density)
    meta = {"design_class": a.design_class, "width_cm": a.width, "height_cm": a.height, "density": density,
            "machine_colors": a.colors, "reed_per_m": a.reed, "consent_train": a.consent.lower() in ("yes", "true", "1")}
    ok, reasons, computed = validate_pair(meta, Path(a.photo), Path(a.truth), P)
    if not ok:
        print("REJECTED:", ", ".join(reasons))
        return 2
    out = Path(a.out) / a.id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.photo, out / ("photo" + Path(a.photo).suffix.lower()))
    shutil.copy(a.truth, out / "truth.tiff")
    meta.update(computed)
    meta.update({"reviewed": True, "design_mode": None, "edit_agent": None, "qc": None, "qc_status": None,
                 "designer_score": None, "fix_minutes": None, "fix_type": None})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ACCEPTED {a.id}: px_per_knot_geo={computed['px_per_knot_geo']:.2f}"
          + ("  [low_res_pair]" if computed["low_res_pair"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
