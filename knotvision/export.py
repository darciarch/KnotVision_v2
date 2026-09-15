"""export: design.tiff (indexed, palette colormap, anisotropic resolution), palette.csv, preview.png,
report.json; consent_train -> data/real/pairs/<job_id>/.

Output: runs/<job_id>/output/
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

from .job import DATA_DIR, Job, dump_json, read_json


def write_indexed_tiff(path: Path, indexed: np.ndarray, palette_rgb: np.ndarray, reed_per_m: float, wefts_per_m: float):
    """tifffile, photometric='palette', colormap = palette RGB (uint16), resolution in knots/inch."""
    k = len(palette_rgb)
    cmap = np.zeros((3, 256), dtype=np.uint16)
    cmap[:, :k] = (palette_rgb.astype(np.uint16) * 257).T
    data = indexed.astype(np.uint8)
    res_x = reed_per_m * 0.0254
    res_y = wefts_per_m * 0.0254
    tifffile.imwrite(str(path), data, photometric="palette", colormap=cmap,
                     resolution=(res_x, res_y), resolutionunit="INCH",
                     metadata={"knotvision": "6.6", "palette_used": int(k)})


def read_indexed_tiff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        data = page.asarray()
        cmap = page.colormap  # (3, 2**bits)
    k = int(data.max()) + 1 if data.size else 0
    rgb = (cmap[:, :k].T / 257).round().astype(np.uint8) if cmap is not None else None
    return data.astype(np.int32), rgb


def write_palette_csv(path: Path, entries: list, stats_entries: list):
    st = {e["index"]: e for e in stats_entries}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "yarn_code", "name", "R", "G", "B", "knots", "coverage_pct", "quantization_deltaE"])
        for e in entries:
            s = st.get(e["index"], {})
            w.writerow([e["index"], e.get("yarn_code", ""), e.get("name", ""), *e["rgb"],
                        s.get("knots", 0), f"{s.get('coverage_pct', 0.0):.3f}",
                        "" if s.get("quantization_deltaE") is None else f"{s['quantization_deltaE']:.2f}"])


def build_report(job: Job) -> dict:
    s = job.state
    P = job.params
    layout = s.get("layout", {})
    rep = {
        "job_id": job.job_id,
        "versions": s.get("versions"),
        "setup": s.get("setup"),
        "photo": s.get("photo"),
        "grid": s.get("grid"),
        "design_class": job.design_class, "class_source": job.class_source,
        "class_auto": s.get("class_auto"), "class_margin": s.get("class_margin"),
        "class_conflict": s.get("class_conflict", False), "class_mixed": s.get("class_mixed", False),
        "class_features": s.get("class_features"),
        "colors": s.get("colors"),
        "fringe": s.get("fringe"),
        "illumination": s.get("illumination"),
        "internal_geometry": s.get("internal_geometry"),
        "palette_confirmed": s.get("palette_confirmed", False),
        "design_mode": s.get("design_mode", P.get("design_mode")),
        "design_mode_requested": s.get("design_mode_requested", P.get("design_mode_requested")),
        "design_mode_class_default": s.get("design_mode_class_default", P.get("design_mode_class_default")),
        "edit": s.get("edit") if s.get("design_mode") == "edit" else None,
        "regularize": s.get("regularize"),
        "guard": s.get("guard"),
        "dither": s.get("dither"),
        "qc": s.get("qc"),
        "timings_s": {**job.timings, "total": round(sum(v for v in job.timings.values() if isinstance(v, (int, float))), 3)},
        "warnings": s.get("warnings", []),
        "partial_view": s.get("partial_view", False), "non_planar": s.get("non_planar", False),
        "non_planar_internal": s.get("non_planar_internal", False),
        "deterministic": s.get("deterministic", True),
        "layout": {k: layout.get(k) for k in ("symmetry", "border_accepted", "frame_hint", "border_period", "medallion", "ground_repeat", "gradient_frac")},
    }
    return rep


def run(job: Job) -> dict:
    t0 = time.time()
    out = job.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dd, pd, pr, qd = job.stage_dir("dither"), job.stage_dir("palette"), job.stage_dir("production"), job.stage_dir("qc")
    indexed = np.load(dd / "indexed.npy")
    pal = read_json(pd / "palette.json")
    entries = pal["entries"]
    palette_rgb = np.array([e["rgb"] for e in entries], dtype=np.uint8)
    g = job.state["grid"]
    write_indexed_tiff(out / "design.tiff", indexed, palette_rgb, g["reed_per_m"], g["wefts_per_m"])
    stats = read_json(pr / "production_stats.json")
    # recount on the final map
    final_entries = []
    for e in entries:
        m = indexed == e["index"]
        q = next((x["quantization_deltaE"] for x in stats["entries"] if x["index"] == e["index"]), None)
        final_entries.append({"index": e["index"], "knots": int(m.sum()), "coverage_pct": float(m.mean() * 100), "quantization_deltaE": q})
    write_palette_csv(out / "palette.csv", entries, final_entries)
    if (qd / "preview.png").exists():
        shutil.copy(qd / "preview.png", out / "preview.png")
    if (qd / "qc.json").exists():
        shutil.copy(qd / "qc.json", out / "qc.json")
    report = build_report(job)
    dump_json(out / "report.json", report)
    # 4. training pair
    if job.consent_train:
        pair = DATA_DIR / "real" / "pairs" / job.job_id
        pair.mkdir(parents=True, exist_ok=True)
        shutil.copy(job.photo_path, pair / ("photo" + Path(job.photo_path).suffix.lower()))
        shutil.copy(out / "design.tiff", pair / "design.tiff")
        shutil.copy(pr / "indexed_raw.npy", pair / "indexed_raw.npy")
        rc = job.stage_dir("rectify") / "corners.json"
        if rc.exists():
            shutil.copy(rc, pair / "corners.json")
        meta = {"consent_train": True, "reviewed": False, "design_class": job.design_class,
                "width_cm": job.width_cm, "height_cm": job.height_cm, "reed_per_m": g["reed_per_m"],
                "wefts_per_m": g["wefts_per_m"], "machine_colors": job.machine_colors,
                "photo_native_px": job.state.get("photo", {}).get("native_long_px"),
                "px_per_knot_geo": g.get("photo_px_per_knot"), "design_mode": report["design_mode"],
                "edit_agent": job.params.get("edit_agent") if report["design_mode"] == "edit" else None,
                "qc": (job.state.get("qc") or {}).get("features"), "qc_status": (job.state.get("qc") or {}).get("status"),
                "designer_score": None, "fix_minutes": None, "fix_type": None}
        dump_json(pair / "meta.json", meta)
    job.state.setdefault("stages_ran", []).append("export")
    job.timings["export"] = round(time.time() - t0, 3)
    job.save()
    return report
