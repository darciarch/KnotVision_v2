"""intake: validation, EXIF rotation, grid (W x H) from density definition, yarn library, parameter
layering, versions/setup blocks.

Output: job.json, photo.png, yarn_lab.npy (optional)
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from . import DESIGN_CLASSES, PX_PER_KNOT
from .color import rgb_to_lab
from .job import CONFIG_DIR, Job, JobError, dump_json, setup_block, versions_block


# ----------------------------------------------------------------------------- grid

def grid_from_density(width_cm: float, height_cm: float, density, reed_per_m: Optional[float],
                      density_definition: str) -> dict:
    """Returns dict(W, H, reed_per_m, wefts_per_m, grid_verified, grid_rounding, measured?).

    knots_m2:     wefts_per_m = density / reed_per_m
    wefts_per_m:  wefts_per_m = density
    reed_x_wefts: density = [reed, wefts]; reed must equal the loom's reed_per_m
    reed_per_m None: only with knots_m2, k = sqrt(density) for both axes -> grid_unverified.
    grid_rounding: the fractional knot count is within 0.1 of a midpoint (rounding error >= 0.4 knot),
    i.e. the integer grid is ambiguous (the spec's "> 0.5" cannot occur for a nearest-integer rounding).
    """
    warnings = []
    if density_definition not in ("knots_m2", "wefts_per_m", "reed_x_wefts"):
        raise JobError("bad_density_definition", density_definition)
    grid_verified = True
    if reed_per_m is None:
        if density_definition != "knots_m2":
            raise JobError("grid_unverified", "reed_per_m is null; only knots_m2 density is possible")
        k = math.sqrt(float(density))
        reed, wefts = k, k
        grid_verified = False
        warnings.append("grid_unverified")
    elif density_definition == "knots_m2":
        reed = float(reed_per_m)
        wefts = float(density) / reed
    elif density_definition == "wefts_per_m":
        reed = float(reed_per_m)
        wefts = float(density)
    else:  # reed_x_wefts
        if not isinstance(density, (list, tuple)) or len(density) != 2:
            raise JobError("bad_density", "reed_x_wefts expects density=[reed, wefts]")
        if abs(float(density[0]) - float(reed_per_m)) > 1e-6:
            raise JobError("grid_input_mismatch",
                           f"density[0]={density[0]} != loom reed_per_m={reed_per_m}")
        reed, wefts = float(density[0]), float(density[1])
    if reed <= 0 or wefts <= 0:
        raise JobError("bad_density", "reed/wefts must be > 0")
    Wf = width_cm / 100.0 * reed
    Hf = height_cm / 100.0 * wefts
    W, H = int(round(Wf)), int(round(Hf))
    if abs(Wf - W) >= 0.4 or abs(Hf - H) >= 0.4:
        warnings.append("grid_rounding")
    if W < 2 or H < 2:
        raise JobError("bad_grid", f"grid too small: {W}x{H}")
    return {"W": W, "H": H, "reed_per_m": reed, "wefts_per_m": wefts, "grid_verified": grid_verified,
            "warnings": warnings}


# ----------------------------------------------------------------------------- yarn library

def load_yarn_csv(path: Path) -> tuple[list[dict], np.ndarray]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"yarn_code": r["yarn_code"], "name": r.get("name", ""),
                         "rgb": [int(r["R"]), int(r["G"]), int(r["B"])],
                         "material": r.get("material", "")})
    if not rows:
        raise JobError("yarn_csv_empty", str(path))
    lab = rgb_to_lab(np.array([r["rgb"] for r in rows], dtype=np.float64) / 255.0)
    return rows, lab


# ----------------------------------------------------------------------------- validation

def validate(job: Job):
    if not Path(job.photo_path).exists():
        raise JobError("photo_missing", str(job.photo_path))
    if not (job.width_cm > 0 and job.height_cm > 0):
        raise JobError("bad_size", "width_cm/height_cm must be > 0")
    if isinstance(job.density, (list, tuple)):
        if any(float(x) <= 0 for x in job.density):
            raise JobError("bad_density", "density entries must be > 0")
    elif not (float(job.density) > 0):
        raise JobError("bad_density", "density must be > 0")
    allowed = job.profile.get("allowed_machine_colors", [])
    if job.machine_colors not in allowed:
        raise JobError("machine_colors_not_allowed", f"{job.machine_colors} not in {allowed}")
    if job.consent_train is None:
        raise JobError("consent_missing", "consent_train must be given")
    if job.design_class == "auto":
        if not job.batch:
            raise JobError("class_required", "--class auto is only accepted with batch/synthetic")
        job.class_source = "auto"
    elif job.design_class == "pictorial":
        raise JobError("class_unsupported", "pictorial is out of scope")
    elif job.design_class not in DESIGN_CLASSES:
        raise JobError("class_unknown", job.design_class)
    else:
        if not (CONFIG_DIR / "classes" / f"{job.design_class}.yaml").exists():
            raise JobError("class_config_missing", job.design_class)
        job.class_source = "designer"
    # override fields
    ov = job.override
    for k in ("force_symmetry", "force_border_repeat", "force_ground_repeat", "regularize", "dither"):
        if ov.get(k) is not None and ov[k] not in ("auto", "on", "off"):
            raise JobError("bad_override", f"{k} must be auto|on|off")
    if job.corners_override is not None:
        c = np.asarray(job.corners_override, dtype=np.float64)
        if c.shape != (4, 2):
            raise JobError("bad_corners", "corners_override must be 4 x [x, y]")
    if job.palette_override is not None and len(job.palette_override) > job.machine_colors:
        raise JobError("bad_palette_override", "more entries than machine_colors")


def validate_edit(job: Job):
    """edit-specific checks after parameter resolution."""
    if job.params.get("design_mode") != "edit":
        return
    agent = job.params.get("agent")
    if not agent:
        raise JobError("edit_agent_missing")
    backend = agent.get("backend")
    if backend not in ("qwen_edit_2511", "qwen_edit_2509", "muse_image", "mock"):
        raise JobError("edit_backend_unknown", str(backend))
    if backend == "muse_image" and not job.profile.get("allow_api", False):
        raise JobError("api_not_allowed", "muse_image needs allow_api: true in the loom profile")


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    job.load_profile()
    validate(job)
    job.resolve()
    validate_edit(job)

    sd = job.stage_dir("intake")
    key = job.cache_key("intake", [job.photo_path] + ([job.yarn_csv] if job.yarn_csv else []),
                        extra={"w": job.width_cm, "h": job.height_cm, "d": job.density,
                               "mc": job.machine_colors, "cls": job.design_class})
    if job.cached(sd, key) and (sd / "photo.png").exists():
        job.timings["intake"] = 0.0
        return job.state

    # 2. EXIF rotation + native size
    img = Image.open(job.photo_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    native_long = max(img.size)
    photo = {"native_long_px": int(native_long),
             "long_px_below_rec": native_long < int(job.params.get("photo_long_px_rec", 3000))}
    if photo["long_px_below_rec"]:
        job.warn("photo_long_px_below_rec")

    # 3. grid
    g = grid_from_density(job.width_cm, job.height_cm, job.density, job.profile.get("reed_per_m"),
                          job.profile.get("density_definition", "knots_m2"))
    for w in g.pop("warnings"):
        job.warn(w)
    W, H = g["W"], g["H"]
    g.update({"photo_px_per_knot": None, "edge_width_knots": None, "photo_px_per_knot_eff": None,
              "knot_grid": {"visible": False, "spacing_px": None, "mismatch_pct": None},
              "upsampled": None, "measured_reed_per_m": None})
    job.state["grid"] = g

    # long side cap: max(4096, 4 * max(W, H))
    cap = max(4096, PX_PER_KNOT * max(W, H))
    if native_long > cap:
        s = cap / native_long
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    img.save(sd / "photo.png")

    # 4. palette upper bound
    job.state["colors"] = {"machine_colors": int(job.machine_colors), "palette_used": None}

    # 5. yarn library
    if job.yarn_csv:
        rows, lab = load_yarn_csv(job.yarn_csv)
        np.save(sd / "yarn_lab.npy", lab)
        dump_json(sd / "yarns.json", rows)

    # 7. versions, setup, photo
    job.state["versions"] = versions_block(job.profile_path, job.design_class, job.yarn_csv,
                                          job.params.get("edit_agent") if job.is_edit else None)
    job.state["setup"] = setup_block(job.profile, job.params, job.yarn_csv)
    job.state["photo"] = photo
    job.state["design_class"] = job.design_class
    job.state["class_source"] = job.class_source
    job.state.setdefault("stages_ran", []).append("intake")
    job.timings["intake"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
