"""qc: hard gates (incl. edit_*), rerender, raw measurements, core feature merge, L2 logistic
score from models/qc_calib.json, decision, reasons, preview.png.

Output: qc.json, rerender.png, preview.png
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import PX_PER_KNOT
from .color import delta_e2000, rgb_to_lab
from .job import MODELS_DIR, Job, dump_json, read_json, sha256_file
from .metrics import banding, edge_density, edge_fidelity, line_jaggedness

HARD_GATE_WARNINGS = ("grid_unverified", "grid_mismatch", "partial_view", "non_planar", "non_planar_internal",
                      "internal_geometry_unmeasurable", "forced_low_res", "pile_blur", "dither_pattern_default",
                      "class_uncertain", "class_unsupported", "class_mixed", "guard_heavy")
EDIT_GATES = ("edit_drift", "edit_added", "edit_missing", "edit_low_res", "edit_timeout", "edit_fallback", "edit_no_gain")
CORE_FEATURES = ("rerender_edge_iou", "palette_deltaE_p95", "rerender_color_deltaE_p95", "line_jaggedness",
                 "noise_1knot_pct", "postprocess_changed_pct", "structure_mismatch", "banding",
                 "edge_width_knots", "illum_residual")


def rerender(indexed: np.ndarray, palette_rgb: np.ndarray, px: int = 2) -> np.ndarray:
    """indexed -> palette RGB at `px` px/knot with the synthetic chain's pile texture (no lighting,
    perspective or camera): a mild per-knot brightness ripple so that edges look like knots."""
    H, W = indexed.shape
    rgb = palette_rgb[indexed].astype(np.float32)
    up = cv2.resize(rgb, (W * px, H * px), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[0:H * px, 0:W * px]
    ripple = 1.0 + 0.03 * np.cos(2 * np.pi * (xx % px) / px) * np.cos(2 * np.pi * (yy % px) / px)
    return np.clip(up * ripple[..., None], 0, 255).astype(np.uint8)


def hard_gates(job: Job) -> list[str]:
    P = job.params
    fails = [w for w in job.warnings if w in HARD_GATE_WARNINGS]
    if job.state["colors"].get("palette_used", 0) > job.machine_colors:
        fails.append("palette_over_machine_colors")
    if job.state.get("non_planar_internal") and not job.state.get("internal_geometry", {}).get("tps_applied"):
        if "non_planar_internal" not in fails:
            fails.append("non_planar_internal")
    if job.state.get("design_mode") == "edit" or job.has_warning("edit_fallback"):
        e = job.state.get("edit") or {}
        prod = job.state.get("production") or {}
        ap = (e.get("align_px") or {}).get("p95")
        if ap is not None and ap > float(P.get("edit_align_max", 6.0)):
            fails.append("edit_drift")
        if prod.get("edit_added_pct") is not None and prod["edit_added_pct"] > float(P.get("edit_added_max", 0.01)):
            fails.append("edit_added")
        if prod.get("edit_missing_pct") is not None and prod["edit_missing_pct"] > float(P.get("edit_missing_max", 0.02)):
            fails.append("edit_missing")
        for w in ("edit_low_res", "edit_timeout", "edit_fallback", "edit_no_gain"):
            if job.has_warning(w):
                fails.append(w)
    return sorted(set(fails))


def core_features(raw: dict, job: Job) -> dict:
    """Merge raw measurements into the fixed core set; apply null rules."""
    P = job.params
    ed_min = float(P.get("edge_density_min", 0.02))
    low_edges = raw["edge_density"] < ed_min
    null_extra = set(P.get("qc_features_null", []))
    f = {
        "rerender_edge_iou": None if low_edges else raw["rerender_edge_iou"],
        "palette_deltaE_p95": raw["palette_deltaE_p95"],
        "rerender_color_deltaE_p95": raw["rerender_color_deltaE_p95"],
        "line_jaggedness": None if low_edges else raw["line_jaggedness"],
        "noise_1knot_pct": raw["noise_1knot_pct"],
        "postprocess_changed_pct": (raw.get("cleanup_changed_pct") or 0.0) + (raw.get("regularize_changed_pct") or 0.0) + (raw.get("guard_reverted_pct") or 0.0),
        "structure_mismatch": None if raw.get("symmetry_consistency") is None and raw.get("repeat_mismatch_raw") is None
        else max(x for x in (raw.get("symmetry_consistency"), raw.get("repeat_mismatch_raw")) if x is not None),
        "banding": raw.get("banding"),
        "edge_width_knots": raw.get("edge_width_knots"),
        "illum_residual": raw.get("illum_residual"),
    }
    for k in null_extra:
        if k in f:
            f[k] = None
    return f


def score(features: dict, calib: dict, design_class: str, design_mode: str) -> tuple[float, Optional[float], list]:
    m = calib["model"]
    names = calib["features"]
    x = []
    for i, n in enumerate(names):
        v = features.get(n)
        z = 0.0 if v is None else (float(v) - m["mean"][i]) / (m["std"][i] or 1.0)
        x.append(z)
    cls_levels = calib.get("class_levels", [])
    cls = design_class if design_class in cls_levels else "other"
    for lvl in cls_levels:
        x.append(1.0 if lvl == cls else 0.0)
    for lvl in calib.get("mode_levels", []):
        x.append(1.0 if lvl == design_mode else 0.0)
    x = np.array(x)
    coef = np.array(m["coef"])
    logit = float(coef @ x + m["intercept"])
    p = 1.0 / (1.0 + np.exp(-logit))
    contrib = [(names[i], float(coef[i] * x[i]), features.get(names[i])) for i in range(len(names))]
    contrib.sort(key=lambda c: c[1])
    reasons = [f"{n}={v:.3g}" if v is not None else f"{n}=null" for n, c, v in contrib[:3] if c < 0]
    p5 = None
    if calib.get("model_p5"):
        m5 = calib["model_p5"]
        x5 = x[:len(names)]
        logit5 = float(np.array(m5["coef"])[:len(x5)] @ x5 + m5["intercept"])
        p5 = 1.0 / (1.0 + np.exp(-logit5))
    return p, p5, reasons


def preview(job: Job, out: Path, rect: np.ndarray, rr: np.ndarray, indexed_raw: np.ndarray,
            indexed_before_dither: np.ndarray, indexed: np.ndarray, palette_rgb: np.ndarray):
    def to_panel(img, label):
        h, w = img.shape[:2]
        s = 400 / max(h, w)
        p = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((420, 400, 3), 255, np.uint8)
        canvas[:p.shape[0], :p.shape[1]] = p
        cv2.putText(canvas, label, (4, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas
    photo = cv2.cvtColor(cv2.imread(str(job.stage_dir("intake") / "photo.png")), cv2.COLOR_BGR2RGB)
    dd = job.stage_dir("design")
    model_out = cv2.cvtColor(cv2.imread(str(dd / "design_rgb.png")), cv2.COLOR_BGR2RGB)
    mode = job.state.get("design_mode")
    label = "model output"
    if mode == "edit" or job.has_warning("edit_fallback"):
        e = job.state.get("edit") or {}
        label = f"edit {e.get('selected', '')}"
    panels = [to_panel(photo, "photo"), to_panel(rect, "rectified"), to_panel(model_out, label),
              to_panel(palette_rgb[indexed_raw], "design (before regularize)"),
              to_panel(palette_rgb[indexed_before_dither], "after regularize"),
              to_panel(palette_rgb[indexed], "after dither"), to_panel(rr, "rerender")]
    cv2.imwrite(str(out), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))


def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("qc")
    dd, pr, rg, st, rd, pd = (job.stage_dir(s) for s in ("dither", "production", "regularize", "structure", "rectify", "palette"))
    P = job.params
    indexed = np.load(dd / "indexed.npy")
    indexed_raw = np.load(pr / "indexed_raw.npy")
    indexed_reg = np.load(rg / "indexed.npy")
    pal = read_json(pd / "palette.json")
    palette_rgb = np.array([e["rgb"] for e in pal["entries"]], dtype=np.uint8)
    palette_lab = np.array([e["lab"] for e in pal["entries"]], dtype=np.float64)
    rect = cv2.cvtColor(cv2.imread(str(rd / "rectified.png")), cv2.COLOR_BGR2RGB)
    gmask = np.load(st / "gradient_mask.npy")
    prod = read_json(pr / "production_stats.json")
    regs = read_json(rg / "regularize_stats.json")
    guard = job.state.get("guard") or {}
    H, W = indexed.shape

    # 1. hard gates
    fails = hard_gates(job)

    # 2. rerender at 2 px/knot
    rr = rerender(indexed, palette_rgb, px=2)
    rect2 = cv2.resize(rect, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(sd / "rerender.png"), cv2.cvtColor(rr, cv2.COLOR_RGB2BGR))

    # 3. raw measurements
    ed = edge_density(rect2)
    lab_rr = rgb_to_lab(rr)
    lab_rect = rgb_to_lab(rect2)
    # palette deltaE p95: rectified pixels vs nearest palette entry (dither mix colour inside the mask)
    lab_knot = cv2.resize(lab_rect.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    d = delta_e2000(lab_knot[:, None, :], palette_lab[None, :, :]).min(axis=1)
    if gmask.any():
        mix = cv2.resize(lab_rr.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        dm = delta_e2000(lab_knot, mix)
        gm = gmask.reshape(-1)
        d = np.where(gm, dm, d)
    # colour: 16x16 block means
    blk = 16
    def block_mean(lab):
        h, w = lab.shape[:2]
        return cv2.resize(lab.astype(np.float32), (max(1, w // blk), max(1, h // blk)), interpolation=cv2.INTER_AREA)
    dcol = delta_e2000(block_mean(lab_rr).reshape(-1, 3), block_mean(lab_rect).reshape(-1, 3))
    gmask_px = cv2.resize(gmask.astype(np.uint8), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST) > 0
    raw = {
        "edge_density": ed,
        "rerender_edge_iou": edge_fidelity(rr, rect2),
        "palette_deltaE_p50": float(np.percentile(d, 50)), "palette_deltaE_p95": float(np.percentile(d, 95)),
        "rerender_color_deltaE_mean": float(dcol.mean()), "rerender_color_deltaE_p95": float(np.percentile(dcol, 95)),
        "line_jaggedness": line_jaggedness(indexed, ~gmask),
        "noise_1knot_pct": prod.get("noise_1knot_pct"), "noise_knots_pct": prod.get("noise_knots_pct"),
        "cleanup_changed_pct": prod.get("cleanup_changed_pct"), "regularize_changed_pct": regs.get("regularize_changed_pct"),
        "guard_reverted_pct": guard.get("reverted_area_pct"),
        "symmetry_consistency": regs.get("symmetry_mismatch"), "repeat_mismatch_raw": regs.get("repeat_mismatch_raw"),
        "banding": banding(lab_rr[..., 0], lab_rect[..., 0], gmask_px, sigma_knots=8.0, px_per_knot=2.0) if gmask.any() else None,
        "edge_width_knots": job.state["grid"].get("edge_width_knots"),
        "illum_residual": (job.state.get("illumination") or {}).get("residual"),
        "illum_range": (job.state.get("illumination") or {}).get("range_L"),
        "wb_confidence": (job.state.get("wb") or {}).get("confidence"),
        "model_uncertainty": (job.state.get("design_stats") or {}).get("model_uncertainty"),
    }
    e = job.state.get("edit") or {}
    if e:
        raw.update({"edit_align_px": e.get("align_px"), "edit_added_pct": prod.get("edit_added_pct"),
                    "edit_missing_pct": prod.get("edit_missing_pct"), "edit_rounds": e.get("rounds"),
                    "edit_candidate_spread": e.get("candidate_spread"), "edit_select_margin": e.get("select_margin")})
    feats = core_features(raw, job)

    # 4. calibration
    calib_path = MODELS_DIR / "qc_calib.json"
    mode = job.state.get("design_mode", P.get("design_mode"))
    cfg_key = [job.design_class, mode, P.get("edit_agent") if mode == "edit" else None,
               (job.override.get("regularize") or P.get("regularize", "auto")) != "off"]
    qc = {"status": None, "score": None, "p5": None, "target": None, "features": feats, "raw": raw,
          "hard_gate_failures": fails, "reasons": [], "config": cfg_key}
    decision = "needs_review"
    if calib_path.exists():
        calib = read_json(calib_path)
        stale = False
        vers = job.state.get("versions", {})
        mh = calib.get("model_hashes", {})
        if mode == "segment" and mh.get("segment") and mh["segment"] != vers.get("segment_ckpt_sha256"):
            stale = True
        if mode == "edit":
            ah = calib.get("agent_hashes", {}).get(P.get("edit_agent"))
            if ah and ah != vers.get("edit_agent_hash"):
                stale = True
        if stale:
            job.warn("qc_calib_stale")
        calibrated = any(list(c) == cfg_key for c in calib.get("calibrated_configs", []))
        if mode == "edit" and calibrated:
            ah = calib.get("agent_hashes", {}).get(P.get("edit_agent"))
            if ah != vers.get("edit_agent_hash"):
                calibrated = False
        if not calibrated:
            job.warn("mode_uncalibrated")
            fails.append("mode_uncalibrated")
        s, p5, reasons = score(feats, calib, job.design_class, mode)
        qc.update({"score": s, "p5": p5, "target": calib.get("target"), "reasons": reasons})
        cls_level = job.design_class if job.design_class in calib.get("class_levels", []) else "other"
        unreliable = cls_level in calib.get("class_unreliable", []) or \
            f"{job.design_class}/{mode}/{P.get('edit_agent')}" in calib.get("config_unreliable", [])
        if not fails and not stale and not unreliable:
            if s >= float(calib.get("qc_accept", P.get("qc_accept", 0.8))):
                decision = "accepted"
            elif s < float(calib.get("qc_reject", P.get("qc_reject", 0.3))):
                decision = "rejected"
        qc["status"] = f"{decision}@{calib.get('target')}"
    else:
        job.warn("qc_uncalibrated")
        qc["status"] = "needs_review@none"
    if job.no_gate and decision == "needs_review" and not fails:
        pass
    qc["hard_gate_failures"] = sorted(set(fails))
    qc["decision"] = decision
    dump_json(sd / "qc.json", qc)
    job.state["qc"] = {k: v for k, v in qc.items() if k != "raw"}
    job.state["qc_raw"] = raw
    preview(job, sd / "preview.png", rect, rr, indexed_raw, indexed_reg, indexed, palette_rgb)
    job.state.setdefault("stages_ran", []).append("qc")
    job.timings["qc"] = round(time.time() - t0, 3)
    job.save()
    return job.state
