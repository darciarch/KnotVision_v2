#!/usr/bin/env python
"""Offline evaluation against truth.tiff: runs the pipeline on pair dirs, measures every spec metric,
writes a per-job CSV + class-broken summary, checks the release criterion per mode, derives the edit
gates (edit_gate_corr) and flags > 10 % regressions against a previous result file.

Pair dir: photo.jpg, truth.tiff, meta.json (design_class, width_cm, height_cm, reed/wefts, machine_colors,
[photo.*, gradient_frac, noise_frac, thin_line_frac, pure, mixed]) and optionally corners_gt.json /
corners.json, layout_gt.json, gradient_mask.npy, noise_mask.npy, illum_gt.npy.

Usage: python eval/run.py --pairs data/synth/pairs [--ids eval/synth_test_ids.txt] [--mode quantize|segment|edit]
       [--agent NAME] [--regularize on|off] [--class auto] [--out eval/results/run_<date>.csv] [--prev PREV.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision import DESIGN_CLASSES  # noqa: E402
from knotvision.color import delta_e2000, rgb_to_lab  # noqa: E402
from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.job import CONFIG_DIR, RUNS_DIR, Job, JobError, load_yaml, read_json  # noqa: E402
from knotvision.metrics import (banding, cc_added_missing, derive_edit_gates, dither_err, edge_iou_indexed,  # noqa: E402
                                line_jaggedness, noise_fidelity, noise_stats, repeat_mismatch, symmetry_mismatch,
                                thin_line_recall)
from knotvision.run import run_job  # noqa: E402

RELEASE = {"cc_added_motif_max": 0, "cc_added_area_pct": 0.002, "cc_missing_pct": 0.01, "thin_line_recall": 0.9,
           "jag_max": 0.15, "repeat_mismatch": 0.01, "banding_max": 2.0, "noise_fidelity_min": 0.8}
METRIC_COLS = ["corner_err_px", "fringe_err_px", "planarity_detect", "tps_gain", "grid_field_err", "illum_err",
               "edge_width_err", "layout_err", "class_ok", "class_uncertain", "class_mixed_flag", "class_conflict",
               "thin_line_frac_err", "knot_deltaE_mean", "knot_deltaE_p95", "banding", "dither_err", "noise_fidelity",
               "noise_1knot_delta", "thin_line_recall", "cc_added_area_pct", "cc_added_motif", "cc_missing_pct",
               "edge_iou", "line_jaggedness_delta", "free_len_share", "repeat_mismatch", "symmetry_diff",
               "regularize_changed_pct", "guard_reverted_pct", "edit_align_px_median", "edit_align_px_p95",
               "edit_added_pct", "edit_missing_pct", "edit_rounds", "edit_candidate_spread", "edit_routed_out",
               "edit_no_gain", "cost_usd", "qc_score", "qc_p5", "time_s"]


# ----------------------------------------------------------------------------- truth helpers

def remap_to_truth(indexed: np.ndarray, palette_lab: np.ndarray, truth_lab: np.ndarray) -> np.ndarray:
    """Output indices -> nearest truth index by deltaE2000 (ties: lowest), so that cc_* can match by index."""
    d = delta_e2000(palette_lab[:, None, :], truth_lab[None, :, :])
    m = np.argmin(d, axis=1)
    return m[indexed]


def _resize_nearest(a: np.ndarray, shape) -> np.ndarray:
    import cv2
    if a.shape == tuple(shape):
        return a
    return cv2.resize(a.astype(np.int32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _mask_iou(a, b) -> float | None:
    if a is None or b is None:
        return None
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


# ----------------------------------------------------------------------------- one job

def evaluate_job(job: Job, rep: dict, pair: Path, meta: dict) -> dict:
    P = job.params
    out = {"job_id": job.job_id, "pair": pair.name, "design_class": meta.get("design_class"),
           "class_detected": rep.get("class_auto"), "mode": rep.get("design_mode"),
           "mode_requested": rep.get("design_mode_requested"), "edit_agent": P.get("edit_agent") if rep.get("design_mode") == "edit" else None,
           "regularize": (job.override.get("regularize") or P.get("regularize", "auto")) != "off",
           "qc_status": (rep.get("qc") or {}).get("status"), "warnings": "|".join(rep.get("warnings", []))}
    out.update({k: None for k in METRIC_COLS})
    truth, truth_rgb = read_indexed_tiff(pair / "truth.tiff")
    truth_lab = rgb_to_lab(truth_rgb)
    indexed = np.load(job.stage_dir("dither") / "indexed.npy")
    indexed_raw = np.load(job.stage_dir("production") / "indexed_raw.npy")
    pal = read_json(job.stage_dir("palette") / "palette.json")
    pal_lab = np.array([e["lab"] for e in pal["entries"]])
    if truth.shape != indexed.shape:
        truth = _resize_nearest(truth, indexed.shape)
    gm_t = np.load(pair / "gradient_mask.npy") if (pair / "gradient_mask.npy").exists() else np.zeros(truth.shape, bool)
    nm_t = np.load(pair / "noise_mask.npy") if (pair / "noise_mask.npy").exists() else np.zeros(truth.shape, bool)
    gm_t = _resize_nearest(gm_t.astype(np.int32), indexed.shape).astype(bool)
    nm_t = _resize_nearest(nm_t.astype(np.int32), indexed.shape).astype(bool)
    excl = gm_t | nm_t
    H, W = indexed.shape
    mmk = int(P.get("motif_min_knots", 10)); lmk = int(P.get("line_min_knots", 6)); lmw = int(P.get("line_max_width", 1))
    # colour
    out_lab = pal_lab[indexed]
    tru_lab = truth_lab[truth]
    de = delta_e2000(out_lab.reshape(-1, 3), tru_lab.reshape(-1, 3)).reshape(H, W)
    if gm_t.any():
        import cv2
        blk = lambda a: cv2.resize(a.astype(np.float32), (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
        deb = delta_e2000(blk(out_lab).reshape(-1, 3), blk(tru_lab).reshape(-1, 3))
        gmb = blk(gm_t.astype(np.float32)).reshape(-1) > 0.5
        vals = np.concatenate([de[~gm_t], deb[gmb]])
    else:
        vals = de.reshape(-1)
    out["knot_deltaE_mean"] = float(vals.mean()); out["knot_deltaE_p95"] = float(np.percentile(vals, 95))
    # structure / fidelity on truth indices
    mapped = remap_to_truth(indexed, pal_lab, truth_lab)
    cc = cc_added_missing(mapped, truth, mmk, exclude=excl, line_min_knots=lmk, line_max_width=lmw)
    out.update({"cc_added_area_pct": cc["cc_added_area_pct"], "cc_added_motif": cc["cc_added_motif"], "cc_missing_pct": cc["cc_missing_pct"]})
    guard = rep.get("guard") or {}
    out["guard_reverted_pct"] = guard.get("reverted_area_pct")
    out["cc_missing_by_stage"] = json.dumps(guard.get("by_stage", {}))
    out["edge_iou"] = edge_iou_indexed(mapped, truth)
    out["thin_line_recall"] = thin_line_recall(mapped, truth, lmk, lmw)
    out["line_jaggedness_delta"] = line_jaggedness(indexed, ~gm_t) - line_jaggedness(truth, ~gm_t)
    if gm_t.any():
        rr_L = out_lab[..., 0].astype(np.float32); tr_L = tru_lab[..., 0].astype(np.float32)
        out["banding"] = banding(rr_L, tr_L, gm_t, sigma_knots=8.0, px_per_knot=1.0)
        out["dither_err"] = dither_err(mapped, truth, gm_t)
    if nm_t.any():
        nf = noise_fidelity(mapped, truth, nm_t, int(P.get("noise_cc_max", 6)))
        out["noise_fidelity"] = nf["noise_fidelity"]
    ns_o = noise_stats(mapped, ~excl, 2, 6, 2); ns_t = noise_stats(truth, ~excl, 2, 6, 2)
    out["noise_1knot_delta"] = ns_o["noise_1knot_frac"] - ns_t["noise_1knot_frac"]
    lay = rep.get("layout") or {}
    rg = rep.get("regularize") or {}
    out["free_len_share"] = (rg.get("segment_classes") or {}).get("free_len_share")
    out["regularize_changed_pct"] = rg.get("changed_pct")
    out["repeat_mismatch"] = rg.get("repeat_mismatch")
    if lay.get("symmetry", {}).get("kind", "none") != "none":
        out["symmetry_diff"] = symmetry_mismatch(mapped, lay["symmetry"]["kind"])
    # geometry (synthetic ground truth)
    cg = pair / "corners_gt.json"
    if cg.exists() and (job.stage_dir("rectify") / "corners.json").exists() and not job.corners_override:
        got = np.array(read_json(job.stage_dir("rectify") / "corners.json")["corners"])
        out["corner_err_px"] = float(np.abs(got - np.array(read_json(cg)["corners"])).mean())
    ph = meta.get("photo", {})
    if "fringe_px" in ph and rep.get("fringe"):
        det = rep["fringe"]["strip_px"]
        out["fringe_err_px"] = float(abs(max(det[0], det[1]) - ph["fringe_px"])) if ph.get("fringe") else float(max(det))
    if "non_planar" in ph:
        out["planarity_detect"] = float(bool(rep.get("non_planar") or rep.get("non_planar_internal")) == bool(ph["non_planar"]))
    ig = rep.get("internal_geometry") or {}
    if ig.get("tps_applied") and ig.get("max_dev_knots_before"):
        out["tps_gain"] = 1 - ig["max_dev_knots_after"] / ig["max_dev_knots_before"]
    kg = (rep.get("grid") or {}).get("knot_grid") or {}
    if kg.get("visible") and rep["grid"].get("measured_reed_per_m"):
        out["grid_field_err"] = abs(rep["grid"]["measured_reed_per_m"] - rep["grid"]["reed_per_m"]) / rep["grid"]["reed_per_m"]
    if "blur_knots" in ph and rep.get("grid", {}).get("edge_width_knots") is not None:
        out["edge_width_err"] = rep["grid"]["edge_width_knots"] - ph["blur_knots"]
    if (pair / "illum_gt.npy").exists() and rep.get("illumination"):
        gt = np.load(pair / "illum_gt.npy")
        gt_range = float((gt.max() - gt.min()) * 50.0)  # multiplicative field -> ~L units at L=50
        out["illum_err"] = abs(rep["illumination"].get("range_L", 0.0) - gt_range)
    # layout error (IoU of masks + border/period/symmetry agreement)
    lg = read_json(pair / "layout_gt.json") if (pair / "layout_gt.json").exists() else None
    if lg is not None:
        parts = []
        gm_o = np.load(job.stage_dir("structure") / "gradient_mask.npy")
        if gm_t.any() or gm_o.any():
            parts.append(1 - (_mask_iou(gm_o, gm_t) or 0.0))
        parts.append(0.0 if bool(lay.get("border_accepted")) == bool(lg.get("border")) else 1.0)
        parts.append(0.0 if (lay.get("symmetry", {}).get("kind") == lg.get("symmetry", "none")) else 1.0)
        parts.append(0.0 if bool((lay.get("ground_repeat") or {}).get("v1")) == bool(lg.get("repeat")) else 1.0)
        out["layout_err"] = float(np.mean(parts))
    # class
    if meta.get("design_class"):
        out["class_ok"] = float(rep.get("class_auto") == meta["design_class"])
    out["class_uncertain"] = float("class_uncertain" in rep.get("warnings", []))
    out["class_mixed_flag"] = float(bool(rep.get("class_mixed")))
    out["class_conflict"] = float(bool(rep.get("class_conflict")))
    if meta.get("thin_line_frac") is not None:
        out["thin_line_frac_err"] = (rep.get("class_features") or {}).get("thin_line_frac", 0.0) - meta["thin_line_frac"]
    # edit
    e = rep.get("edit") or {}
    if e:
        ap = e.get("align_px") or {}
        out.update({"edit_align_px_median": ap.get("median"), "edit_align_px_p95": ap.get("p95"),
                    "edit_added_pct": e.get("added_pct"), "edit_missing_pct": e.get("missing_pct"),
                    "edit_rounds": e.get("rounds"), "edit_candidate_spread": e.get("candidate_spread"),
                    "cost_usd": e.get("cost_usd"), "edit_no_gain": float("edit_no_gain" in rep.get("warnings", []))})
    out["edit_routed_out"] = float("edit_thin_lines" in rep.get("warnings", []))
    qc = rep.get("qc") or {}
    out["qc_score"] = qc.get("score"); out["qc_p5"] = qc.get("p5")
    out["time_s"] = (rep.get("timings_s") or {}).get("total")
    out["pure"] = meta.get("pure"); out["mixed"] = meta.get("mixed")
    return out


# ----------------------------------------------------------------------------- batch

def run_pairs(pairs: list[Path], mode: str | None, agent: str | None, regularize: str | None, design_class: str | None,
              runs_dir: Path, profile: str = "wilton_default", tag: str = "") -> list[dict]:
    rows = []
    for p in pairs:
        meta = read_json(p / "meta.json")
        photo = next((p / n for n in ("photo.jpg", "photo.png") if (p / n).exists()), None)
        if photo is None or not (p / "truth.tiff").exists():
            continue
        corners = None
        for cf in ("corners_gt.json", "corners.json"):
            if (p / cf).exists():
                corners = read_json(p / cf)["corners"]
                break
        ov = {}
        if mode:
            ov["design_mode"] = mode
        if agent:
            ov["edit_agent"] = agent
        if regularize:
            ov["regularize"] = regularize
        cls = design_class or meta.get("design_class", "auto")
        density = meta.get("density") or (meta.get("reed_per_m", 400) * meta.get("wefts_per_m", meta.get("reed_per_m", 400)))
        jid = f"eval{tag}_{p.name}_{mode or 'default'}_{agent or ''}_{regularize or 'auto'}".rstrip("_")
        job = Job(job_id=jid, photo_path=photo, width_cm=meta["width_cm"], height_cm=meta["height_cm"], density=density,
                  machine_colors=meta.get("machine_colors", 8), consent_train=False, design_class=cls,
                  profile_name=profile, batch=True, override=ov, corners_override=corners, runs_dir=runs_dir)
        t0 = time.time()
        try:
            rep = run_job(job)
        except JobError as e:
            rows.append({"job_id": jid, "pair": p.name, "design_class": meta.get("design_class"), "mode": mode,
                         "edit_agent": agent, "stopped": e.code, "warnings": "|".join(job.warnings), "time_s": time.time() - t0})
            continue
        r = evaluate_job(job, rep, p, meta)
        r["stopped"] = None
        rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Mean per metric, broken down by design_class (+ total); counts and stop rate."""
    def agg(sel):
        s = {"n": len(sel), "stopped_rate": (sum(1 for r in sel if r.get("stopped")) / len(sel)) if sel else None}
        ok = [r for r in sel if not r.get("stopped")]
        for m in METRIC_COLS:
            vals = [r[m] for r in ok if r.get(m) is not None]
            s[m] = float(np.mean(vals)) if vals else None
        p95 = [r["edit_align_px_p95"] for r in ok if r.get("edit_align_px_p95") is not None]
        s["edit_align_px_p95_p95"] = float(np.percentile(p95, 95)) if p95 else None
        rounds = [r["edit_rounds"] for r in ok if r.get("edit_rounds") is not None]
        s["edit_rounds_median"] = float(np.median(rounds)) if rounds else None
        costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
        s["cost_usd_median"] = float(np.median(costs)) if costs else None
        times = [r["time_s"] for r in ok if r.get("time_s") is not None]
        s["time_s_median"] = float(np.median(times)) if times else None
        return s
    out = {"total": agg(rows)}
    for c in DESIGN_CLASSES:
        sel = [r for r in rows if r.get("design_class") == c]
        if sel:
            out[c] = agg(sel)
    pure = [r for r in rows if r.get("pure")]
    mixed = [r for r in rows if r.get("mixed")]
    if pure:
        out["total"]["class_acc_pure"] = float(np.mean([r["class_ok"] for r in pure if r.get("class_ok") is not None] or [0]))
        out["total"]["class_mixed_false_rate"] = float(np.mean([r["class_mixed_flag"] for r in pure if r.get("class_mixed_flag") is not None] or [0]))
    if mixed:
        out["total"]["class_mixed_recall"] = float(np.mean([r["class_mixed_flag"] for r in mixed if r.get("class_mixed_flag") is not None] or [0]))
    return out


def release_check(rows: list[dict], profile: dict, mode: str) -> dict:
    """Release criterion for one mode, measured on the jobs where that mode actually ran."""
    sel = [r for r in rows if not r.get("stopped") and r.get("mode") == mode]
    fails = []
    if not sel:
        return {"mode": mode, "n": 0, "pass": False, "failures": ["no jobs"]}
    added_max = int(profile.get("edit_added_motif_max", 0)) if mode == "edit" else 0
    if any((r.get("cc_added_motif") or 0) > added_max for r in sel):
        fails.append("cc_added_motif")
    def mean_of(k, pred=lambda r: True):
        v = [r[k] for r in sel if r.get(k) is not None and pred(r)]
        return float(np.mean(v)) if v else None
    checks = [("cc_added_area_pct", lambda v: v <= RELEASE["cc_added_area_pct"]),
              ("cc_missing_pct", lambda v: v <= RELEASE["cc_missing_pct"]),
              ("thin_line_recall", lambda v: v >= RELEASE["thin_line_recall"]),
              ("knot_deltaE_mean", lambda v: v < float(profile.get("eval_deltaE_max", 3.0))),
              ("knot_deltaE_p95", lambda v: v < 2 * float(profile.get("eval_deltaE_max", 3.0))),
              ("repeat_mismatch", lambda v: v <= RELEASE["repeat_mismatch"]),
              ("banding", lambda v: v <= RELEASE["banding_max"]),
              ("noise_fidelity", lambda v: v >= RELEASE["noise_fidelity_min"])]
    for k, ok in checks:
        v = mean_of(k)
        if v is not None and not ok(v):
            fails.append(k)
    jag = mean_of("line_jaggedness_delta", lambda r: r.get("regularize"))
    if jag is not None and jag > RELEASE["jag_max"]:
        fails.append("line_jaggedness_delta")
    res = {"mode": mode, "n": len(sel), "failures": fails}
    if mode == "edit":
        p95 = [r["edit_align_px_p95"] for r in sel if r.get("edit_align_px_p95") is not None]
        if p95 and np.percentile(p95, 95) > float(profile.get("edit_align_max", 6.0)):
            fails.append("edit_align_px_p95")
        rounds = [r["edit_rounds"] for r in sel if r.get("edit_rounds") is not None]
        if rounds and np.median(rounds) > float(profile.get("edit_max_rounds", 3)) - 1:
            fails.append("edit_rounds")
        g = derive_edit_gates([r.get("edit_added_pct") or 0 for r in sel], [r.get("cc_added_area_pct") or 0 for r in sel],
                              [r.get("edit_missing_pct") or 0 for r in sel], [r.get("cc_missing_pct") or 0 for r in sel],
                              float(profile.get("edit_gate_corr_min", 0.6)))
        res["edit_gates"] = g
        if g["edit_gate_source"] != "eval":
            fails.append("edit_gate_corr")
        res["edit_route_out_rate"] = float(np.mean([r.get("edit_routed_out") or 0 for r in rows if not r.get("stopped")]))
        res["edit_no_gain_rate"] = float(np.mean([r.get("edit_no_gain") or 0 for r in sel]))
    res["pass"] = not fails
    return res


def regression_check(cur: dict, prev: dict, tol: float = 0.10) -> list[str]:
    """Metrics where any class got > 10 % worse than the previous run (direction-aware)."""
    higher_better = {"thin_line_recall", "edge_iou", "noise_fidelity", "qc_score", "class_ok", "planarity_detect", "tps_gain"}
    worse = []
    for cls, s in cur.items():
        p = prev.get(cls)
        if not p:
            continue
        for m in METRIC_COLS:
            a, b = s.get(m), p.get(m)
            if a is None or b is None or abs(b) < 1e-9:
                continue
            rel = (a - b) / abs(b)
            if (m in higher_better and rel < -tol) or (m not in higher_better and rel > tol):
                worse.append(f"{cls}:{m}:{b:.4g}->{a:.4g}")
    return worse


def write_csv(rows: list[dict], path: Path):
    cols = ["job_id", "pair", "design_class", "class_detected", "mode", "mode_requested", "edit_agent", "regularize", "qc_status",
            "stopped", "warnings", "pure", "mixed", "cc_missing_by_stage"] + METRIC_COLS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ids", help="file with one pair id per line (default: all)")
    ap.add_argument("--mode", choices=["quantize", "segment", "edit"])
    ap.add_argument("--agent")
    ap.add_argument("--regularize", choices=["on", "off"])
    ap.add_argument("--class", dest="design_class", help="auto to test class detection; default: meta design_class")
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--runs", default=str(RUNS_DIR / "eval"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / f"run_{time.strftime('%Y%m%d_%H%M')}.csv"))
    ap.add_argument("--prev", help="previous summary json for the regression check")
    ap.add_argument("--write-gates", action="store_true", help="write edit_added_max/edit_missing_max to the profile when corr passes")
    a = ap.parse_args(argv)
    pairs = sorted(p for p in Path(a.pairs).iterdir() if p.is_dir())
    if a.ids:
        ids = set(Path(a.ids).read_text().split())
        pairs = [p for p in pairs if p.name in ids]
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    rows = run_pairs(pairs, a.mode, a.agent, a.regularize, a.design_class, Path(a.runs), a.profile)
    write_csv(rows, Path(a.out))
    summary = summarize(rows)
    modes = sorted({r.get("mode") for r in rows if r.get("mode")})
    summary["release"] = {m: release_check(rows, P, m) for m in modes}
    if a.prev and Path(a.prev).exists():
        summary["regressions"] = regression_check(summary, read_json(a.prev))
    sj = Path(a.out).with_suffix(".summary.json")
    sj.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["release"], indent=2, default=str))
    print(f"rows: {len(rows)} -> {a.out}\nsummary -> {sj}")
    if a.write_gates and "edit" in summary["release"]:
        g = summary["release"]["edit"].get("edit_gates", {})
        if g.get("edit_gate_source") == "eval":
            sys.path.insert(0, str(ROOT / "tools"))
            from archive import write_profile_fields
            write_profile_fields({"edit_added_max": round(g["edit_added_max"], 5), "edit_missing_max": round(g["edit_missing_max"], 5),
                                  "edit_gate_source": "eval"}, a.profile)
            print("profile updated: edit gates from eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
