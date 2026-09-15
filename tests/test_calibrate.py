import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from calibrate_qc import calibrate, choose_target, select_features, thresholds  # noqa: E402
from capacity import capacity  # noqa: E402
from knotvision.qc import CORE_FEATURES  # noqa: E402


def _rows(n=150, seed=0, edit_agent=None, n_edit=0):
    """Synthetic score sheet: score correlates with rerender_edge_iou and -palette_deltaE_p95."""
    rng = np.random.RandomState(seed)
    classes = ["classic"] * 45 + ["geometric"] * 45 + ["gradient"] * 30 + ["distressed"] * 20 + ["minimal"] * 10
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        iou = rng.uniform(0.2, 0.9)
        de = rng.uniform(2, 20)
        latent = 3 * iou - 0.08 * de + rng.normal(0, 0.35)
        score = int(np.clip(np.round(2.5 + 1.6 * latent), 1, 5))
        feats = {"rerender_edge_iou": iou, "palette_deltaE_p95": de, "rerender_color_deltaE_p95": rng.uniform(2, 12),
                 "line_jaggedness": rng.uniform(0, 0.5), "noise_1knot_pct": rng.uniform(0, 0.05),
                 "postprocess_changed_pct": rng.uniform(0, 0.05), "structure_mismatch": None if rng.uniform() < 0.6 else rng.uniform(0, 0.3),
                 "banding": rng.uniform(0, 3) if cls == "gradient" else None,
                 "edge_width_knots": rng.uniform(0.5, 2.5), "illum_residual": rng.uniform(0, 2)}
        mode = "edit" if (edit_agent and i < n_edit) else "quantize"
        rows.append({"job_id": f"j{i}", "design_class": cls, "mode": mode, "edit_agent": edit_agent if mode == "edit" else None,
                     "regularize": True, "designer_score": float(score),
                     "designer_score_2": float(np.clip(score + rng.choice([-1, 0, 0, 0, 1]), 1, 5)) if i < 12 else None,
                     "fix_minutes": float(rng.lognormal(3.0, 0.8)), "fix_type": rng.choice(["color", "geometry", "motif", "regularity", "sheen"]),
                     "features": feats})
    return rows


def test_target_selection():
    assert choose_target(np.array([5] * 20 + [3] * 20)) == 5
    assert choose_target(np.array([4] * 20 + [3] * 20)) == 4
    assert choose_target(np.array([4] * 20 + [3] * 5)) is None


def test_feature_selection_priority_and_null_drop():
    rows = _rows(60)
    feats, dropped = select_features(rows, 4)
    assert feats == list(CORE_FEATURES[:4])
    assert dropped.get("banding") == "null_share" or "banding" not in feats
    assert all(v == "n_cap" for k, v in dropped.items() if k in CORE_FEATURES[4:] and k != "banding" and k != "structure_mismatch")


def test_calibration_end_to_end():
    rows = _rows(150)
    res = calibrate(rows, {"edit_gate_source": "assumed"})
    assert res["status"] == "ok", res
    assert res["target"] in (3, 4, 5) and res["cv_auc"] >= 0.75
    n_null = sum(1 for v in res["dropped_features"].values() if v == "null_share")
    assert len(res["features"]) == min(10, 150 // 15) - n_null   # > 50 % null features leave the model
    assert "other" in res["class_levels"] and "classic" in res["class_levels"]
    assert res["mode_levels"] == ["segment", "quantize", "edit"]
    assert res["intra_rater_kappa"] is not None
    assert 0 <= res["qc_reject"] <= res["qc_accept"] <= 1
    assert ["classic", "quantize", None, True] in res["calibrated_configs"]
    m = res["model"]
    assert len(m["coef"]) == len(res["features"]) + len(res["class_levels"]) + len(res["mode_levels"])
    assert isinstance(res["score_hist"], dict) and isinstance(res["fix_type_hist"], dict)


def test_p5_only_with_enough_samples():
    rows = _rows(150)
    for r in rows:
        r["designer_score"] = min(r["designer_score"], 4.0)
    res = calibrate(rows, {})
    assert res["model_p5"] is None


def test_edit_config_needs_eval_gates_and_quota():
    rows = _rows(160, edit_agent="qwen2511_loop", n_edit=40)
    res_assumed = calibrate(rows, {"edit_gate_source": "assumed"})
    assert not any(c[1] == "edit" for c in res_assumed["calibrated_configs"])
    res_eval = calibrate(rows, {"edit_gate_source": "eval"})
    edit_cfgs = [c for c in res_eval["calibrated_configs"] if c[1] == "edit"]
    assert edit_cfgs and all(c[2] == "qwen2511_loop" for c in edit_cfgs)
    assert "qwen2511_loop" in res_eval["agent_hashes"]


def test_thresholds_rule():
    prob = np.linspace(0, 1, 100)
    y = (prob > 0.6).astype(int)
    scores = np.where(prob < 0.3, 2, 4)
    acc, rej = thresholds(prob, y, scores)
    # accept = lowest threshold whose selected set has precision >= 0.9 (just below the 0.6 label edge)
    assert 0.5 <= acc <= 0.61 and rej <= acc and rej >= 0.25


def test_capacity_medians_by_mode():
    rows = _rows(120, edit_agent="a", n_edit=30)
    for r in rows:
        r["qc_status"] = "accepted@4" if r["designer_score"] >= 4 else "needs_review@4"
        r["cost_usd"] = 0.02 if r["mode"] == "edit" else None
    c = capacity(rows, 8.0)
    assert c["total"]["median_min"] > 0 and c["total"]["p90_min"] >= c["total"]["median_min"]
    assert c["total"]["jobs_per_day_bad"] <= c["total"]["jobs_per_day_typical"]
    assert "edit/a" in c["by_mode"] and "quantize" in c["by_mode"]
    assert c["edit"]["cost_usd_median"] == 0.02
    assert set(c["accepted_share"]["by_class"]) >= {"classic", "geometric"}
