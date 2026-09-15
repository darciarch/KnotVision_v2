import json

import numpy as np
import pytest

from knotvision import qc as Q
from knotvision.job import MODELS_DIR, agent_hash, read_json
from tests.conftest import make_job, run


def _calib(features, configs, target=4, accept=0.8, reject=0.3, extra=None):
    n = len(features)
    c = {"features": features, "dropped_features": {}, "class_levels": ["classic", "geometric", "gradient", "other"],
         "mode_levels": ["segment", "quantize", "edit"], "calibrated_configs": configs, "agent_hashes": {},
         "model": {"type": "logistic_l2", "C": 1.0, "mean": [0.5] * n, "std": [0.2] * n,
                   "coef": [1.0] * n + [0.0] * 4 + [0.0] * 3, "intercept": 2.0},
         "model_p5": None, "target": target, "qc_accept": accept, "qc_reject": reject, "cv_auc": 0.8,
         "cv_auc_by_class": {}, "cv_auc_by_config": {}, "class_unreliable": [], "config_unreliable": [],
         "model_hashes": {"segment": None, "quantize": "x"}, "date": "2026-09-15"}
    c.update(extra or {})
    return c


@pytest.fixture
def calib_file(monkeypatch, tmp_path):
    """Point MODELS_DIR at a temp dir and let the test write qc_calib.json there."""
    from knotvision import job as jobmod
    monkeypatch.setattr(jobmod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(Q, "MODELS_DIR", tmp_path)

    def write(c):
        (tmp_path / "qc_calib.json").write_text(json.dumps(c))
        return tmp_path / "qc_calib.json"
    return write


def test_rerender_edge_iou_high_on_truth(geometric_pair):
    d = geometric_pair["design"]
    rr = Q.rerender(d.indexed, d.palette_rgb, px=2)
    from knotvision.metrics import edge_fidelity
    flat = np.repeat(np.repeat(d.palette_rgb[d.indexed], 2, 0), 2, 1)
    assert edge_fidelity(rr, flat) > 0.9


def test_no_calibration_means_needs_review(geometric_run):
    qc = geometric_run["report"]["qc"]
    assert qc["status"] == "needs_review@none" and qc["score"] is None and qc["decision"] == "needs_review"
    assert "qc_uncalibrated" in geometric_run["report"]["warnings"]
    assert set(qc["features"]) == set(Q.CORE_FEATURES)


def test_core_feature_merge_and_null_rules(geometric_run):
    job = geometric_run["job"]
    raw = dict(job.state["qc_raw"])
    f = Q.core_features(raw, job)
    assert abs(f["postprocess_changed_pct"] - ((raw["cleanup_changed_pct"] or 0) + (raw["regularize_changed_pct"] or 0) + (raw["guard_reverted_pct"] or 0))) < 1e-9
    raw2 = {**raw, "symmetry_consistency": None, "repeat_mismatch_raw": None}
    assert Q.core_features(raw2, job)["structure_mismatch"] is None
    raw3 = {**raw, "symmetry_consistency": 0.1, "repeat_mismatch_raw": 0.3}
    assert Q.core_features(raw3, job)["structure_mismatch"] == 0.3
    raw4 = {**raw, "edge_density": 0.0}
    f4 = Q.core_features(raw4, job)
    assert f4["rerender_edge_iou"] is None and f4["line_jaggedness"] is None


def test_score_null_is_standardized_zero_and_reasons():
    feats = ["rerender_edge_iou", "palette_deltaE_p95", "banding"]
    calib = _calib(feats, [])
    p, p5, reasons = Q.score({"rerender_edge_iou": 0.5, "palette_deltaE_p95": 0.1, "banding": None}, calib, "geometric", "quantize")
    # z: 0, -2, 0 -> logit = 2 - 2 = 0 -> p = 0.5
    assert abs(p - 0.5) < 1e-9 and p5 is None
    assert reasons == ["palette_deltaE_p95=0.1"]
    p2, _, _ = Q.score({"rerender_edge_iou": 0.9, "palette_deltaE_p95": 0.9, "banding": 0.9}, calib, "geometric", "quantize")
    assert p2 > p


def test_calibrated_decision_status_and_uncalibrated_config(geometric_pair, tmp_root, calib_file):
    feats = ["rerender_edge_iou", "palette_deltaE_p95"]
    calib_file(_calib(feats, [["geometric", "quantize", None, True]]))
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_calib")
    rep = run(job)
    qc = rep["qc"]
    assert qc["target"] == 4 and qc["status"].endswith("@4") and qc["score"] is not None
    assert "mode_uncalibrated" not in qc["hard_gate_failures"]
    # regularize off -> different config tuple -> uncalibrated
    job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_calib_off", override={"regularize": "off"})
    rep2 = run(job2)
    assert "mode_uncalibrated" in rep2["qc"]["hard_gate_failures"] and rep2["qc"]["status"].startswith("needs_review")


def test_hash_mismatch_and_unreliable(geometric_pair, tmp_root, calib_file):
    feats = ["rerender_edge_iou"]
    calib_file(_calib(feats, [["geometric", "quantize", None, True]], extra={"class_unreliable": ["geometric"]}))
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_unrel")
    rep = run(job)
    assert rep["qc"]["decision"] == "needs_review"


def test_edit_hard_gates_and_agent_hash(geometric_pair, tmp_root, calib_file, mock_agent):
    agent = mock_agent("t_mock_qc", mock={"mode": "identity"})
    feats = ["rerender_edge_iou"]
    calib_file(_calib(feats, [["geometric", "edit", agent, True]], extra={"agent_hashes": {agent: "stale"}}))
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_edit_qc",
                   override={"design_mode": "edit", "edit_agent": agent})
    rep = run(job)
    assert rep["design_mode"] == "edit"
    assert "mode_uncalibrated" in rep["qc"]["hard_gate_failures"]
    assert "qc_calib_stale" in rep["warnings"]
    # correct hash -> calibrated
    calib_file(_calib(feats, [["geometric", "edit", agent, True]], extra={"agent_hashes": {agent: agent_hash(agent)}}))
    job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_edit_qc2",
                    override={"design_mode": "edit", "edit_agent": agent})
    rep2 = run(job2)
    assert "mode_uncalibrated" not in rep2["qc"]["hard_gate_failures"]


def test_hard_gates_list(geometric_run):
    job = geometric_run["job"]
    fails = Q.hard_gates(job)
    for w in job.warnings:
        if w in Q.HARD_GATE_WARNINGS:
            assert w in fails


def test_preview_written(geometric_run):
    import cv2
    p = cv2.imread(str(geometric_run["job"].stage_dir("qc") / "preview.png"))
    assert p is not None and p.shape[1] == 7 * 400
