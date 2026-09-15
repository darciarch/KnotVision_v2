"""Edit mode with the mock backend (no GPU). See agents/backends/mock.py for the knobs."""
import numpy as np
import pytest

from knotvision import PX_PER_KNOT
from knotvision.agents.loop import candidate_seed, tile_plan, blend_tiles
from knotvision.design_edit import working_resolution
from knotvision.job import read_json
from tests.conftest import make_job, run


def _edit_job(pair, tmp_root, agent, job_id, **kw):
    return make_job(pair["dir"], tmp_root / "runs", "geometric", job_id=job_id,
                    override={"design_mode": "edit", "edit_agent": agent, **kw.pop("override", {})}, **kw)


def test_working_resolution_and_tiling():
    P = {"edit_px_per_knot_target": 2.0, "edit_px_per_knot_min": 1.0, "edit_tiling": "auto", "edit_tile_px": 1024}
    wr0 = working_resolution(400, 600, P, 1536)
    assert not wr0["tiled"] and wr0["px_per_knot"] == 2.0          # 1200 px long side fits a single pass
    wr = working_resolution(400, 1000, P, 1536)                     # single pass would be 1.536 px/knot
    assert wr["tiled"] and wr["px_per_knot"] == 2.0 and wr["n_tiles"] > 1 and wr["w"] % 16 == 0 and wr["h"] % 16 == 0
    wr2 = working_resolution(400, 1000, {**P, "edit_tiling": "off"}, 1536)
    assert not wr2["tiled"] and wr2["px_per_knot"] < 2.0 and not wr2["low_res"]
    wr3 = working_resolution(1000, 1600, {**P, "edit_tiling": "off"}, 1536)
    assert wr3["low_res"]
    boxes = tile_plan(2048, 1024, 1024, 128)
    assert len(boxes) == 3 and all(x1 - x0 <= 1024 for x0, _, x1, _ in boxes)
    tiles = [np.full((y1 - y0, x1 - x0, 3), 100, np.uint8) for x0, y0, x1, y1 in boxes]
    out = blend_tiles(tiles, boxes, 2048, 1024, 128)
    assert out.shape == (1024, 2048, 3) and np.abs(out.astype(int) - 100).max() <= 1


def test_seed_formula():
    assert candidate_seed(7, 2, 3, 4) == 7 + 2000 + 30 + 4


def test_baseline_first_cached_across_agents(geometric_pair, tmp_root, mock_agent):
    a1 = mock_agent("t_mock_a1", mock={"mode": "identity"})
    job = _edit_job(geometric_pair, tmp_root, a1, "edit_base1")
    rep = run(job)
    sd = job.stage_dir("design")
    assert (sd / "baseline" / "baseline_rgb.png").exists() and (sd / "baseline" / "ref_indexed.npy").exists()
    assert rep["edit"]["baseline_cached"] is False
    assert (sd / "edit" / "round_0" / "cand_0.png").exists() and (sd / "edit" / "prompt_0.txt").exists()
    assert (sd / "edit" / "select.json").exists()
    a2 = mock_agent("t_mock_a2", mock={"mode": "identity"}, edit_n_candidates=1)
    job2 = _edit_job(geometric_pair, tmp_root, a2, "edit_base1", override={"edit_agent": a2})
    rep2 = run(job2)
    assert rep2["edit"]["baseline_cached"] is True and rep2["edit"]["agent"] == a2
    import cv2
    out = cv2.imread(str(sd / "design_rgb.png"))
    g = job.state["grid"]
    assert out.shape[:2] == (PX_PER_KNOT * g["H"], PX_PER_KNOT * g["W"])
    ein = cv2.imread(str(sd / "edit_input.png"))
    assert ein.shape[0] % 16 == 0 and ein.shape[1] % 16 == 0


def test_identity_backend_passes_and_stops_first_round(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_id", mock={"mode": "identity"}, edit_max_rounds=3)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_identity")
    rep = run(job)
    e = rep["edit"]
    assert rep["design_mode"] == "edit" and e["rounds"] == 1 and e["selected"].startswith("round_0")
    assert e["align_px"]["p95"] <= 1.5 and e["added_pct"] <= 0.01 and e["missing_pct"] <= 0.02
    assert e["cost_usd"] == 0.0 and e["api_egress"] is False and rep["deterministic"] is True
    for g in ("edit_drift", "edit_added", "edit_missing", "edit_no_gain", "edit_fallback"):
        assert g not in rep["qc"]["hard_gate_failures"]


def test_shift_is_recovered_by_alignment(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_shift", mock={"mode": "shift", "shift_px": [3, 0]}, edit_n_candidates=1, edit_max_rounds=1)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_shift")
    rep = run(job)
    e = rep["edit"]
    assert e["align_px"]["p95"] <= 2.5 and "edit_drift" not in rep["qc"]["hard_gate_failures"]


def test_large_shift_triggers_edit_drift(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_drift", mock={"mode": "shift", "shift_px": [40, 25]}, edit_n_candidates=1, edit_max_rounds=1)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_drift", override={"edit_align_max": 1.0, "edit_gain_tol": 1.0})
    rep = run(job)
    assert "edit_drift" in rep["qc"]["hard_gate_failures"] or "edit_no_gain" in rep["qc"]["hard_gate_failures"]


def test_added_block_raises_added_pct_and_feedback_prompt(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_add", mock={"mode": "add_block", "block": [60, 60, 140, 140], "color": [255, 0, 255]},
                   edit_n_candidates=1, edit_max_rounds=2)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_added", override={"edit_gain_tol": 1.0})
    rep = run(job)
    sd = job.stage_dir("design") / "edit"
    feats = read_json(sd / "round_0" / "cand_0_features.json")
    assert feats["added_pct"] > 0.0
    prod = read_json(job.stage_dir("production") / "production_stats.json")
    assert prod["edit_added_pct"] > 0.0
    if feats["added_pct"] > job.params["edit_added_max"]:
        assert "edit_added" in rep["qc"]["hard_gate_failures"]
        assert "Remove any element that is not in the reference photo" in (sd / "prompt_1.txt").read_text()


def test_removed_motif_raises_missing_and_gate(geometric_pair, tmp_root, mock_agent):
    d = geometric_pair["design"]
    a = mock_agent("t_mock_rm", mock={"mode": "remove_block", "block": [40, 40, 200, 200]}, edit_n_candidates=1, edit_max_rounds=1)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_missing", override={"edit_gain_tol": 1.0, "edit_missing_max": 0.001})
    rep = run(job)
    prod = read_json(job.stage_dir("production") / "production_stats.json")
    assert prod["edit_missing_pct"] > 0.0
    assert "edit_missing" in rep["qc"]["hard_gate_failures"]


def test_reference_is_baseline_not_coarse_labels(geometric_run):
    """edit_added_pct against ref_indexed (same quantization path) is lower than against coarse labels."""
    from knotvision.metrics import edit_added_missing
    job = geometric_run["job"]
    raw = np.load(job.stage_dir("production") / "indexed_raw.npy")
    coarse = np.load(job.stage_dir("structure") / "coarse_labels.npy")
    a_ref, _ = edit_added_missing(raw, raw, 10)
    a_coarse, _ = edit_added_missing(raw, coarse, 10)
    assert a_ref <= a_coarse


def test_no_gain_uses_baseline(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_degrade", mock={"mode": "degrade", "sigma": 90}, edit_n_candidates=1, edit_max_rounds=1)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_nogain")
    rep = run(job)
    assert "edit_no_gain" in rep["warnings"] and "edit_no_gain" in rep["qc"]["hard_gate_failures"]
    assert rep["edit"]["selected"] == "baseline"
    import cv2
    sd = job.stage_dir("design")
    assert np.array_equal(cv2.imread(str(sd / "design_rgb.png")), cv2.imread(str(sd / "baseline" / "baseline_rgb.png")))


def test_backend_failure_three_times_falls_back(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_fail", mock={"mode": "identity", "fail_times": 3}, edit_n_candidates=1, edit_max_rounds=1)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_fail")
    rep = run(job)
    assert "edit_fallback" in rep["warnings"] and rep["design_mode"] == "quantize"
    assert "edit_fallback" in rep["qc"]["hard_gate_failures"]
    assert rep["design_mode_requested"] == "edit"


def test_max_rounds_respected_and_ties_lowest(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_rounds", mock={"mode": "add_block", "block": [10, 10, 120, 120], "color": [0, 255, 0]},
                   edit_n_candidates=2, edit_max_rounds=2)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_rounds", override={"edit_gain_tol": 1.0})
    rep = run(job)
    sel = read_json(job.stage_dir("design") / "edit" / "select.json")
    assert rep["edit"]["rounds"] <= 2
    scores = sel["scores"]
    top = [s for s in scores if abs(s["score"] - scores[0]["score"]) < 1e-9]
    best = min(top, key=lambda s: (s["round"], s["candidate"]))
    assert (sel["round"], sel["candidate"]) == (best["round"], best["candidate"])


def test_timeout(geometric_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_timeout", mock={"mode": "add_block", "block": [10, 10, 120, 120]}, edit_n_candidates=2, edit_max_rounds=3)
    job = _edit_job(geometric_pair, tmp_root, a, "edit_timeout", override={"edit_max_wall_s": 0.0, "edit_gain_tol": 1.0})
    rep = run(job)
    assert "edit_timeout" in rep["warnings"] or "edit_fallback" in rep["warnings"]


def test_class_additions_in_prompt(gradient_pair, tmp_root, mock_agent):
    a = mock_agent("t_mock_prompt", mock={"mode": "identity"}, edit_n_candidates=1, edit_max_rounds=1)
    job = make_job(gradient_pair["dir"], tmp_root / "runs", "gradient", job_id="edit_prompt",
                   override={"design_mode": "edit", "edit_agent": a})
    run(job)
    p = (job.stage_dir("design") / "edit" / "prompt_0.txt").read_text()
    assert "smooth color transition" in p and "hex:" in p


def test_muse_reporting_fields():
    from knotvision.agents.backends.muse_image import MuseImageBackend
    import os
    os.environ["MUSE_API_KEY"] = "test"
    try:
        b = MuseImageBackend({"resolution": "2K"}, provider="fal", price_usd=0.01)
        assert b.supports_seed is False and b.api_egress is True and b.price_usd == 0.01
    finally:
        del os.environ["MUSE_API_KEY"]
