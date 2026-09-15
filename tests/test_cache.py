import yaml

from knotvision import STAGES
from knotvision.job import CONFIG_DIR, Job, read_json
from tests.conftest import make_job, run


def test_second_run_runs_nothing(geometric_pair, tmp_root):
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache1")
    run(job)
    job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache1")
    rep = run(job2)
    assert set(job2.state.get("stages_ran", [])) <= {"qc", "export"}  # qc/export are cheap and always refresh
    assert all(job2.timings[s] == 0.0 for s in ("intake", "rectify", "structure", "palette", "design", "production", "regularize", "dither"))
    assert rep["warnings"] == job.state["warnings"] or set(rep["warnings"]) == set(job.state["warnings"])


def test_rerun_from_production_with_palette(geometric_pair, tmp_root):
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache2")
    run(job)
    j2 = Job.load(job.run_dir)
    j2.palette_override = [{"rgb": [200, 180, 150]}, {"rgb": [60, 40, 30]}, {"rgb": [150, 60, 40]}]
    j2.confirm_palette = True
    j2.params = {}
    rep = run(j2, from_stage="palette")
    ran = j2.state["stages_ran"]
    assert "structure" not in ran and "rectify" not in ran and "palette" in ran and "production" in ran
    assert rep["palette_confirmed"] is True and rep["colors"]["palette_used"] <= 3


def test_class_file_change_reruns_after_structure(geometric_pair, tmp_root):
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache3")
    run(job)
    p = CONFIG_DIR / "classes" / "geometric.yaml"
    orig = p.read_text()
    try:
        p.write_text("reg_tol_knots: 0.9\n")
        job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache3")
        run(job2)
        ran = job2.state["stages_ran"]
        assert "regularize" in ran and "rectify" not in ran
    finally:
        p.write_text(orig)


def test_agent_change_reruns_only_design_edit(geometric_pair, tmp_root, mock_agent):
    a1 = mock_agent("t_mock_c1", mock={"mode": "identity"}, edit_n_candidates=1, edit_max_rounds=1)
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache4",
                   override={"design_mode": "edit", "edit_agent": a1})
    run(job)
    a2 = mock_agent("t_mock_c2", mock={"mode": "identity"}, edit_n_candidates=2, edit_max_rounds=1)
    job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache4",
                    override={"design_mode": "edit", "edit_agent": a2})
    rep = run(job2)
    ran = job2.state["stages_ran"]
    assert "design" in ran and "palette" not in ran and "structure" not in ran
    assert rep["edit"]["baseline_cached"] is True
    # edit_seed is part of the key
    job3 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache4",
                    override={"design_mode": "edit", "edit_agent": a2, "edit_seed": 5})
    run(job3)
    assert "design" in job3.state["stages_ran"]


def test_stage_warnings_restored_from_cache(geometric_pair, tmp_root):
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache5")
    run(job)
    w1 = set(job.state["warnings"])
    job2 = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="cache5")
    run(job2)
    assert set(job2.state["warnings"]) == w1
