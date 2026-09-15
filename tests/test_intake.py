import numpy as np
import pytest
import yaml

from knotvision import intake
from knotvision.job import CONFIG_DIR, Job, JobError, load_agent_config, load_yaml, resolve_params, setup_block
from tests.conftest import make_job, run


def test_density_definitions():
    g = intake.grid_from_density(160, 230, 160000, 400, "knots_m2")
    assert (g["W"], g["H"]) == (640, 920) and g["wefts_per_m"] == 400 and g["grid_verified"]
    g = intake.grid_from_density(160, 230, 380, 420, "wefts_per_m")
    assert g["reed_per_m"] == 420 and g["wefts_per_m"] == 380 and g["H"] == 874
    g = intake.grid_from_density(100, 100, [400, 380], 400, "reed_x_wefts")
    assert (g["W"], g["H"]) == (400, 380)


def test_grid_input_mismatch():
    with pytest.raises(JobError) as e:
        intake.grid_from_density(100, 100, [420, 380], 400, "reed_x_wefts")
    assert e.value.code == "grid_input_mismatch"


def test_reed_null_square_assumption_and_unverified():
    g = intake.grid_from_density(100, 100, 160000, None, "knots_m2")
    assert g["reed_per_m"] == 400 and g["wefts_per_m"] == 400 and not g["grid_verified"]
    assert "grid_unverified" in g["warnings"]
    with pytest.raises(JobError):
        intake.grid_from_density(100, 100, 400, None, "wefts_per_m")


def test_grid_rounding_warning():
    g = intake.grid_from_density(100.11, 100, 160000, 400, "knots_m2")
    assert "grid_rounding" in g["warnings"]


def _base_job(tmp_path, **kw):
    photo = tmp_path / "p.jpg"
    if not photo.exists():
        import cv2
        cv2.imwrite(str(photo), np.full((64, 64, 3), 128, np.uint8))
    args = dict(job_id="j", photo_path=photo, width_cm=100, height_cm=100, density=160000, machine_colors=8,
                consent_train=True, design_class="classic", runs_dir=tmp_path / "runs")
    args.update(kw)
    j = Job(**args)
    j.load_profile()
    return j


def test_machine_colors_not_allowed(tmp_path):
    j = _base_job(tmp_path, machine_colors=7)
    with pytest.raises(JobError) as e:
        intake.validate(j)
    assert e.value.code == "machine_colors_not_allowed"


def test_class_required_in_real_job_auto_only_batch(tmp_path):
    j = _base_job(tmp_path, design_class="auto")
    with pytest.raises(JobError) as e:
        intake.validate(j)
    assert e.value.code == "class_required"
    j = _base_job(tmp_path, design_class="auto", batch=True)
    intake.validate(j)
    assert j.class_source == "auto"
    j = _base_job(tmp_path, design_class="pictorial")
    with pytest.raises(JobError) as e:
        intake.validate(j)
    assert e.value.code == "class_unsupported"


def test_photo_long_px_below_rec_is_only_a_warning(geometric_run):
    rep = geometric_run["report"]
    assert "photo_long_px_below_rec" in rep["warnings"]
    assert rep["photo"]["long_px_below_rec"] is True
    assert rep["qc"]["status"] is not None  # job ran to the end


def test_parameter_layering_with_agent_layer():
    profile = {"design_mode": "quantize", "edit_seed": 0, "edit_max_rounds": 3, "reg_tol_knots": 0.7,
               "allowed_machine_colors": [8]}
    cls = {"design_mode": "edit", "edit_agent": "x", "reg_tol_knots": 0.5}
    agent = {"backend": "mock", "edit_max_rounds": 5, "reg_tol_knots": 9.9, "num_inference_steps": 4}
    p = resolve_params(profile, cls, {"edit_seed": 42}, agent)
    assert p["design_mode"] == "edit" and p["edit_agent"] == "x"
    assert p["edit_max_rounds"] == 5           # agent overrides edit_* from the class/profile
    assert p["reg_tol_knots"] == 0.5           # agent may not touch non-edit fields
    assert p["edit_seed"] == 42                # job override beats the agent
    assert p["agent"]["num_inference_steps"] == 4
    assert p["design_mode_class_default"] == "edit"
    # job override wins over agent too
    p = resolve_params(profile, cls, {"edit_max_rounds": 1}, agent)
    assert p["edit_max_rounds"] == 1


def test_class_may_not_set_loom_fields():
    with pytest.raises(JobError) as e:
        resolve_params({"allowed_machine_colors": [8]}, {"reed_per_m": 400}, {})
    assert e.value.code == "class_overrides_loom_field"


def test_edit_requires_agent(tmp_path):
    j = _base_job(tmp_path, override={"design_mode": "edit"})
    with pytest.raises(JobError) as e:
        j.resolve()
    assert e.value.code == "edit_agent_missing"


def test_muse_requires_allow_api(tmp_path):
    j = _base_job(tmp_path, override={"design_mode": "edit", "edit_agent": "muse_loop"})
    assert j.profile.get("allow_api") is False
    j.resolve()
    with pytest.raises(JobError) as e:
        intake.validate_edit(j)
    assert e.value.code == "api_not_allowed"


def test_setup_block_fields(geometric_run):
    s = geometric_run["report"]["setup"]
    for k in ("reed_per_m", "density_definition", "dither_pattern", "yarn_library", "motif_min_knots", "noise_scale",
              "qc_calib", "sam", "segment_ckpt", "photo_survey", "edit_backends"):
        assert k in s
    assert set(s["edit_backends"]) == {"qwen_edit_2511", "qwen_edit_2509", "muse_image"}
    assert s["reed_per_m"] in ("profile", "assumed")


def test_yarn_csv_loaded(tmp_path):
    rows, lab = intake.load_yarn_csv(CONFIG_DIR / "yarns" / "sample.csv")
    assert len(rows) == 8 and lab.shape == (8, 3) and rows[0]["yarn_code"] == "12"
