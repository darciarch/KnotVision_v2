import json

import numpy as np
import pytest

from knotvision import structure as S
from knotvision.job import MODELS_DIR, JobError, read_json
from tests.conftest import design_of, make_job, photo_of, run, write_pair


def _run_auto(tmp_root, name, template, seed, **feat):
    d = design_of(template, seed=seed, **feat)
    r = photo_of(d, seed=seed)
    p = write_pair(d, r, tmp_root / "pairs" / name)
    job = make_job(p, tmp_root / "runs", "auto", job_id=f"auto_{name}")
    rep = run(job)
    return d, job, rep


def test_layout_border_period_and_corner_regions(geometric_run):
    d, job, rep = geometric_run["design"], geometric_run["job"], geometric_run["report"]
    lay = read_json(job.stage_dir("structure") / "layout.json")
    gt = d.layout_gt
    assert gt["border"] is not None
    assert lay["border_accepted"] is True
    t_gt = gt["border"]["thickness_knots"]
    scale = lay["scale_512"]
    for e in ("top", "bottom", "left", "right"):
        t_px512 = lay["border"][e]["thickness_px"]
        assert abs(t_px512 * scale / 4 - t_gt) <= max(2, 0.3 * t_gt)
    regions = np.load(job.stage_dir("structure") / "regions.npy")
    assert regions.shape == (d.H, d.W) and (regions == 4).any() and (regions == 1).any()
    assert regions[d.H // 2, d.W // 2] in (0, 2)
    # period: top/bottom strips share the period of the generator
    pk = gt["border"]["period_knots"]
    per = [lay["border_period"][e]["period_px"] for e in ("top", "bottom") if lay["border_period"][e]["period_px"]]
    assert per and all(abs(p * scale / 4 - pk) <= max(2, 0.25 * pk) for p in per)


def test_ground_repeat_vectors(geometric_run):
    d, job = geometric_run["design"], geometric_run["job"]
    lay = read_json(job.stage_dir("structure") / "layout.json")
    v1 = d.layout_gt["repeat"]["v1"]; v2 = d.layout_gt["repeat"]["v2"]
    rep = lay["ground_repeat"]
    assert rep["v1"] is not None
    gt_lengths = sorted([np.hypot(*v1), np.hypot(*v2)])
    found = [np.hypot(*rep["v1"])] + ([np.hypot(*rep["v2"])] if rep["v2"] else [])
    assert any(abs(f - g) <= 2.0 or abs(f - 2 * g) <= 2.0 for f in found for g in gt_lengths)


def test_coarse_labels_shape_and_region_consistency(geometric_run):
    job = geometric_run["job"]
    d = geometric_run["design"]
    labels = np.load(job.stage_dir("structure") / "coarse_labels.npy")
    regions = np.load(job.stage_dir("structure") / "regions.npy")
    assert labels.shape == (d.H, d.W) == regions.shape
    # whole components share one region
    from knotvision.metrics import connected_components
    for c in connected_components(labels)[:50]:
        vals = np.unique(regions[c.full_mask(labels.shape)])
        assert len(vals) == 1


def test_panel_design_has_no_border(tmp_root):
    d = design_of("minimal", seed=51, border=False, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=51)
    p = write_pair(d, r, tmp_root / "pairs" / "panel")
    job = make_job(p, tmp_root / "runs", "minimal", job_id="panel")
    run(job)
    lay = read_json(job.stage_dir("structure") / "layout.json")
    assert lay["border_accepted"] is False and lay["frame_hint"] == "panel"


def test_gradient_mask_iou(gradient_run):
    d, job = gradient_run["design"], gradient_run["job"]
    gm = np.load(job.stage_dir("structure") / "gradient_mask.npy")
    gt = d.gradient_mask
    inter = (gm & gt).sum(); union = (gm | gt).sum()
    assert inter / max(1, union) >= 0.7
    assert gradient_run["report"]["class_features"]["gradient_frac"] > 0.5


def test_class_detection_pure_templates(tmp_root):
    cases = [("geometric", 61, dict(border=True, repeat=True, gradient=False, noise=False, thin_lines=False, symmetry=False)),
             ("gradient", 62, dict(border=False, repeat=False, noise=False, thin_lines=False, symmetry=False)),
             ("distressed", 63, dict(border=False, repeat=False, gradient=False, thin_lines=False, symmetry=False)),
             ("minimal", 64, dict(border=False, gradient=False, noise=False, thin_lines=False, symmetry=False))]
    hits = 0
    for tpl, seed, feat in cases:
        d, job, rep = _run_auto(tmp_root, f"pure_{tpl}", tpl, seed, **feat)
        assert rep["class_source"] == "auto"
        assert rep["class_margin"] is not None
        hits += rep["class_auto"] == tpl
    assert hits >= 3


def test_flat_gradient_is_gradient_not_pictorial(tmp_root):
    d, job, rep = _run_auto(tmp_root, "flatgrad", "gradient", 71, border=False, repeat=False, noise=False, thin_lines=False, symmetry=False)
    assert rep["class_auto"] == "gradient"


def test_mixed_gradient_noise_flags_class_mixed(tmp_root):
    d = design_of("distressed", seed=81, border=False, repeat=False, gradient=True, thin_lines=False, symmetry=False)
    if d.meta["gradient_frac"] < 0.25:
        pytest.skip("generator did not produce a large enough gradient region for this seed")
    r = photo_of(d, seed=81)
    p = write_pair(d, r, tmp_root / "pairs" / "mixed")
    job = make_job(p, tmp_root / "runs", "auto", job_id="auto_mixed")
    rep = run(job)
    assert rep["class_mixed"] is True and "class_mixed" in rep["qc"]["hard_gate_failures"]
    assert rep["class_auto"] == "distressed"


def test_pile_blur_blocks_distressed(tmp_root):
    d = design_of("distressed", seed=91, border=False, repeat=False, gradient=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=91, blur_knots=2.6, px_per_knot=3.5)
    p = write_pair(d, r, tmp_root / "pairs" / "blur_distressed")
    job = make_job(p, tmp_root / "runs", "auto", job_id="auto_blur_dist", override={"allow_low_res": True})
    rep = run(job)
    assert "pile_blur" in rep["warnings"] and rep["class_auto"] != "distressed"


def test_camera_noise_goes_to_1knot_not_noise_frac(tmp_root):
    d = design_of("geometric", seed=101, border=True, repeat=True, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=101, noise_sigma=9.0, jpeg_q=75)
    p = write_pair(d, r, tmp_root / "pairs" / "camnoise")
    job = make_job(p, tmp_root / "runs", "geometric", job_id="camnoise")
    rep = run(job)
    f = rep["class_features"]
    assert f["noise_frac"] < 0.08
    assert rep["class_auto"] != "distressed"


def test_class_conflict_when_designer_differs(geometric_run):
    rep = geometric_run["report"]
    assert rep["class_source"] == "designer"
    if rep["class_auto"] != "geometric" and rep["class_margin"] >= 0.2:
        assert rep["class_conflict"] is True
    else:
        assert rep["class_conflict"] is False


def test_thin_line_frac_and_edit_routing(tmp_root, mock_agent):
    d = design_of("geometric", seed=111, border=True, repeat=False, gradient=False, noise=False, thin_lines=True, symmetry=False)
    assert d.meta["thin_line_frac"] > 0.005
    r = photo_of(d, seed=111, blur_knots=0.4, px_per_knot=3.5)
    p = write_pair(d, r, tmp_root / "pairs" / "thin")
    agent = mock_agent("t_mock_thin")
    job = make_job(p, tmp_root / "runs", "geometric", job_id="thin_edit", override={"design_mode": "edit", "edit_agent": agent})
    rep = run(job)
    assert abs(rep["class_features"]["thin_line_frac"] - d.meta["thin_line_frac"]) < 0.02
    if rep["class_features"]["thin_line_frac"] >= 0.01:
        assert "edit_thin_lines" in rep["warnings"] and rep["design_mode"] == "quantize"
        assert rep["design_mode_requested"] == "edit"


def test_segment_fallback_when_checkpoint_missing(geometric_pair, tmp_root):
    assert not (MODELS_DIR / "knotvision_segment.pt").exists()
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="segfb", override={"design_mode": "segment"})
    rep = run(job)
    assert "segment_fallback" in rep["warnings"] and rep["design_mode"] == "quantize"
    assert rep["design_mode_requested"] == "segment" and rep["design_mode_class_default"] == "quantize"


def test_classify_margin_rules():
    P = {"edge_density_min": 0.02, "gradient_class_frac": 0.25, "noise_class_frac": 0.08}
    base = dict(edge_density=0.05, n_colors_est=6, gradient_frac=0.0, noise_frac=0.0, line_frac=0.2, symmetry="none",
                symmetry_conf=0.2, border_accepted=False, border_period_conf=0.0, ground_repeat_conf=0.0,
                medallion_conf=0.0, pile_blur=False)
    r = S.classify({**base, "gradient_frac": 0.5}, P)
    assert r["class_auto"] == "gradient" and abs(r["class_margin"] - 1.0) < 1e-9
    r = S.classify({**base, "gradient_frac": 0.26}, P)
    assert r["class_auto"] == "gradient" and r["class_margin"] < 0.2
    r = S.classify({**base, "noise_frac": 0.2, "gradient_frac": 0.5}, P)
    assert r["class_auto"] == "distressed"
    r = S.classify({**base, "noise_frac": 0.2, "pile_blur": True}, P)
    assert r["class_auto"] != "distressed"
    r = S.classify({**base, "n_colors_est": 2, "edge_density": 0.005}, P)
    assert r["class_auto"] == "minimal"
    r = S.classify({**base, "symmetry": "hv", "symmetry_conf": 0.9}, P)
    assert r["class_auto"] == "classic" and r["class_margin"] > 0
    r = S.classify({**base, "line_frac": 0.9}, P)
    assert r["class_auto"] == "geometric"
    r = S.classify(base, P)
    assert r["class_auto"] == "classic" and 0 <= r["class_margin"] <= 1
    r = S.classify({**base, "gradient_frac": 0.7, "edge_density": 0.1}, P)
    assert r["class_auto"] == "pictorial"


def test_unmeasurable_internal_geometry_is_a_gate(tmp_root):
    d = design_of("minimal", seed=121, border=False, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=121, blur_knots=1.2, px_per_knot=2.0)
    p = write_pair(d, r, tmp_root / "pairs" / "unmeas")
    job = make_job(p, tmp_root / "runs", "minimal", job_id="unmeas")
    rep = run(job)
    ig = rep["internal_geometry"]
    if not ig["source"]:
        assert ig["status"] == "unmeasurable" and "internal_geometry_unmeasurable" in rep["qc"]["hard_gate_failures"]


def test_knot_grid_visible_on_sharp_high_res(tmp_root):
    d = design_of("minimal", seed=131, border=False, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=131, blur_knots=0.3, px_per_knot=4.0, noise_sigma=1.0, jpeg_q=95)
    p = write_pair(d, r, tmp_root / "pairs" / "knotgrid")
    job = make_job(p, tmp_root / "runs", "minimal", job_id="knotgrid")
    rep = run(job)
    kg = rep["grid"]["knot_grid"]
    if kg["visible"]:
        assert abs(kg["spacing_px"] - 4.0) / 4.0 <= 0.05
        assert "grid_mismatch" not in rep["warnings"]
        assert abs(rep["grid"]["measured_reed_per_m"] - rep["grid"]["reed_per_m"]) / rep["grid"]["reed_per_m"] < 0.03
