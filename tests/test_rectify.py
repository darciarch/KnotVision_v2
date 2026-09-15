import json

import cv2
import numpy as np
import pytest

from knotvision import PX_PER_KNOT, rectify
from knotvision.job import JobError, read_json
from tests.conftest import design_of, make_job, photo_of, run, write_pair


def test_output_is_exactly_4W_x_4H_and_no_clahe(geometric_run):
    job = geometric_run["job"]
    g = job.state["grid"]
    rect = cv2.imread(str(job.stage_dir("rectify") / "rectified.png"))
    con = cv2.imread(str(job.stage_dir("rectify") / "rectified_contrast.png"))
    assert rect.shape[:2] == (PX_PER_KNOT * g["H"], PX_PER_KNOT * g["W"])
    assert con.shape == rect.shape
    # CLAHE raises local contrast: the contrast image has a larger L std than the colour-correct one
    assert con.astype(np.float32).std() > rect.astype(np.float32).std() * 0.9
    assert not np.array_equal(con, rect)


def test_corners_override_and_corner_error(geometric_run):
    job = geometric_run["job"]
    c = read_json(job.stage_dir("rectify") / "corners.json")
    assert c["manual"] is True
    gt = np.array(geometric_run["render"]["corners"])
    assert np.abs(np.array(c["corners"]) - gt).max() < 1.0


def test_corner_detection_from_mask_synthetic(geometric_pair):
    """Mask-based corner extraction (what runs after SAM) on the ground-truth quad mask."""
    r = geometric_pair["render"]
    photo = r["photo"]
    mask = rectify.quad_mask(photo.shape, np.array(r["corners"]))
    corners, weak = rectify.corners_from_mask(mask)
    assert not weak
    err = np.abs(corners - rectify.order_corners(np.array(r["corners"]))).max()
    assert err < 8


def test_sam_unavailable_without_corners(geometric_pair, tmp_root):
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_nosam", corners_override=None)
    with pytest.raises(JobError) as e:
        run(job)
    assert e.value.code == "needs_corners"
    assert "sam_unavailable" in job.warnings


def test_fringe_detection_and_strip_back_projection(tmp_root):
    d = design_of("minimal", seed=11, border=True, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=11, fringe=True, px_per_knot=3.0, blur_knots=0.5)
    # rectify on the coarse quad of the pile + fringe: detect_fringe works on the coarse-rectified image
    photo = r["photo"]
    fp = r["meta"]["fringe_px"]
    # build the coarse quad including the fringe strips
    c = np.array(r["corners"], np.float32)
    top = c[0] + (c[0] - c[3]) / np.linalg.norm(c[0] - c[3]) * fp
    cq = np.array([c[0] - (c[3] - c[0]) / np.linalg.norm(c[3] - c[0]) * fp,
                   c[1] - (c[2] - c[1]) / np.linalg.norm(c[2] - c[1]) * fp,
                   c[2] + (c[2] - c[1]) / np.linalg.norm(c[2] - c[1]) * fp,
                   c[3] + (c[3] - c[0]) / np.linalg.norm(c[3] - c[0]) * fp], np.float32)
    cw = int(np.linalg.norm(cq[1] - cq[0])); ch = int(np.linalg.norm(cq[3] - cq[0]))
    Hc = cv2.getPerspectiveTransform(cq, np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32))
    coarse = cv2.warpPerspective(photo, Hc, (cw, ch))
    det, strips, notes = rectify.detect_fringe(coarse, 0.15)
    assert det[0] and det[1] and not det[2] and not det[3]
    assert abs(strips[0] - fp) <= 3 and abs(strips[1] - fp) <= 3


def test_partial_view_stop_and_allow(tmp_root, geometric_pair):
    r = geometric_pair["render"]
    photo = r["photo"]
    # crop the photo so that the rug touches the image border
    c = np.array(r["corners"])
    x0 = int(c[:, 0].min()) + 20
    cropped = photo[:, x0:]
    p = tmp_root / "pairs" / "partial"
    p.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p / "photo.jpg"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    (p / "meta.json").write_text((geometric_pair["dir"] / "meta.json").read_text())
    shifted = (c - [x0, 0]).tolist()
    (p / "corners_gt.json").write_text(json.dumps({"corners": shifted}))
    job = make_job(p, tmp_root / "runs", "geometric", job_id="partial_stop")
    with pytest.raises(JobError) as e:
        run(job)
    assert e.value.code == "partial_view_stop" and job.state["partial_view"] is True
    job2 = make_job(p, tmp_root / "runs", "geometric", job_id="partial_ok", override={"allow_partial": True})
    rep = run(job2)
    assert rep["partial_view"] is True and "partial_view" in rep["qc"]["hard_gate_failures"]


def test_illumination_flattening_reduces_gradient():
    rng = np.random.RandomState(0)
    base = np.full((200, 300, 3), 140, np.uint8)
    for _ in range(12):                      # motifs scattered so that the ramp is identifiable
        y, x = rng.randint(0, 170), rng.randint(0, 270)
        base[y:y + 25, x:x + 25] = [60, 90, 120]
    xx = np.linspace(0.7, 1.3, 300)[None, :, None]
    lit = np.clip(base.astype(np.float32) * xx, 0, 255).astype(np.uint8)
    flat, info = rectify.flatten_illumination(lit, deg=2)
    assert info["flattened"] and info["range_L"] > 5
    from knotvision.color import rgb_to_lab
    L = rgb_to_lab(flat)[..., 0]
    ground = np.all(base == 140, axis=2)
    left = L[:, :60][ground[:, :60]].mean()
    right = L[:, 240:][ground[:, 240:]].mean()
    assert abs(left - right) < 4.0


def test_edge_width_and_effective_resolution_gates(tmp_root):
    d = design_of("geometric", seed=21, border=True, repeat=True, gradient=False, noise=False, thin_lines=False, symmetry=False)
    sharp = photo_of(d, seed=21, blur_knots=0.5, px_per_knot=3.0)
    blurry = photo_of(d, seed=21, blur_knots=3.2, px_per_knot=2.0)
    ps = write_pair(d, sharp, tmp_root / "pairs" / "sharp")
    pb = write_pair(d, blurry, tmp_root / "pairs" / "blurry")
    js = make_job(ps, tmp_root / "runs", "geometric", job_id="sharp")
    rs = run(js)
    assert rs["grid"]["edge_width_knots"] < 1.5
    assert rs["grid"]["photo_px_per_knot_eff"] > 1.5 and "low_photo_resolution" not in rs["warnings"]
    jb = make_job(pb, tmp_root / "runs", "geometric", job_id="blurry")
    with pytest.raises(JobError) as e:
        run(jb)
    assert e.value.code == "low_res_stop" and jb.has_warning("pile_blur")
    jb2 = make_job(pb, tmp_root / "runs", "geometric", job_id="blurry_forced", override={"allow_low_res": True})
    rb = run(jb2)
    assert "forced_low_res" in rb["warnings"] and "forced_low_res" in rb["qc"]["hard_gate_failures"]
    assert rb["grid"]["edge_width_knots"] > 2.0


def test_1800px_long_side_passes_when_effective_resolution_ok(tmp_root):
    d = design_of("geometric", seed=31, W=180, H=240, border=True, repeat=True, gradient=False, noise=False,
                  thin_lines=False, symmetry=False)
    r = photo_of(d, seed=31, blur_knots=0.6, px_per_knot=1800 / 240 / 1.3)
    p = write_pair(d, r, tmp_root / "pairs" / "px1800")
    assert r["meta"]["photo_long_px"] < 2000
    job = make_job(p, tmp_root / "runs", "geometric", job_id="px1800")
    rep = run(job)
    assert rep["grid"]["photo_px_per_knot_eff"] >= 0.8 and rep["photo"]["long_px_below_rec"]


def test_non_planar_tps_synthetic(tmp_root):
    d = design_of("geometric", seed=41, border=True, repeat=True, gradient=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=41, non_planar=True, px_per_knot=3.0, blur_knots=0.5)
    p = write_pair(d, r, tmp_root / "pairs" / "nonplanar")
    job = make_job(p, tmp_root / "runs", "geometric", job_id="nonplanar")
    rep = run(job)
    # the outer quad is given (corners_override) so the deformation shows up in the internal geometry
    ig = rep["internal_geometry"]
    assert ig["source"] and ig["max_dev_knots_before"] is not None
