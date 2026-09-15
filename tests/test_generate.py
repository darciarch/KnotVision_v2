import json

import numpy as np

from generate_designs import TEMPLATES, make_design
from generate import render_flat, render_photo
from knotvision.metrics import connected_components, is_thin
from tests.conftest import SMALL_DIST, design_of, photo_of


def test_pure_and_mixed_consistent_with_meta():
    for tpl in TEMPLATES:
        for seed in range(3):
            d = make_design(seed, dict(SMALL_DIST), template=tpl, W=96, H=128)
            m = d.meta
            assert d.indexed.shape == (128, 96) and d.palette_rgb.shape[0] == m["n_colors"]
            assert set(np.unique(d.indexed)) <= set(range(m["n_colors"]))
            assert abs(m["gradient_frac"] - d.gradient_mask.mean()) < 1e-9
            assert abs(m["noise_frac"] - d.noise_mask.mean()) < 1e-9
            assert m["mixed"] == (m["gradient_frac"] >= 0.25 and m["noise_frac"] >= 0.08)
            if m["pure"]:
                assert m["template"] == tpl
            if tpl == "distressed":
                assert 0.05 <= m["noise_frac"] <= 0.3
            if tpl == "gradient":
                assert m["gradient_frac"] >= 0.25


def test_thin_line_frac_matches_meta():
    d = design_of("geometric", seed=200, border=False, repeat=False, gradient=False, noise=False, thin_lines=True, symmetry=False)
    measured = 0
    for c in connected_components(d.indexed):
        if is_thin(c.full_mask(d.indexed.shape), 6, 1):
            measured += c.area
    measured /= d.indexed.size
    assert abs(measured - d.meta["thin_line_frac"]) <= 0.005


def test_dist_sampling_ranges():
    for seed in range(5):
        d = make_design(seed, dict(SMALL_DIST))
        m = d.meta
        assert SMALL_DIST["n_colors"]["min"] <= m["n_colors"] <= max(SMALL_DIST["n_colors"]["max"], 3)
        assert m["machine_colors"] in SMALL_DIST["machine_colors"]
        assert 25 <= m["width_cm"] <= 30.5 and 32 <= m["height_cm"] <= 38.5


def test_flat_png_matches_truth_and_photo_corners():
    d = design_of("geometric", seed=201, border=True, repeat=True, gradient=False, noise=False, thin_lines=False, symmetry=False)
    flat = render_flat(d.indexed, d.palette_rgb)
    assert flat.shape == (d.H * 4, d.W * 4, 3)
    back = flat[2::4, 2::4]
    assert (back == d.palette_rgb[d.indexed]).all()
    r = photo_of(d, seed=201)
    c = np.array(r["corners"])
    assert c.shape == (4, 2) and (c[:, 0] >= 0).all() and (c[:, 1] >= 0).all()
    assert r["meta"]["px_per_knot"] == 3.0 and r["photo"].dtype == np.uint8


def test_symmetry_and_layout_gt():
    d = design_of("classic", seed=202)
    assert d.meta["symmetry"] == "hv"
    W = d.W
    assert (d.indexed[:, :W // 2] == d.indexed[:, W - W // 2:][:, ::-1]).all()
    assert d.layout_gt["border"] is not None and d.layout_gt["medallion"] is not None
