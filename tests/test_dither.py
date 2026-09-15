import numpy as np

from knotvision.color import rgb_to_lab
from knotvision.dither import apply_dither, load_pattern, ramp_indices


def _ramp_setup(H=32, W=64):
    pal_rgb = np.array([[40, 40, 40], [220, 220, 220], [200, 30, 30]], np.uint8)
    pal_lab = rgb_to_lab(pal_rgb)
    entries = [{"index": i, "rgb": pal_rgb[i].tolist(), "lab": pal_lab[i].tolist(),
                "gradient_endpoint": [0, "lab_lo"] if i == 0 else [0, "lab_hi"] if i == 1 else None} for i in range(3)]
    t = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    lab_smooth = (pal_lab[0][None, None, :] * (1 - t[..., None]) + pal_lab[1][None, None, :] * t[..., None]).astype(np.float32)
    gmask = np.ones((H, W), bool)
    regions = [{"id": 0, "bbox": [0, 0, W - 1, H - 1], "lab_lo": pal_lab[0].tolist(), "lab_hi": pal_lab[1].tolist()}]
    indexed = np.zeros((H, W), np.int32)
    return indexed, lab_smooth, gmask, regions, {"entries": entries}, pal_lab, t


def test_block_mix_ratio_and_determinism():
    indexed, lab_s, gmask, regions, pal, pal_lab, t = _ramp_setup()
    pat = load_pattern(None)
    out1, info = apply_dither(indexed, lab_s, gmask, regions, pal, pat, 4)
    out2, _ = apply_dither(indexed, lab_s, gmask, regions, pal, pat, 4)
    assert (out1 == out2).all() and info["pairs"] == [(0, 1)]
    inner = out1[4:-4, 4:-4]
    tt = t[4:-4, 4:-4]
    for x in range(0, inner.shape[1] - 4, 4):
        blk = inner[:, x:x + 4]
        ratio = (blk == 1).mean()
        assert abs(ratio - np.round(tt[0, x + 2] * 4) / 4) <= 0.26


def test_no_spill_and_edge_snaps_to_endpoint():
    indexed, lab_s, gmask, regions, pal, pal_lab, t = _ramp_setup()
    gmask[:, 40:] = False
    indexed[:, 40:] = 2
    out, _ = apply_dither(indexed, lab_s, gmask, regions, pal, load_pattern(None), 4)
    assert (out[:, 40:] == 2).all()
    assert set(np.unique(out[0, :40])) <= {0, 1}          # top edge row: only endpoints
    assert (out[:, 0] == 0).all()                         # left edge (t ~ 0) -> index 0


def test_ramp_indices_multi_segment():
    pal = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 0, 0]], np.uint8))
    chain = ramp_indices(pal[0], pal[1], pal, (0, 1))
    assert chain == [0, 2, 1]


def test_stage_default_pattern_warning_and_report(gradient_run):
    rep = gradient_run["report"]
    assert rep["dither"]["applied"] is True and rep["dither"]["mask_pct"] > 0
    assert "dither_pattern_default" in rep["warnings"]
    assert "dither_pattern_default" in rep["qc"]["hard_gate_failures"]


def test_stage_skips_without_mask(geometric_run):
    rep = geometric_run["report"]
    assert rep["dither"]["applied"] is False
