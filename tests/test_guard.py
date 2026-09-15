import numpy as np

from knotvision.guard import assert_no_motif_loss, summarize
from knotvision.job import read_json


def _pair():
    before = np.zeros((50, 50), np.int32)
    before[5:20, 5:20] = 1     # 225 knots
    before[30:34, 30:34] = 2   # 16 knots
    before[40:43, 40:42] = 3   # 6 knots
    return before


def test_lost_big_component_is_written_back_only(tmp_path):
    before = _pair()
    after = before.copy()
    after[5:20, 5:20] = 0       # motif 1 lost
    after[30:34, 30:34] = 0     # 16-knot motif lost (11-20 band -> written back with motif_min 10)
    after[40:43, 40:42] = 0     # 6 knots: below the threshold, stays lost
    after[0, 0] = 5             # unrelated change is kept
    gj = tmp_path / "guard.json"
    out = assert_no_motif_loss(before, after, {"motif_min_knots": 10}, "s1", None, gj)
    assert (out[5:20, 5:20] == 1).all() and (out[30:34, 30:34] == 2).all()
    assert (out[40:43, 40:42] == 0).all() and out[0, 0] == 5
    log = read_json(gj)
    assert log[0]["reverted_components"] == 2


def test_noop_inside_gradient_mask_and_distressed(tmp_path):
    before = _pair()
    after = before.copy(); after[5:20, 5:20] = 0
    gm = np.zeros((50, 50), bool); gm[0:25, 0:25] = True
    out = assert_no_motif_loss(before, after, {"motif_min_knots": 10}, "s", gm)
    assert (out[5:20, 5:20] == 0).all()
    out = assert_no_motif_loss(before, after, {"motif_min_knots": 10, "design_class": "distressed"}, "s")
    assert (out[5:20, 5:20] == 0).all()


def test_guard_heavy_and_stage_breakdown(tmp_path):
    before = _pair()
    after = np.zeros_like(before)
    gj = tmp_path / "guard.json"
    assert_no_motif_loss(before, after, {"motif_min_knots": 10}, "a", None, gj)
    assert_no_motif_loss(before, after, {"motif_min_knots": 10}, "b", None, gj)
    s = summarize(gj, 0.02)
    assert s["by_stage"] == {"a": 2, "b": 2} and s["guard_heavy"] is True
    assert abs(s["reverted_area_pct"] - 2 * 241 / 2500) < 1e-9
