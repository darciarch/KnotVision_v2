import numpy as np

from knotvision import regularize as R
from knotvision.job import read_json
from knotvision.metrics import line_jaggedness, repeat_mismatch


PARAMS = {"reg_tol_knots": 0.7, "reg_min_line_knots": 4, "reg_free_runs": True, "line_min_knots": 6,
          "line_max_width": 1, "min_region_knots": 2, "motif_min_knots": 10}


def _diag_jagged(H=60, W=60, seed=0):
    rng = np.random.RandomState(seed)
    a = np.zeros((H, W), np.int32)
    for y in range(H):
        a[y, : y + rng.randint(-1, 2)] = 1
    return a


def test_jagged_diagonal_becomes_clean_stairs():
    a = _diag_jagged()
    prot = np.zeros(a.shape, bool)
    out, seg = R.smooth_boundaries(a, prot, PARAMS)
    assert line_jaggedness(out) < line_jaggedness(a)
    assert abs(seg["line"] + seg["arc"] + seg["free"] - 1.0) < 1e-6


def test_noisy_straight_line_becomes_straight():
    rng = np.random.RandomState(1)
    a = np.zeros((60, 80), np.int32)
    for y in range(60):
        a[y, 40 + rng.randint(-1, 2):] = 1
    out, _ = R.smooth_boundaries(a, np.zeros(a.shape, bool), PARAMS)
    cols = [int(np.argmax(out[y] == 1)) for y in range(4, 56)]
    assert max(cols) - min(cols) <= 1


def test_arc_fit_residual():
    yy, xx = np.mgrid[0:80, 0:80]
    circ = ((yy - 40) ** 2 + (xx - 40) ** 2 <= 25 ** 2).astype(np.int32)
    out, seg = R.smooth_boundaries(circ, np.zeros(circ.shape, bool), PARAMS)
    diff = (out != circ).mean()
    assert diff < 0.02 and seg["arc"] > 0.3


def test_free_runs_preserve_endpoints():
    c = np.array([[0, 0], [1, 2], [2, 3], [3, 3], [4, 6], [5, 6], [6, 8], [7, 9], [8, 11], [9, 12], [10, 14]], float)
    sm = R._free_run(c, 2.0)
    assert np.allclose(sm[0], c[0]) and np.allclose(sm[-1], c[-1])
    assert np.hypot(*(sm - c).T).max() <= 2.0
    assert (R._free_run(c, 0.0) == c).all()      # over tolerance -> original kept


def test_repeat_single_knot_error_removed_and_corner_untouched(tmp_path):
    a = np.zeros((64, 64), np.int32)
    a[::8, :] = 1
    region = np.ones(a.shape, bool)
    b = a.copy(); b[8, 20] = 0
    out = R.enforce_repeat(b, (8, 0), None, region)
    assert repeat_mismatch(out, (8, 0), None) == 0.0 and out[8, 20] == 1


def test_symmetry_vote_lowest_index_on_tie():
    a = np.zeros((10, 10), np.int32); a[:, 5:] = 1   # perfectly asymmetric: h mirror gives 2-way ties
    out = R.enforce_symmetry(a, "h", np.ones(a.shape, bool))
    assert (out == 0).all()


def test_border_period_enforced():
    a = np.zeros((40, 80), np.int32)
    regions = np.zeros((40, 80), np.uint8); regions[:6, :] = 1; regions[-6:, :] = 1
    for x in range(4, 76, 12):
        a[2:4, x:x + 2] = 2; a[36:38, x:x + 2] = 2
    b = a.copy(); b[2, 16] = 0
    out = R.enforce_border_period(b, regions, {"h": 12, "v": None})
    assert out[2, 16] == 2


def test_stage_off_still_reports_stats(geometric_pair, tmp_root):
    from tests.conftest import make_job, run
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_regoff", override={"regularize": "off"})
    rep = run(job)
    st = rep["regularize"]
    assert st["applied"] is False and st["changed_pct"] == 0.0 and st["line_jaggedness_before"] is not None
    raw = np.load(job.stage_dir("production") / "indexed_raw.npy")
    assert (np.load(job.stage_dir("regularize") / "indexed.npy") == raw).all()


def test_stage_on_geometric(geometric_run):
    rep = geometric_run["report"]["regularize"]
    assert rep["applied"] is True
    assert rep["line_jaggedness_after"] <= rep["line_jaggedness_before"] + 0.05
    sc = rep["segment_classes"]
    assert abs(sc["line"] + sc["arc"] + sc["free"] - 1) < 1e-6 and sc["free_len_share"] == sc["free"]
    job = geometric_run["job"]
    regions = np.load(job.stage_dir("structure") / "regions.npy")
    raw = np.load(job.stage_dir("production") / "indexed_raw.npy")
    out = np.load(job.stage_dir("regularize") / "indexed.npy")
    corner = regions == 4
    if corner.any():
        assert (out[corner] == raw[corner]).all()


def test_truth_design_changes_little(geometric_pair):
    d = geometric_pair["design"]
    out, _ = R.smooth_boundaries(d.indexed, np.zeros(d.indexed.shape, bool), PARAMS)
    assert (out != d.indexed).mean() < 0.03
