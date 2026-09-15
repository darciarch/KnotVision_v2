import numpy as np

from knotvision import PX_PER_KNOT
from knotvision.color import rgb_to_lab
from knotvision.job import read_json
from knotvision.production import cleanup, coverage_merge, mode_pool, quick_quantize, segment_quantize


PAL = np.array([[200, 180, 150], [60, 40, 30], [150, 60, 40]], np.uint8)
PAL_LAB = rgb_to_lab(PAL)


def test_three_colour_quantization_exact():
    idx = np.zeros((20, 30), np.int32); idx[5:15, 5:25] = 1; idx[:, 28:] = 2
    rgb = np.repeat(np.repeat(PAL[idx], 4, 0), 4, 1)
    out, de = quick_quantize(rgb, PAL_LAB, 30, 20)
    assert (out == idx).all() and de.max() < 1e-4


def test_pooling_tie_rule():
    # 4x4 block with 8 px of index 0 and 8 px of index 1: centre pixels decide, then lowest index
    idx_px = np.zeros((4, 4), np.int32); idx_px[:, 2:] = 1
    dist = np.ones((4, 4), np.float32)
    dist[1, 2] = 0.1   # index 1 has the closest centre pixel
    assert mode_pool(idx_px, dist, 1, 1, 2)[0, 0] == 1
    assert mode_pool(idx_px, np.ones((4, 4), np.float32), 1, 1, 2)[0, 0] == 0  # full tie -> lowest


def test_cleanup_islands_and_strips():
    a = np.zeros((30, 30), np.int32)
    a[3, 3] = 1                  # 1x1
    a[10, 10:12] = 1             # 1x2
    a[20, 5:15] = 2              # 1x10 strip
    out, changed = cleanup(a, 2)
    assert out[3, 3] == 0 and (out[10, 10:12] == 0).all() and (out[20, 5:15] == 2).all()
    assert changed > 0
    out0, ch0 = cleanup(a, 0)
    assert (out0 == a).all() and ch0 == 0.0
    out1, _ = cleanup(a, 1)
    assert out1[3, 3] == 0 and (out1[10, 10:12] == 1).all()   # min_region_knots 1: only single knots
    gm = np.zeros((30, 30), bool); gm[0:5, 0:5] = True
    out2, _ = cleanup(a, 2, exempt=gm)
    assert out2[3, 3] == 1


def test_segment_two_pass_never_creates_third_entry():
    H, W = 10, 10
    lab = np.zeros((H, W, 3)); lab[:, :5] = PAL_LAB[0]; lab[:, 5:] = PAL_LAB[1]
    # boundary knots get an intermediate colour that is nearest to entry 2
    lab[:, 4:6] = PAL_LAB[2] * 0.6 + (PAL_LAB[0] + PAL_LAB[1]) * 0.2
    bnd = np.zeros((H, W)); bnd[:, 4:6] = 0.9
    out, _ = segment_quantize(lab, bnd, PAL_LAB, 0.5, 2, None)
    assert set(np.unique(out)) <= {0, 1}


def test_coverage_merge_keeps_motif_colour_and_thin_lines():
    a = np.zeros((100, 100), np.int32)
    a[10, 10:60] = 1              # thin line, 0.5 % coverage
    for i in range(6):            # 6 small components of colour 2 (motif colour criterion)
        a[30 + i * 8, 30:32] = 2
    a[80:82, 80:82] = 3           # 4 knots of colour 3: merged
    pal = rgb_to_lab(np.array([[200, 200, 200], [0, 0, 0], [200, 0, 0], [190, 190, 190]], np.uint8))
    entries = [{"index": i, "gradient_endpoint": None} for i in range(4)]
    params = {"line_min_knots": 6, "line_max_width": 1, "motif_min_knots": 10, "motif_color_min_cc": 5}
    out, merges = coverage_merge(a, pal, 1.0, entries, params)
    assert (out[10, 10:60] == 1).all()
    assert (out == 2).sum() == 12
    assert (out == 3).sum() == 0 and merges[0]["from"] == 3 and merges[0]["to"] == 0


def test_production_stage_outputs(geometric_run):
    job = geometric_run["job"]
    pr = job.stage_dir("production")
    stats = read_json(pr / "production_stats.json")
    raw = np.load(pr / "indexed_raw.npy")
    g = job.state["grid"]
    assert raw.shape == (g["H"], g["W"])
    assert stats["palette_used"] <= job.machine_colors
    assert stats["noise_1knot_pct"] is not None and stats["cleanup_changed_pct"] is not None
    guard = read_json(job.stages_dir / "guard.json")
    assert {e["stage"] for e in guard} >= {"production_cleanup", "production_merge"}
    assert (raw >= 0).all() and raw.max() < len(read_json(job.stage_dir("palette") / "palette.json")["entries"])


def test_distressed_nulls_noise_stats(distressed_run):
    stats = read_json(distressed_run["job"].stage_dir("production") / "production_stats.json")
    assert stats["noise_knots_pct"] is None and stats["cleanup_changed_pct"] is None
