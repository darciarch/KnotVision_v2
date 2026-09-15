import numpy as np

from knotvision import metrics as M


def _blank(H=60, W=80):
    return np.zeros((H, W), np.int32)


def test_is_thin_single_home():
    m = np.zeros((30, 30), bool)
    m[10, 5:25] = True                       # 1-knot wide, 20 long
    assert M.is_thin(m, 6, 1)
    m2 = m.copy(); m2[11, 5:25] = True       # 2 wide
    assert not M.is_thin(m2, 6, 1)
    m3 = np.zeros((30, 30), bool); m3[10, 5:9] = True   # too short
    assert not M.is_thin(m3, 6, 1)
    m4 = np.zeros((30, 30), bool)
    for i in range(20):
        m4[5 + i, 5 + i] = True            # diagonal 1-wide
    assert M.is_thin(m4, 6, 1)


def test_cc_added_missing_and_quantization_residue():
    truth = _blank()
    truth[10:30, 10:30] = 1                  # motif A
    truth[40:50, 50:70] = 2                  # motif B
    out = truth.copy()
    r = M.cc_added_missing(out, truth, motif_min_knots=10)
    assert r["cc_added_motif"] == 0 and r["cc_missing_pct"] == 0 and r["cc_added_area_pct"] == 0
    # quantisation residue: a 3-10 knot speckle does not count as an added motif
    out2 = out.copy(); out2[55:57, 5:8] = 1
    r = M.cc_added_missing(out2, truth, motif_min_knots=10)
    assert r["cc_added_motif"] == 0 and 0 < r["cc_added_area_pct"] < 0.005 and r["added_pct"] == 0
    # a new 15x15 blob is an added motif
    out3 = out.copy(); out3[5:20, 55:70] = 1
    r = M.cc_added_missing(out3, truth, motif_min_knots=10)
    assert r["cc_added_motif"] == 1 and r["added_pct"] > 0
    # removing motif B
    out4 = out.copy(); out4[40:50, 50:70] = 0
    r = M.cc_added_missing(out4, truth, motif_min_knots=10)
    assert abs(r["cc_missing_pct"] - 200 / truth.size) < 1e-9 and r["missing_pct"] > 0


def test_thin_line_recall():
    truth = _blank()
    truth[10, 5:60] = 1
    truth[30, 5:60] = 2
    out = truth.copy()
    assert M.thin_line_recall(out, truth) == 1.0
    out[30, :] = 0
    assert M.thin_line_recall(out, truth) == 0.5
    assert M.thin_line_recall(_blank(), _blank()) is None


def test_line_jaggedness_orders_clean_vs_jagged():
    clean = _blank(); clean[:, 40:] = 1
    jag = clean.copy()
    rng = np.random.RandomState(0)
    for y in range(60):
        jag[y, 40 + rng.randint(-2, 3):] = 1
        jag[y, :40 + rng.randint(-2, 3)] = 0
    assert M.line_jaggedness(jag) > M.line_jaggedness(clean)


def test_repeat_and_symmetry_mismatch():
    a = _blank(64, 64)
    a[::8, :] = 1
    assert M.repeat_mismatch(a, (8, 0), None) == 0.0
    b = a.copy(); b[8, 3] = 0
    assert M.repeat_mismatch(b, (8, 0), None) > 0
    s = _blank(64, 64); s[:, :32] = 1; s[:, 32:] = 1
    assert M.symmetry_mismatch(s, "h") == 0.0
    s[:, :10] = 2
    assert M.symmetry_mismatch(s, "h") > 0
    assert M.symmetry_mismatch(s, "none") is None


def test_banding_banded_greater_than_dithered():
    H, W = 64, 128
    xx = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    rect_L = 30 + 40 * xx
    banded = 30 + 40 * (np.floor(xx * 3) / 3)
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    yy, xk = np.mgrid[0:H, 0:W]
    thr = bayer[yy % 4, xk % 4]
    dith = np.where(np.round(xx * 8) / 8 > thr, 70.0, 30.0).astype(np.float32)
    mask = np.ones((H, W), bool)
    assert M.banding(banded, rect_L, mask) > M.banding(dith, rect_L, mask)


def test_noise_fidelity():
    truth = _blank(64, 64)
    rng = np.random.RandomState(1)
    for _ in range(40):
        y, x = rng.randint(0, 60, 2)
        truth[y:y + 2, x:x + 2] = 1
    nm = np.ones((64, 64), bool)
    same = M.noise_fidelity(truth, truth, nm)
    assert abs(same["noise_fidelity"] - 1.0) < 1e-9
    wiped = M.noise_fidelity(_blank(64, 64), truth, nm)
    assert wiped["noise_fidelity"] < 0.5
    noisy = truth.copy()
    for _ in range(80):
        y, x = rng.randint(0, 63, 2)
        noisy[y, x] = 1
    r = M.noise_fidelity(noisy, truth, nm)
    assert r["noise_cc_ratio"] > 1.0 and r["noise_fidelity"] < 1.0


def test_noise_stats_single_knot_separate():
    a = _blank(40, 40)
    a[5, 5] = 1                       # 1-knot island
    a[10:12, 10:12] = 1               # 4-knot island
    s = M.noise_stats(a, None, 2, 6, 2)
    assert abs(s["noise_1knot_frac"] - 1 / 1600) < 1e-9
    assert abs(s["noise_frac"] - 4 / 1600) < 1e-9


def test_edge_fidelity_identity_and_shift():
    img = np.zeros((128, 128, 3), np.uint8); img[:, 64:] = 200
    assert M.edge_fidelity(img, img) == 1.0
    shifted = np.roll(img, 12, axis=1)
    assert M.edge_fidelity(shifted, img) < 0.5


def test_derive_edit_gates():
    rng = np.random.RandomState(0)
    cc = rng.uniform(0, 0.004, 60)
    ea = cc * 2 + rng.normal(0, 0.0002, 60)
    cm = rng.uniform(0, 0.02, 60)
    em = cm * 1.5 + rng.normal(0, 0.0005, 60)
    g = M.derive_edit_gates(ea, cc, em, cm, corr_min=0.6)
    assert g["edit_gate_source"] == "eval" and g["edit_added_max"] is not None
    g2 = M.derive_edit_gates(rng.uniform(size=60), cc, em, cm)
    assert g2["edit_gate_source"] == "assumed"
