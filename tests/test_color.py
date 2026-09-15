import numpy as np

from knotvision.color import delta_e2000, hex_of, lab_to_rgb, nearest_palette_index, rgb_to_lab


def test_srgb_lab_roundtrip_and_reference_values():
    rgb = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], np.uint8)
    lab = rgb_to_lab(rgb)
    assert np.allclose(lab[0], [100, 0, 0], atol=0.02)
    assert np.allclose(lab[1], [0, 0, 0], atol=0.02)
    assert abs(lab[2][0] - 53.24) < 0.1 and abs(lab[2][1] - 80.09) < 0.2 and abs(lab[2][2] - 67.20) < 0.2
    assert abs(lab[5][0] - 53.59) < 0.1
    back = lab_to_rgb(lab)
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_ciede2000_reference_pairs():
    # Sharma et al. (2005) test data, pairs 1, 7, 17, 25
    cases = [
        ([50.0000, 2.6772, -79.7751], [50.0000, 0.0000, -82.7485], 2.0425),
        ([50.0000, 0.0000, 0.0000], [50.0000, -1.0000, 2.0000], 2.3669),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.1736, 0.5854], 1.0000),
        ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
    ]
    for a, b, expected in cases:
        assert abs(float(delta_e2000(np.array(a), np.array(b))) - expected) < 1e-3


def test_nearest_palette_ties_lowest_index():
    pal = rgb_to_lab(np.array([[100, 100, 100], [100, 100, 100], [0, 0, 0]], np.uint8))
    idx, dist = nearest_palette_index(rgb_to_lab(np.array([[100, 100, 100]], np.uint8)), pal)
    assert idx[0] == 0 and dist[0] < 1e-6


def test_hex():
    assert hex_of([200, 183, 152]) == "#C8B798"
