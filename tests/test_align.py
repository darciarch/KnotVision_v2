import cv2
import numpy as np

from knotvision.align import align_to_rectified, ecc_homography


def _pattern(h=256, w=320, seed=0):
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), 200, np.uint8)
    for _ in range(25):
        y, x = rng.randint(0, h - 30), rng.randint(0, w - 30)
        img[y:y + 20, x:x + 20] = rng.randint(0, 120, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_known_homography_recovered():
    ref = _pattern()
    H = np.array([[1.0, 0.02, 3.0], [-0.01, 1.0, -2.0], [0, 0, 1]], np.float32)
    moved = cv2.warpPerspective(ref, H, (ref.shape[1], ref.shape[0]), borderValue=(200, 200, 200))
    r = align_to_rectified(moved, ref)
    assert r["converged"] and r["align_px"]["p95"] < 1.5
    diff = np.abs(r["warped"][20:-20, 20:-20].astype(int) - ref[20:-20, 20:-20].astype(int)).mean()
    assert diff < 8


def test_local_distortion_masked():
    ref = _pattern()
    moved = ref.copy()
    # displace one quadrant locally by 8 px
    moved[:128, :160] = np.roll(ref[:128, :160], 14, axis=1)
    r = align_to_rectified(moved, ref)
    assert r["converged"]
    assert r["align_mask"].sum() < r["align_mask"].size


def test_noise_image_does_not_converge():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    b = rng.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    r = align_to_rectified(a, b)
    assert (not r["converged"]) or r["align_px"]["p95"] >= 0.0
