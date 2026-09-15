"""Alignment of an image to `rectified`: global ECC homography (Lab L, 3-level pyramid) + 4x4 block
local affine. Single home; called by design_edit (candidates) and train/align.py (truth renders).

align_to_rectified(img, rectified) -> warped, H_refined, H_local, align_px, align_mask
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .color import rgb_to_lab
from .metrics import edge_fidelity


def _L8(rgb: np.ndarray) -> np.ndarray:
    L = rgb_to_lab(rgb)[..., 0]
    return np.clip(L * 2.55, 0, 255).astype(np.float32)


def ecc_homography(src_L: np.ndarray, ref_L: np.ndarray, levels: int = 3, iters: int = 50,
                   eps: float = 1e-4) -> Optional[np.ndarray]:
    """Coarse-to-fine ECC; returns 3x3 H mapping src -> ref, or None if it never converges."""
    H = np.eye(3, dtype=np.float32)
    pyr_src, pyr_ref = [src_L], [ref_L]
    for _ in range(levels - 1):
        pyr_src.append(cv2.pyrDown(pyr_src[-1]))
        pyr_ref.append(cv2.pyrDown(pyr_ref[-1]))
    converged = False
    for lvl in range(levels - 1, -1, -1):
        s, r = pyr_src[lvl], pyr_ref[lvl]
        scale = 2.0 ** lvl
        Hl = H.copy()
        Hl[0, 2] /= scale
        Hl[1, 2] /= scale
        Hl[2, 0] *= scale
        Hl[2, 1] *= scale
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        try:
            _, Hl = cv2.findTransformECC(r, s, Hl, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            converged = True
        except cv2.error:
            if lvl == 0 and not converged:
                return None
            continue
        H = Hl.copy()
        H[0, 2] *= scale
        H[1, 2] *= scale
        H[2, 0] /= scale
        H[2, 1] /= scale
    return H if converged else None


def local_affine(src_L: np.ndarray, ref_L: np.ndarray, grid: int = 4, iters: int = 30,
                 max_affine_dev: float = 2.0) -> tuple[list, np.ndarray]:
    """Per-block (grid x grid) alignment. The block translation is measured by phase correlation
    (robust on weakly textured blocks); the affine ECC refinement starts from it and is kept only when
    its centre displacement stays within `max_affine_dev` px of the translation (an affine on a block
    with one flat motif can slide while keeping its correlation). Returns (list of 2x3 or None,
    block displacement magnitude (grid, grid))."""
    h, w = ref_L.shape
    bh, bw = h // grid, w // grid
    out, disp = [], np.full((grid, grid), np.inf, dtype=np.float64)
    win = cv2.createHanningWindow((bw, bh), cv2.CV_32F)
    for by in range(grid):
        row = []
        for bx in range(grid):
            ys, xs = slice(by * bh, (by + 1) * bh), slice(bx * bw, (bx + 1) * bw)
            s, r = src_L[ys, xs], ref_L[ys, xs]
            if s.std() < 1.0 or r.std() < 1.0:
                row.append(None)
                disp[by, bx] = 0.0
                continue
            (dx, dy), _resp = cv2.phaseCorrelate(r.astype(np.float32), s.astype(np.float32), win)
            A0 = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            d_pc = float(np.hypot(dx, dy))
            A = A0.copy()
            try:
                _, A = cv2.findTransformECC(r, s, A, cv2.MOTION_AFFINE,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-4), None, 5)
                cx, cy = bw / 2, bh / 2
                px = A[0, 0] * cx + A[0, 1] * cy + A[0, 2]
                py = A[1, 0] * cx + A[1, 1] * cy + A[1, 2]
                d_aff = float(np.hypot(px - cx, py - cy))
                if abs(d_aff - d_pc) > max_affine_dev:
                    A, d_aff = A0, d_pc
            except cv2.error:
                A, d_aff = A0, d_pc
            disp[by, bx] = d_aff
            row.append(A.tolist())
        out.append(row)
    return out, disp


def align_to_rectified(img: np.ndarray, rectified: np.ndarray, grid: int = 4, mask_px: float = 4.0,
                       mask_fid: float = 0.3) -> dict:
    """Warp `img` onto `rectified`. Returns dict(warped, H_refined, H_local, align_px{median,p95},
    align_mask (grid x grid bool), converged)."""
    if img.shape[:2] != rectified.shape[:2]:
        img = cv2.resize(img, (rectified.shape[1], rectified.shape[0]), interpolation=cv2.INTER_CUBIC)
    ref_L, src_L = _L8(rectified), _L8(img)
    H = ecc_homography(src_L, ref_L)
    if H is None:
        return {"warped": img, "H_refined": None, "H_local": None,
                "align_px": {"median": float("inf"), "p95": float("inf")},
                "align_mask": np.zeros((grid, grid), dtype=bool), "converged": False}
    h, w = ref_L.shape
    warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)  # no black strips for the block ECC to latch onto
    Hl, disp = local_affine(_L8(warped), ref_L, grid=grid)
    # fidelity per block
    fid = np.zeros((grid, grid))
    bh, bw = h // grid, w // grid
    for by in range(grid):
        for bx in range(grid):
            ys, xs = slice(by * bh, (by + 1) * bh), slice(bx * bw, (bx + 1) * bw)
            fid[by, bx] = edge_fidelity(warped[ys, xs], rectified[ys, xs])
    finite = disp[np.isfinite(disp)]
    align_px = {"median": float(np.median(finite)) if len(finite) else float("inf"),
                "p95": float(np.percentile(finite, 95)) if len(finite) else float("inf")}
    align_mask = (disp <= mask_px) & (fid >= mask_fid)
    return {"warped": warped, "H_refined": H.tolist(), "H_local": Hl, "align_px": align_px,
            "align_mask": align_mask, "block_fidelity": fid, "converged": True}
