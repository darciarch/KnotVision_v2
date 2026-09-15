#!/usr/bin/env python
"""Dither pattern from the archive: find dithered regions in archive designs (4x4 blocks mixing exactly
two indices, forming areas >= 64 knots) and recover the designer's ordered matrix from the position
ranks at which the second index appears as the mix ratio grows. Writes config/dither/archive.yaml
(the designer confirms it, then the profile's dither_pattern is set to "archive").

Also exposes dither_regions(indexed) — the rule train/align.py uses for the truth gradient_mask.

Usage: python tools/dither_stats.py [--archive DIR] [--out config/dither/archive.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive import ROOT, iter_designs, merge_same_rgb  # noqa: E402

from knotvision.job import CONFIG_DIR  # noqa: E402

BLOCK = 4


def block_pairs(indexed: np.ndarray, block: int = BLOCK):
    """Per 4x4 block: (n_distinct, a, b, ratio_b) with a < b; ratio_b = share of the second index."""
    H, W = indexed.shape
    hb, wb = H // block, W // block
    n = np.zeros((hb, wb), np.int32)
    a = np.full((hb, wb), -1, np.int32)
    b = np.full((hb, wb), -1, np.int32)
    t = np.zeros((hb, wb), np.float32)
    for y in range(hb):
        for x in range(wb):
            blk = indexed[y * block:(y + 1) * block, x * block:(x + 1) * block]
            vals, counts = np.unique(blk, return_counts=True)
            n[y, x] = len(vals)
            if len(vals) == 2:
                a[y, x], b[y, x] = vals
                t[y, x] = counts[1] / counts.sum()
    return n, a, b, t


def dither_regions(indexed: np.ndarray, block: int = BLOCK, min_area_knots: int = 64) -> np.ndarray:
    """Knot mask of dithered areas: 4x4 blocks with exactly two indices, both indices interleaved
    (neither forms one solid half), adjacent such blocks sharing the same pair, region >= 64 knots."""
    H, W = indexed.shape
    n, a, b, t = block_pairs(indexed, block)
    hb, wb = n.shape
    cand = np.zeros((hb, wb), bool)
    for y in range(hb):
        for x in range(wb):
            if n[y, x] != 2 or not (0.15 <= t[y, x] <= 0.85):
                continue
            blk = indexed[y * block:(y + 1) * block, x * block:(x + 1) * block]
            m = blk == b[y, x]
            # interleaved: the minority index forms >= 2 pieces or touches >= 3 rows/cols
            lab, k = ndi.label(m)
            rows = m.any(1).sum(); cols = m.any(0).sum()
            cand[y, x] = k >= 2 or (rows >= 3 and cols >= 3)
    # group by identical pair
    key = np.where(cand, a * 10000 + b, -1)
    mask = np.zeros((hb, wb), bool)
    for v in np.unique(key[cand]):
        m = key == v
        lab, k = ndi.label(m)
        sizes = ndi.sum(m, lab, index=np.arange(1, k + 1)) * block * block
        for i, s in enumerate(sizes, start=1):
            if s >= min_area_knots:
                mask |= lab == i
    out = np.zeros((H, W), bool)
    out[:hb * block, :wb * block] = np.kron(mask, np.ones((block, block), bool))
    return out


def recover_matrix(indexed: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, int]:
    """Ordered matrix from dithered blocks: position rank = mean over blocks of the ratio at which the
    position first shows the second index (a threshold matrix). Returns (matrix, n_blocks)."""
    n, a, b, t = block_pairs(indexed, block)
    region = dither_regions(indexed, block)
    hb, wb = n.shape
    acc = np.zeros((block, block)); cnt = np.zeros((block, block))
    used = 0
    for y in range(hb):
        for x in range(wb):
            if not region[y * block, x * block] or n[y, x] != 2:
                continue
            blk = indexed[y * block:(y + 1) * block, x * block:(x + 1) * block]
            is_b = blk == b[y, x]
            # positions holding b at ratio t have threshold <= t; positions holding a have threshold > t
            acc += np.where(is_b, t[y, x], 1.0 + t[y, x])
            cnt += 1
            used += 1
    if used == 0:
        return None, 0
    score = acc / np.maximum(cnt, 1)
    ranks = np.argsort(np.argsort(score.ravel())).reshape(block, block)
    return ranks.astype(int), used


def compute(archive_dir: Path | None) -> dict:
    mats, weights, levels = [], [], []
    for _id, idx, rgb in iter_designs(archive_dir):
        idx, _ = merge_same_rgb(idx, rgb)
        m, n = recover_matrix(idx)
        if m is not None:
            mats.append(m); weights.append(n)
            _, _, _, t = block_pairs(idx)
            reg = dither_regions(idx)
            tt = t[reg[::BLOCK, ::BLOCK]] if reg.any() else np.array([])
            if len(tt):
                levels.append(len(np.unique(np.round(tt * 16))))
    if not mats:
        return {"matrix": None, "n_blocks": 0}
    # weighted mean rank per position -> final ranks
    W = np.array(weights, float)
    mean_rank = sum(m * w for m, w in zip(mats, W)) / W.sum()
    ranks = np.argsort(np.argsort(mean_rank.ravel())).reshape(BLOCK, BLOCK)
    lv = int(np.median(levels)) if levels else 4
    lv = max(2, min(16, lv))
    return {"matrix": ranks.tolist(), "n_blocks": int(W.sum()), "levels": lv, "n_designs": len(mats)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive")
    ap.add_argument("--out", default=str(CONFIG_DIR / "dither" / "archive.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    r = compute(Path(a.archive) if a.archive else None)
    print(r)
    if r["matrix"] is None:
        print("no dithered regions found in the archive")
        return 1
    if not a.dry_run:
        doc = {"name": "archive", "type": "ordered", "levels": r["levels"], "matrix": r["matrix"],
               "source": {"n_designs": r["n_designs"], "n_blocks": r["n_blocks"], "confirmed_by_designer": False}}
        Path(a.out).write_text("# Recovered by tools/dither_stats.py; set confirmed_by_designer: true after review,\n"
                               "# then set dither_pattern: archive in the loom profile.\n" + yaml.safe_dump(doc, sort_keys=False))
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
