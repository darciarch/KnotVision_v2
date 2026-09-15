"""Shared agent loop for edit mode.

candidates -> align (knotvision.align) -> quick quantize (production.quick_quantize) -> features vs
ref_indexed -> weighted selection -> stop / feedback -> next round. Wall-clock cap edit_max_wall_s.
Cost accounting for API backends. Everything is written under stages/05_design/edit/.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .. import PX_PER_KNOT
from ..align import align_to_rectified
from ..color import rgb_to_lab, delta_e2000
from ..job import dump_json
from ..metrics import edge_fidelity, edit_added_missing, line_jaggedness, noise_stats
from ..production import quick_quantize
from . import prompts as PR
from .base import Backend, BackendError

# scale constants for z = feature / scale (profile-independent, fixed)
FEATURE_SCALE = {"edge_iou": 1.0, "added_pct": 0.01, "missing_pct": 0.02, "align_px": 4.0,
                 "palette_deltaE_p95": 8.0, "noise_1knot_pct": 0.05, "line_jaggedness": 0.5}


def candidate_seed(edit_seed: int, r: int, k: int, t: int = 0) -> int:
    return int(edit_seed + 1000 * r + 10 * k + t)


def tile_plan(w: int, h: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) boxes covering w x h with `tile` size and `overlap`."""
    if w <= tile and h <= tile:
        return [(0, 0, w, h)]
    step = tile - overlap
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    boxes = []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            boxes.append((max(0, x0), max(0, y0), min(w, x0 + tile), min(h, y0 + tile)))
    return boxes


def blend_tiles(tiles: list[np.ndarray], boxes: list, w: int, h: int, overlap: int) -> np.ndarray:
    """Linear blending in the overlap zones."""
    acc = np.zeros((h, w, 3), np.float64)
    wacc = np.zeros((h, w, 1), np.float64)
    for img, (x0, y0, x1, y1) in zip(tiles, boxes):
        th, tw = y1 - y0, x1 - x0
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC) if img.shape[:2] != (th, tw) else img
        wy = np.ones(th)
        wx = np.ones(tw)
        ramp = np.linspace(0, 1, max(2, overlap))
        if y0 > 0:
            wy[:overlap] = ramp[:overlap]
        if y1 < h:
            wy[th - overlap:] = ramp[::-1][:overlap] if overlap <= th else wy[th - overlap:]
        if x0 > 0:
            wx[:overlap] = ramp[:overlap]
        if x1 < w:
            wx[tw - overlap:] = ramp[::-1][:overlap] if overlap <= tw else wx[tw - overlap:]
        wgt = (wy[:, None] * wx[None, :])[..., None]
        acc[y0:y1, x0:x1] += img.astype(np.float64) * wgt
        wacc[y0:y1, x0:x1] += wgt
    return np.clip(acc / np.maximum(wacc, 1e-6), 0, 255).astype(np.uint8)


def candidate_features(cand_aligned: np.ndarray, rectified: np.ndarray, ref_indexed: np.ndarray,
                       palette_lab: np.ndarray, W: int, H: int, params: dict, gmask: Optional[np.ndarray],
                       align_px: dict) -> tuple[dict, np.ndarray]:
    indexed_cand, _ = quick_quantize(cand_aligned, palette_lab, W, H)
    lab = rgb_to_lab(cand_aligned).reshape(-1, 3)
    step = max(1, lab.shape[0] // 50000)
    d = delta_e2000(lab[::step][:, None, :], palette_lab[None, :, :]).min(axis=1)
    added, missing = edit_added_missing(indexed_cand, ref_indexed, int(params.get("motif_min_knots", 10)), gmask)
    ns = noise_stats(indexed_cand, None if gmask is None else ~gmask, int(params.get("noise_cc_min", 2)),
                     int(params.get("noise_cc_max", 6)), int(params.get("min_region_knots", 2)))
    feats = {
        "edge_iou": edge_fidelity(cand_aligned, rectified),
        "palette_deltaE_p95": float(np.percentile(d, 95)),
        "added_pct": float(added), "missing_pct": float(missing),
        "noise_1knot_pct": ns["noise_1knot_frac"],
        "line_jaggedness": line_jaggedness(indexed_cand, None if gmask is None else ~gmask),
        "align_px": float(align_px["p95"]), "align_px_median": float(align_px["median"]),
    }
    return feats, indexed_cand


def select_score(feats: dict, weights: dict) -> float:
    s = 0.0
    for k, w in weights.items():
        v = feats.get(k)
        if v is None or not np.isfinite(v):
            return -np.inf
        s += float(w) * float(v) / FEATURE_SCALE.get(k, 1.0)
    return s


def violations(feats: dict, params: dict) -> list[str]:
    """Stop criterion (spec 3g): the four edit_* thresholds. A candidate that passes all four ends
    the loop."""
    v = []
    if feats["align_px"] > float(params.get("edit_align_max", 6.0)):
        v.append("align_px")
    if feats["added_pct"] > float(params.get("edit_added_max", 0.01)):
        v.append("added_pct")
    if feats["missing_pct"] > float(params.get("edit_missing_max", 0.02)):
        v.append("missing_pct")
    if feats["palette_deltaE_p95"] > float(params.get("edit_palette_deltaE_max", 8.0)):
        v.append("palette_deltaE_p95")
    return v


def feedback_violations(feats: dict, params: dict) -> list[str]:
    """Feedback rules (spec 3h): the stop thresholds plus speckle and jaggedness hints."""
    v = violations(feats, params)
    if feats["noise_1knot_pct"] > 0.05:
        v.append("noise_1knot_pct")
    if feats["line_jaggedness"] > 0.5:
        v.append("line_jaggedness")
    return v


def run_loop(backend: Backend, agent: dict, params: dict, edit_input: np.ndarray, palette_card: np.ndarray,
             rectified: np.ndarray, ref_indexed: np.ndarray, palette_lab: np.ndarray, hex_list: list[str],
             design_class: str, W: int, H: int, gmask: Optional[np.ndarray], out_dir: Path,
             tiled: bool, tile_px: int, overlap: int = 128) -> dict:
    """Returns dict(selected_rgb, features, rounds, n_candidates, select, timeout, cost_usd, n_images,
    fallback (bool), error)."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cand = int(agent.get("edit_n_candidates", 1))
    max_rounds = int(agent.get("edit_max_rounds", 1))
    weights = agent.get("select_weights", {})
    wall_cap = float(params.get("edit_max_wall_s", 900))
    seed0 = int(params.get("edit_seed", 0))
    h_in, w_in = edit_input.shape[:2]
    boxes = tile_plan(w_in, h_in, tile_px, overlap) if tiled else [(0, 0, w_in, h_in)]
    prompt = PR.base_prompt(hex_list, design_class)
    (out_dir / "prompt_0.txt").write_text(prompt)
    photo_img = Image.fromarray(edit_input)
    card_img = Image.fromarray(palette_card)
    current_input = edit_input
    all_cands = []   # (score, r, k, feats, rgb_aligned)
    n_images = 0
    timeout = False
    failures = 0
    best_prev = None
    rounds_done = 0
    for r in range(max_rounds):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(exist_ok=True)
        round_cands = []
        for k in range(n_cand):
            if time.time() - t0 > wall_cap:
                timeout = True
                break
            tiles = []
            try:
                for t, (x0, y0, x1, y1) in enumerate(boxes):
                    src = Image.fromarray(current_input[y0:y1, x0:x1])
                    imgs = [src, card_img] if r == 0 else [src, card_img, Image.fromarray(edit_input[y0:y1, x0:x1])]
                    seed = candidate_seed(seed0, r, k, t) if backend.supports_seed else 0
                    res = backend.edit(imgs, prompt, PR.NEGATIVE_PROMPT, seed, {**agent, "round": r, "candidate": k, "tile": t})
                    n_images += 1
                    tiles.append(np.array(res.convert("RGB")))
            except BackendError as e:
                failures += 1
                if failures >= 3:
                    return {"fallback": True, "error": str(e), "rounds": rounds_done, "n_images": n_images,
                            "cost_usd": n_images * backend.price_usd, "timeout": timeout, "wall_s": time.time() - t0}
                continue
            cand = blend_tiles(tiles, boxes, w_in, h_in, overlap) if tiled else tiles[0]
            cand = cv2.resize(cand, (W * PX_PER_KNOT, H * PX_PER_KNOT), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(round_dir / f"cand_{k}.png"), cv2.cvtColor(cand, cv2.COLOR_RGB2BGR))
            al = align_to_rectified(cand, rectified)
            aligned = al["warped"]
            cv2.imwrite(str(round_dir / f"cand_{k}_aligned.png"), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            if not al["converged"]:
                feats = {"align_px": float("inf"), "edge_iou": 0.0, "added_pct": 1.0, "missing_pct": 1.0,
                         "palette_deltaE_p95": 100.0, "noise_1knot_pct": 1.0, "line_jaggedness": 1.0, "align_px_median": float("inf")}
            else:
                feats, _ = candidate_features(aligned, rectified, ref_indexed, palette_lab, W, H, params, gmask, al["align_px"])
            score = select_score(feats, weights)
            feats["score"] = score
            dump_json(round_dir / f"cand_{k}_features.json", feats)
            round_cands.append((score, r, k, feats, aligned))
            all_cands.append((score, r, k, feats, aligned))
        rounds_done = r + 1
        if not round_cands:
            if timeout:
                break
            continue
        # best of this round; ties -> lowest (r, k)
        round_cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        best = round_cands[0]
        v = violations(best[3], params)
        if not v or timeout:
            break
        if r < max_rounds - 1:
            fb = PR.feedback(feedback_violations(best[3], params), design_class)
            prompt = PR.round_prompt(prompt, fb)
            (out_dir / f"prompt_{r + 1}.txt").write_text(prompt)
            # next round edits the previous best candidate (in edit_input resolution)
            current_input = cv2.resize(best[4], (w_in, h_in), interpolation=cv2.INTER_AREA)
        if time.time() - t0 > wall_cap:
            timeout = True
            break
    if not all_cands:
        return {"fallback": True, "error": "no candidates", "rounds": rounds_done, "n_images": n_images,
                "cost_usd": n_images * backend.price_usd, "timeout": timeout, "wall_s": time.time() - t0}
    all_cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    best = all_cands[0]
    scores = [c[0] for c in all_cands if np.isfinite(c[0])]
    spread = float(np.std(scores)) if len(scores) > 1 else 0.0
    margin = float(all_cands[0][0] - all_cands[1][0]) if len(all_cands) > 1 and np.isfinite(all_cands[1][0]) else None
    select = {"round": best[1], "candidate": best[2], "selected": f"round_{best[1]}/cand_{best[2]}",
              "scores": [{"round": c[1], "candidate": c[2], "score": c[0]} for c in all_cands],
              "select_margin": margin, "candidate_spread": spread}
    dump_json(out_dir / "select.json", select)
    return {"fallback": False, "selected_rgb": best[4], "features": best[3], "rounds": rounds_done,
            "n_candidates": n_cand, "select": select, "timeout": timeout, "n_images": n_images,
            "cost_usd": n_images * backend.price_usd, "wall_s": time.time() - t0}
