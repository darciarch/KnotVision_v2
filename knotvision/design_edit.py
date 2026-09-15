"""design / edit: baseline + ref_indexed, working resolution + tiling, agent loop, gain check,
design_rgb.png + the report's `edit` block.

0. Baseline (always, cached separately from edit_* params): quantize path -> baseline_rgb.png,
   quick quantize -> ref_indexed.npy; baseline_edge_iou.
1. edit_px_per_knot = min(target, backend.max_long_px / max(W, H)); tiling when auto and below target.
2.-3. prompt + loop (agents/loop.py).
4. gain check vs baseline_edge_iou - edit_gain_tol -> edit_no_gain.
5. selected candidate -> design_rgb.png, design_lab_smooth.npy.
6. backend failure -> baseline, design_mode -> quantize, edit_fallback.
7. cost_usd, api_egress.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import PX_PER_KNOT
from .agents.base import BackendError, make_backend
from .agents.loop import run_loop, tile_plan
from .color import hex_of, rgb_to_lab
from .design_quant import quantize_rgb
from .job import Job, dump_json, read_json, sha256_file
from .metrics import edge_fidelity, guided
from .production import quick_quantize


def ensure_baseline(job: Job, rect: np.ndarray, palette_lab: np.ndarray, out_dir: Path, rect_png: Path, palette_json: Path) -> dict:
    """Baseline is keyed on rectified + palette only (not on edit_* params / agent)."""
    bdir = out_dir / "baseline"
    bdir.mkdir(parents=True, exist_ok=True)
    key = job.cache_key("design_baseline", [rect_png, palette_json], exclude_prefix="edit_")
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    if job.cached(bdir, key) and (bdir / "ref_indexed.npy").exists():
        base = cv2.cvtColor(cv2.imread(str(bdir / "baseline_rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        info = read_json(bdir / "baseline.json")
        info["cached"] = True
        return {"rgb": base, "ref_indexed": np.load(bdir / "ref_indexed.npy"), **info}
    base = quantize_rgb(rect)
    ref, _ = quick_quantize(base, palette_lab, W, H)
    cv2.imwrite(str(bdir / "baseline_rgb.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    np.save(bdir / "ref_indexed.npy", ref.astype(np.int32))
    info = {"baseline_edge_iou": edge_fidelity(base, rect)}
    dump_json(bdir / "baseline.json", info)
    job.mark_done(bdir, key)
    return {"rgb": base, "ref_indexed": ref, "cached": False, **info}


def working_resolution(W: int, H: int, params: dict, max_long_px: int) -> dict:
    target = float(params.get("edit_px_per_knot_target", 2.0))
    single = min(target, max_long_px / max(W, H))
    tiling = params.get("edit_tiling", "auto")
    if tiling == "auto" and single < target:
        ppk, tiled = target, True
    else:
        ppk, tiled = single, False
    w = int(round(W * ppk / 16.0) * 16)
    h = int(round(H * ppk / 16.0) * 16)
    w, h = max(16, w), max(16, h)
    n_tiles = len(tile_plan(w, h, int(params.get("edit_tile_px", 1024)), 128)) if tiled else 1
    return {"px_per_knot": ppk, "tiled": tiled, "w": w, "h": h, "n_tiles": n_tiles,
            "low_res": ppk < float(params.get("edit_px_per_knot_min", 1.0))}


def run(job: Job, rect_png: Path, out_dir: Path) -> dict:
    t0 = time.time()
    P = job.params
    agent = P.get("agent") or {}
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    pd, st = job.stage_dir("palette"), job.stage_dir("structure")
    rect = cv2.cvtColor(cv2.imread(str(rect_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    pal = read_json(pd / "palette.json")
    palette_lab = np.array([e["lab"] for e in pal["entries"]], dtype=np.float64)
    hex_list = [hex_of(e["rgb"]) for e in pal["entries"]]
    card = cv2.cvtColor(cv2.imread(str(pd / "palette_card.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gmask = np.load(st / "gradient_mask.npy")
    gmask = gmask if gmask.any() else None

    # 0. baseline
    base = ensure_baseline(job, rect, palette_lab, out_dir, rect_png, pd / "palette.json")
    edit_block = {"agent": P.get("edit_agent"), "backend": agent.get("backend"), "seed": int(P.get("edit_seed", 0)),
                  "baseline_edge_iou": base["baseline_edge_iou"], "baseline_cached": base["cached"],
                  "cost_usd": 0.0, "api_egress": False, "gate_source": P.get("edit_gate_source", "assumed")}

    def use_baseline(reason: str, extra: dict):
        cv2.imwrite(str(out_dir / "design_rgb.png"), cv2.cvtColor(base["rgb"], cv2.COLOR_RGB2BGR))
        job.warn(reason)
        edit_block.update(extra)
        edit_block["selected"] = "baseline"
        edit_block["wall_s"] = round(time.time() - t0, 3)
        job.state["edit"] = edit_block
        return {"mode": "edit", "wall_s": edit_block["wall_s"], "edit": edit_block}

    # backend
    try:
        backend = make_backend(agent.get("backend"), agent, job.profile)
    except BackendError as e:
        job.state["design_mode"] = "quantize"
        job.params["design_mode"] = "quantize"
        return use_baseline("edit_fallback", {"error": str(e)})
    edit_block["api_egress"] = bool(backend.api_egress)
    job.state["deterministic"] = bool(backend.supports_seed)
    job.state["versions"]["edit_backend_version"] = backend.version
    lora = agent.get("edit_lora")
    if lora:
        p = Path(job.profile_path).parent.parent.parent / "models" / "edit_lora" / f"{lora}.safetensors"
        job.state["versions"]["edit_lora_sha256"] = sha256_file(p) if p.exists() else None

    # 1. working resolution
    wr = working_resolution(W, H, P, backend.max_long_px)
    edit_block.update({"px_per_knot": wr["px_per_knot"], "tiled": wr["tiled"], "n_tiles": wr["n_tiles"]})
    if wr["low_res"]:
        job.warn("edit_low_res")
    edit_input = cv2.resize(rect, (wr["w"], wr["h"]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "edit_input.png"), cv2.cvtColor(edit_input, cv2.COLOR_RGB2BGR))

    # 2.-3. loop
    res = run_loop(backend, agent, P, edit_input, card, rect, base["ref_indexed"], palette_lab, hex_list,
                   job.design_class, W, H, gmask, out_dir / "edit", wr["tiled"], int(P.get("edit_tile_px", 1024)))
    edit_block["cost_usd"] = float(res.get("cost_usd", 0.0))
    edit_block["rounds"] = res.get("rounds", 0)
    edit_block["n_candidates"] = int(agent.get("edit_n_candidates", 1))
    if res.get("timeout"):
        job.warn("edit_timeout")
    if res.get("fallback"):
        job.state["design_mode"] = "quantize"
        job.params["design_mode"] = "quantize"
        return use_baseline("edit_fallback", {"error": res.get("error")})

    feats = res["features"]
    edit_block.update({"align_px": {"median": feats["align_px_median"], "p95": feats["align_px"]},
                       "added_pct": feats["added_pct"], "missing_pct": feats["missing_pct"],
                       "selected_edge_iou": feats["edge_iou"], "candidate_spread": res["select"]["candidate_spread"],
                       "select_margin": res["select"]["select_margin"], "selected": res["select"]["selected"]})
    # 4. gain check
    if feats["edge_iou"] < base["baseline_edge_iou"] - float(P.get("edit_gain_tol", 0.02)):
        return use_baseline("edit_no_gain", {})
    # 5. selected candidate
    sel = res["selected_rgb"]
    cv2.imwrite(str(out_dir / "design_rgb.png"), cv2.cvtColor(sel, cv2.COLOR_RGB2BGR))
    lab_s = cv2.resize(rgb_to_lab(guided(sel, radius=8, eps=0.01)).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    np.save(out_dir / "design_lab_smooth.npy", lab_s)
    edit_block["wall_s"] = round(time.time() - t0, 3)
    job.state["edit"] = edit_block
    return {"mode": "edit", "wall_s": edit_block["wall_s"], "edit": edit_block}
