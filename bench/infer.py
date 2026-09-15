#!/usr/bin/env python
"""Timing / cost benchmark: per-stage and per-mode wall time (edit: per candidate, per tile, per round,
total) on three rug sizes -> bench/timings.json. Report and regression tracking only: there is no time
budget and no mode downgrade (edit takes minutes; the field accepts that).

Usage: python bench/infer.py [--modes quantize,segment,edit] [--agent qwen2511_direct] [--sizes 120x180,160x230,200x300]
       [--gpu-label 24GB] [--out bench/timings.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "synth"))

from generate import render_photo, save_pair  # noqa: E402
from generate_designs import load_dist, make_design  # noqa: E402
from knotvision.job import Job, JobError, RUNS_DIR, read_json  # noqa: E402
from knotvision.run import run_job  # noqa: E402


def bench_one(size_cm: tuple, mode: str, agent: str | None, runs_dir: Path, seed: int = 0) -> dict:
    dist = load_dist()
    reed = 400.0
    W, H = int(size_cm[0] / 100 * reed), int(size_cm[1] / 100 * reed)
    d = make_design(seed, dist, template="geometric", W=W, H=H)
    rng = np.random.RandomState(seed)
    r = render_photo(d, rng, dist, {"px_per_knot": 2.5, "blur_knots": 1.0, "fringe": False, "non_planar": False, "occlusion": False})
    pair = runs_dir / "bench_pairs" / f"{size_cm[0]}x{size_cm[1]}"
    save_pair(d, r, pair)
    ov = {"design_mode": mode}
    if agent:
        ov["edit_agent"] = agent
    job = Job(job_id=f"bench_{size_cm[0]}x{size_cm[1]}_{mode}_{agent or ''}".rstrip("_"), photo_path=pair / "photo.jpg",
              width_cm=size_cm[0], height_cm=size_cm[1], density=reed * reed, machine_colors=8, consent_train=False,
              design_class="geometric", batch=True, override=ov, corners_override=r["corners"], runs_dir=runs_dir)
    t0 = time.time()
    try:
        rep = run_job(job)
    except JobError as e:
        return {"size_cm": list(size_cm), "mode": mode, "agent": agent, "stopped": e.code, "total_s": time.time() - t0}
    out = {"size_cm": list(size_cm), "grid": [W, H], "mode": rep["design_mode"], "mode_requested": mode, "agent": agent,
           "stages_s": rep["timings_s"], "total_s": rep["timings_s"]["total"], "warnings": rep["warnings"]}
    e = rep.get("edit") or {}
    if e:
        n_img = max(1, int(e.get("n_candidates", 1)) * max(1, int(e.get("rounds", 1))) * max(1, int(e.get("n_tiles", 1))))
        out["edit"] = {"wall_s": e.get("wall_s"), "rounds": e.get("rounds"), "n_candidates": e.get("n_candidates"),
                       "n_tiles": e.get("n_tiles"), "per_image_s": (e.get("wall_s") or 0) / n_img,
                       "per_candidate_s": (e.get("wall_s") or 0) / max(1, int(e.get("n_candidates", 1)) * max(1, int(e.get("rounds", 1)))),
                       "per_round_s": (e.get("wall_s") or 0) / max(1, int(e.get("rounds", 1))), "cost_usd": e.get("cost_usd")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="quantize")
    ap.add_argument("--agent")
    ap.add_argument("--sizes", default="120x180,160x230,200x300")
    ap.add_argument("--gpu-label", default="cpu")
    ap.add_argument("--runs", default=str(RUNS_DIR / "bench"))
    ap.add_argument("--out", default=str(ROOT / "bench" / "timings.json"))
    a = ap.parse_args(argv)
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]
    results = []
    for mode in a.modes.split(","):
        for sz in sizes:
            res = bench_one(sz, mode, a.agent if mode == "edit" else None, Path(a.runs))
            print(json.dumps(res))
            results.append(res)
    doc = {"date": time.strftime("%Y-%m-%d"), "host": platform.node(), "gpu": a.gpu_label, "python": platform.python_version(),
           "results": results}
    prev = read_json(a.out) if Path(a.out).exists() else None
    if prev:
        doc["previous_date"] = prev.get("date")
    Path(a.out).write_text(json.dumps(doc, indent=2))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
