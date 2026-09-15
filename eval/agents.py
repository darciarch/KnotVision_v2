#!/usr/bin/env python
"""Agent race: eval/run.py with design_mode: edit for every config/agents/*.yaml (tiled / untiled as
separate rows), plus segment and quantize baselines in the same table. Output
eval/results/agents_<date>.csv (class x agent rows) and the per-class winner:

    winner(class) = among agents that pass the release criterion in that class, the one with the
    highest mean qc_score; ties -> cheaper (time_s median, then cost_usd median); none -> null
    (the class keeps its segment / quantize default). At most two agents enter QC calibration.

muse_image runs only on test_core_* + 30 synthetic (API cost); open-weight backends on the full set.

Usage: python eval/agents.py --pairs data/synth/pairs [--ids eval/synth_test_ids.txt] [--agents qwen2511_loop,...]
       [--tiling auto,off] [--out eval/results/agents_<date>.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from knotvision import DESIGN_CLASSES  # noqa: E402
from knotvision.job import CONFIG_DIR, RUNS_DIR, load_yaml, read_json  # noqa: E402
from run import METRIC_COLS, release_check, run_pairs, summarize  # noqa: E402

MAX_CALIBRATED_AGENTS = 2
MUSE_MAX_SYNTH = 30


def race_table(rows: list[dict], profile: dict) -> list[dict]:
    """One row per (class, mode/agent[/tiling]) with the summary metrics + release pass."""
    table = []
    configs = sorted({(r.get("mode"), r.get("edit_agent"), r.get("tiling")) for r in rows if not r.get("stopped")},
                     key=lambda c: (str(c[0]), str(c[1]), str(c[2])))
    for mode, agent, tiling in configs:
        sel = [r for r in rows if r.get("mode") == mode and r.get("edit_agent") == agent and r.get("tiling") == tiling]
        for cls in list(DESIGN_CLASSES) + ["total"]:
            cs = sel if cls == "total" else [r for r in sel if r.get("design_class") == cls]
            if not cs:
                continue
            summ = summarize(cs)["total"]
            rel = release_check(cs, profile, mode)
            row = {"design_class": cls, "mode": mode, "edit_agent": agent, "tiling": tiling, "n": len(cs),
                   "release_pass": rel["pass"], "release_failures": "|".join(rel["failures"]),
                   "qc_score": summ.get("qc_score"), "time_s_median": summ.get("time_s_median"),
                   "cost_usd_median": summ.get("cost_usd_median"), "edit_rounds_median": summ.get("edit_rounds_median"),
                   "edit_route_out_rate": rel.get("edit_route_out_rate"), "edit_no_gain_rate": rel.get("edit_no_gain_rate")}
            for m in METRIC_COLS:
                row[m] = summ.get(m)
            table.append(row)
    return table


def pick_winners(table: list[dict]) -> dict:
    """Per class: passing edit agents ranked by qc_score desc, then time asc, then cost asc."""
    winners = {}
    for cls in DESIGN_CLASSES:
        cands = [r for r in table if r["design_class"] == cls and r["mode"] == "edit" and r["release_pass"]]
        if not cands:
            winners[cls] = None
            continue
        cands.sort(key=lambda r: (-(r["qc_score"] if r["qc_score"] is not None else -1.0),
                                  r["time_s_median"] if r["time_s_median"] is not None else np.inf,
                                  r["cost_usd_median"] if r["cost_usd_median"] is not None else np.inf,
                                  str(r["edit_agent"]), str(r["tiling"])))
        w = cands[0]
        winners[cls] = {"agent": w["edit_agent"], "tiling": w["tiling"], "qc_score": w["qc_score"]}
    return winners


def agents_to_calibrate(winners: dict, max_agents: int = MAX_CALIBRATED_AGENTS) -> list[str]:
    """At most two agents (the ones winning the most classes; ties by name) enter QC calibration."""
    count = {}
    for w in winners.values():
        if w:
            count[w["agent"]] = count.get(w["agent"], 0) + 1
    return [a for a, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:max_agents]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--agents", help="comma list; default: all config/agents/*.yaml")
    ap.add_argument("--tiling", default="auto,off")
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--runs", default=str(RUNS_DIR / "agents"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / f"agents_{time.strftime('%Y%m%d')}.csv"))
    ap.add_argument("--baselines", default="quantize,segment")
    a = ap.parse_args(argv)
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    pairs = sorted(p for p in Path(a.pairs).iterdir() if p.is_dir())
    if a.ids:
        ids = set(Path(a.ids).read_text().split())
        pairs = [p for p in pairs if p.name in ids]
    agents = a.agents.split(",") if a.agents else sorted(p.stem for p in (CONFIG_DIR / "agents").glob("*.yaml"))
    rows = []
    for mode in [m for m in a.baselines.split(",") if m]:
        rs = run_pairs(pairs, mode, None, None, None, Path(a.runs), a.profile, tag=f"_{mode}")
        for r in rs:
            r["tiling"] = None
        rows.extend(rs)
    for agent in agents:
        cfg = load_yaml(CONFIG_DIR / "agents" / f"{agent}.yaml")
        sel = pairs
        if cfg.get("backend") == "muse_image":
            core = [p for p in pairs if not p.name.startswith("proc_")]
            synth = [p for p in pairs if p.name.startswith("proc_")][:MUSE_MAX_SYNTH]
            sel = core + synth
        for tiling in a.tiling.split(","):
            # tiling is an edit_* parameter: pass it through the job override
            rs = run_pairs_with_tiling(sel, agent, tiling, Path(a.runs), a.profile)
            rows.extend(rs)
    table = race_table(rows, P)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["design_class", "mode", "edit_agent", "tiling", "n", "release_pass", "release_failures", "qc_score",
            "time_s_median", "cost_usd_median", "edit_rounds_median", "edit_route_out_rate", "edit_no_gain_rate"] + METRIC_COLS
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    winners = pick_winners(table)
    res = {"winners": winners, "calibrate": agents_to_calibrate(winners), "csv": a.out}
    Path(a.out).with_suffix(".winners.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def run_pairs_with_tiling(pairs, agent, tiling, runs_dir, profile):
    from knotvision.job import Job, JobError
    from knotvision.run import run_job
    from run import evaluate_job
    rows = []
    for p in pairs:
        meta = read_json(p / "meta.json")
        photo = next((p / n for n in ("photo.jpg", "photo.png") if (p / n).exists()), None)
        if photo is None:
            continue
        corners = None
        for cf in ("corners_gt.json", "corners.json"):
            if (p / cf).exists():
                corners = read_json(p / cf)["corners"]
                break
        density = meta.get("density") or (meta.get("reed_per_m", 400) * meta.get("wefts_per_m", meta.get("reed_per_m", 400)))
        jid = f"race_{p.name}_{agent}_{tiling}"
        job = Job(job_id=jid, photo_path=photo, width_cm=meta["width_cm"], height_cm=meta["height_cm"], density=density,
                  machine_colors=meta.get("machine_colors", 8), consent_train=False, design_class=meta.get("design_class", "auto"),
                  profile_name=profile, batch=True, override={"design_mode": "edit", "edit_agent": agent, "edit_tiling": tiling},
                  corners_override=corners, runs_dir=runs_dir)
        try:
            rep = run_job(job)
        except JobError as e:
            rows.append({"job_id": jid, "pair": p.name, "design_class": meta.get("design_class"), "mode": "edit",
                         "edit_agent": agent, "tiling": tiling, "stopped": e.code})
            continue
        r = evaluate_job(job, rep, p, meta)
        r["stopped"] = None
        r["edit_agent"] = agent      # keep the agent even when routing moved the job away from edit
        r["tiling"] = tiling
        rows.append(r)
    return rows


if __name__ == "__main__":
    sys.exit(main())
