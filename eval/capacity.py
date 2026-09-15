#!/usr/bin/env python
"""Capacity: fix-time is right-skewed and differs by class, so medians and p90 are reported, never
means. jobs/day_typical = designer_hours * 60 / median, jobs/day_bad = designer_hours * 60 / p90;
accepted share per class and mode; cost_usd median and GPU wall time for edit.

Usage: python eval/capacity.py --sheet eval/score_sheet.csv [--runs runs] [--hours 8]
API:   capacity(rows, designer_hours) -> dict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from calibrate_qc import load_sheet  # noqa: E402
from knotvision.job import RUNS_DIR, read_json  # noqa: E402


def _stats(mins: list, hours: float) -> dict:
    if not mins:
        return {"n": 0, "median_min": None, "p90_min": None, "jobs_per_day_typical": None, "jobs_per_day_bad": None}
    m = np.asarray(mins, float)
    med, p90 = float(np.median(m)), float(np.percentile(m, 90))
    return {"n": int(len(m)), "median_min": med, "p90_min": p90,
            "jobs_per_day_typical": (hours * 60 / med) if med > 0 else None,
            "jobs_per_day_bad": (hours * 60 / p90) if p90 > 0 else None}


def capacity(rows: list[dict], designer_hours: float = 8.0) -> dict:
    by_class, by_mode = {}, {}
    for r in rows:
        if r.get("fix_minutes") is None:
            continue
        by_class.setdefault(r["design_class"], []).append(r["fix_minutes"])
        key = r["mode"] + (f"/{r['edit_agent']}" if r.get("edit_agent") else "")
        by_mode.setdefault(key, []).append(r["fix_minutes"])
    out = {"designer_hours": designer_hours,
           "total": _stats([m for v in by_class.values() for m in v], designer_hours),
           "by_class": {k: _stats(v, designer_hours) for k, v in sorted(by_class.items())},
           "by_mode": {k: _stats(v, designer_hours) for k, v in sorted(by_mode.items())}}

    def acc_share(pred):
        sel = [r for r in rows if pred(r) and r.get("qc_status")]
        return (sum(1 for r in sel if str(r["qc_status"]).startswith("accepted")) / len(sel)) if sel else None
    out["accepted_share"] = {"by_class": {c: acc_share(lambda r, c=c: r["design_class"] == c) for c in by_class},
                             "by_mode": {m: acc_share(lambda r, m=m: (r["mode"] + (f"/{r['edit_agent']}" if r.get("edit_agent") else "")) == m) for m in by_mode}}
    edit_rows = [r for r in rows if r.get("mode") == "edit"]
    if edit_rows:
        costs = [r["cost_usd"] for r in edit_rows if r.get("cost_usd") is not None]
        walls = [r["edit_wall_s"] for r in edit_rows if r.get("edit_wall_s") is not None]
        out["edit"] = {"cost_usd_median": float(np.median(costs)) if costs else None,
                       "gpu_wall_s_median": float(np.median(walls)) if walls else None, "n": len(edit_rows)}
    low_res = [r for r in rows if r.get("warnings") and "low_photo_resolution" in r["warnings"]]
    out["low_photo_resolution_rate"] = len(low_res) / len(rows) if rows else None
    return out


def attach_reports(rows: list[dict], runs_dir: Path) -> list[dict]:
    out = []
    for r in rows:
        rp = runs_dir / r["job_id"] / "output" / "report.json"
        r = dict(r)
        if rp.exists():
            rep = read_json(rp)
            r["qc_status"] = (rep.get("qc") or {}).get("status")
            r["warnings"] = rep.get("warnings", [])
            e = rep.get("edit") or {}
            r["cost_usd"] = e.get("cost_usd")
            r["edit_wall_s"] = e.get("wall_s")
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ROOT / "eval" / "score_sheet.csv"))
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = attach_reports(load_sheet(Path(a.sheet)), Path(a.runs))
    res = capacity(rows, a.hours)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
