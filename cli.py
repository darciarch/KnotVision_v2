#!/usr/bin/env python
"""KnotVision CLI.

    cli.py run  --photo P --width CM --height CM --density D --colors N --class C --consent yes|no [options]
    cli.py rerun JOB_ID --from STAGE [--class C] [--palette palette.json] [--confirm-palette] [--mode M] ...
    cli.py setup-check [--profile NAME] [--probe-api]
    cli.py batch DIR --class auto [--mode M] [--agent A] [--limit N]     # synthetic / evaluation pairs
    cli.py retention                                                     # apply retention_days
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knotvision import STAGES  # noqa: E402
from knotvision.job import RUNS_DIR, Job, JobError, apply_retention, read_json  # noqa: E402


def _parse_density(s: str):
    if "x" in s:
        a, b = s.lower().split("x")
        return [float(a), float(b)]
    return float(s)


def _parse_corners(s):
    if not s:
        return None
    vals = [float(v) for v in s.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 8:
        raise SystemExit("--corners expects 8 numbers: x0,y0,x1,y1,x2,y2,x3,y3")
    return [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]


def _override_from_args(a) -> dict:
    ov = {}
    for k in ("mode", "agent", "seed", "regularize", "dither", "force_symmetry", "force_border_repeat",
              "force_ground_repeat", "dither_pattern"):
        v = getattr(a, k, None)
        if v is None:
            continue
        key = {"mode": "design_mode", "agent": "edit_agent", "seed": "edit_seed", "dither_pattern": "dither_pattern_override"}.get(k, k)
        ov[key] = v
    if getattr(a, "allow_low_res", False):
        ov["allow_low_res"] = True
    if getattr(a, "allow_partial", False):
        ov["allow_partial"] = True
    return ov


def _add_common(p):
    p.add_argument("--profile", default="wilton_default")
    p.add_argument("--yarns", help="config/yarns/<name>.csv or a path")
    p.add_argument("--mode", choices=["segment", "quantize", "edit"])
    p.add_argument("--agent")
    p.add_argument("--seed", type=int)
    p.add_argument("--corners", help="x0,y0,...,x3,y3 in photo pixels")
    p.add_argument("--palette", help="JSON file: [{rgb: [r,g,b], yarn_code, name}, ...]")
    p.add_argument("--confirm-palette", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--regularize", choices=["auto", "on", "off"])
    p.add_argument("--dither", choices=["auto", "on", "off"])
    p.add_argument("--dither-pattern")
    for k in ("force_symmetry", "force_border_repeat", "force_ground_repeat"):
        p.add_argument("--" + k.replace("_", "-"), dest=k, choices=["auto", "on", "off"])
    p.add_argument("--allow-low-res", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))


def _yarn_path(s):
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p
    q = Path(__file__).resolve().parent / "config" / "yarns" / (s if s.endswith(".csv") else s + ".csv")
    return q if q.exists() else p


def cmd_run(a):
    from knotvision.run import run_job
    consent = a.consent.lower() in ("yes", "true", "1")
    job_id = a.job_id or f"{Path(a.photo).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    job = Job(job_id=job_id, photo_path=Path(a.photo), width_cm=a.width, height_cm=a.height,
              density=_parse_density(a.density), machine_colors=a.colors, consent_train=consent,
              design_class=a.design_class, profile_name=a.profile, yarn_csv=_yarn_path(a.yarns),
              batch=a.batch, override=_override_from_args(a), corners_override=_parse_corners(a.corners),
              palette_override=read_json(a.palette) if a.palette else None, no_gate=a.no_gate,
              confirm_palette=a.confirm_palette, runs_dir=Path(a.runs_dir))
    return _run(job, None)


def cmd_rerun(a):
    from knotvision.run import run_job  # noqa: F401
    run_dir = Path(a.runs_dir) / a.job_id
    if not (run_dir / "job.json").exists():
        raise SystemExit(f"no such job: {run_dir}")
    job = Job.load(run_dir)
    if a.design_class:
        job.design_class = a.design_class
        job.class_source = "designer" if a.design_class != "auto" else "auto"
    if a.palette:
        job.palette_override = read_json(a.palette)
        job.confirm_palette = True
    if a.confirm_palette:
        job.confirm_palette = True
    if a.corners:
        job.corners_override = _parse_corners(a.corners)
    job.no_gate = job.no_gate or a.no_gate
    job.override.update(_override_from_args(a))
    job.params = {}
    return _run(job, a.from_stage)


def _run(job: Job, from_stage):
    from knotvision.run import run_job
    try:
        rep = run_job(job, from_stage=from_stage)
    except JobError as e:
        print(f"STOPPED {job.job_id}: {e.code} {e.message}".rstrip(), file=sys.stderr)
        print(f"  warnings: {job.warnings}", file=sys.stderr)
        return 2
    qc = rep.get("qc") or {}
    print(f"{job.job_id}: {qc.get('status')}  mode={rep.get('design_mode')}  class={rep.get('design_class')}"
          f"  palette_used={rep['colors'].get('palette_used')}  total={rep['timings_s'].get('total')}s")
    if qc.get("hard_gate_failures"):
        print(f"  hard gates: {qc['hard_gate_failures']}")
    if rep.get("warnings"):
        print(f"  warnings: {rep['warnings']}")
    print(f"  output: {job.output_dir}")
    return 0


def cmd_setup_check(a):
    from knotvision.setup_check import check, format_report
    items = check(a.profile, probe_api=a.probe_api)
    print(format_report(items))
    if a.json:
        print(json.dumps(items, indent=2))
    return 0


def cmd_batch(a):
    """Run every pair dir under DIR (photo.jpg + meta.json [+ corners_gt.json]) with --class auto."""
    from knotvision.run import run_job
    pairs = sorted(p for p in Path(a.dir).iterdir() if p.is_dir() and (p / "meta.json").exists())
    if a.limit:
        pairs = pairs[:a.limit]
    rows = []
    for p in pairs:
        meta = read_json(p / "meta.json")
        photo = next((p / n for n in ("photo.jpg", "photo.png") if (p / n).exists()), None)
        if photo is None:
            continue
        corners = read_json(p / "corners_gt.json")["corners"] if (p / "corners_gt.json").exists() else None
        job = Job(job_id=f"batch_{p.name}", photo_path=photo, width_cm=meta["width_cm"], height_cm=meta["height_cm"],
                  density=meta.get("density", meta.get("reed_per_m", 400) * meta.get("wefts_per_m", 400)),
                  machine_colors=meta.get("machine_colors", 8), consent_train=False,
                  design_class=a.design_class, profile_name=a.profile, batch=True,
                  override=_override_from_args(a), corners_override=corners, no_gate=a.no_gate,
                  runs_dir=Path(a.runs_dir))
        try:
            rep = run_job(job)
            rows.append({"id": p.name, "status": (rep.get("qc") or {}).get("status"), "class": rep.get("design_class"),
                         "class_auto": rep.get("class_auto"), "mode": rep.get("design_mode"), "warnings": rep.get("warnings")})
        except JobError as e:
            rows.append({"id": p.name, "status": f"stopped:{e.code}", "warnings": job.warnings})
        print(json.dumps(rows[-1]))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


def cmd_retention(a):
    removed = apply_retention(Path(a.runs_dir))
    for r in removed:
        print("removed", r)
    print(f"{len(removed)} path(s) removed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="knotvision", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--photo", required=True)
    p.add_argument("--width", type=float, required=True, help="pile width, cm (no fringe)")
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--density", required=True, help="per density_definition; reed_x_wefts as 'REEDxWEFTS'")
    p.add_argument("--colors", type=int, required=True, help="machine_colors")
    p.add_argument("--class", dest="design_class", required=True, help="classic|geometric|gradient|distressed|minimal|auto")
    p.add_argument("--consent", required=True, help="consent_train yes|no")
    p.add_argument("--job-id")
    p.add_argument("--batch", action="store_true", help="allow --class auto (synthetic / batch evaluation)")
    _add_common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("rerun")
    p.add_argument("job_id")
    p.add_argument("--from", dest="from_stage", required=True, choices=STAGES)
    p.add_argument("--class", dest="design_class")
    _add_common(p)
    p.set_defaults(fn=cmd_rerun)

    p = sub.add_parser("setup-check")
    p.add_argument("--profile", default="wilton_default")
    p.add_argument("--probe-api", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_setup_check)

    p = sub.add_parser("batch")
    p.add_argument("dir")
    p.add_argument("--class", dest="design_class", default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    _add_common(p)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("retention")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.set_defaults(fn=cmd_retention)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
