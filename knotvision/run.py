"""Stage orchestration: run / rerun --from <stage>. Each stage is one input -> one output and is
skipped when its input hash (incl. resolved params) is unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import STAGES
from . import intake, rectify, structure, palette, production, regularize, dither, qc, export
from . import design_quant, design_seg, design_edit
from .job import Job, JobError, dump_json


def run_design(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("design")
    rect_png = job.stage_dir("rectify") / "rectified.png"
    pd, st = job.stage_dir("palette"), job.stage_dir("structure")
    mode = job.state.get("design_mode", job.params.get("design_mode", "quantize"))
    inputs = [rect_png, pd / "palette.json", st / "gradient_mask.npy"]
    if mode == "edit":
        inputs.append(pd / "palette_card.png")
    key = job.cache_key("design", inputs, extra={"mode": mode}, exclude_prefix=None if mode == "edit" else "edit_")
    if job.cached(sd, key) and (sd / "design_rgb.png").exists():
        job.timings["design"] = 0.0
        return job.state
    if mode == "segment":
        try:
            stats = design_seg.run(job, rect_png, sd)
        except JobError as e:
            job.warn("segment_fallback")
            job.state["design_mode"] = "quantize"
            job.params["design_mode"] = "quantize"
            stats = design_quant.run(job, rect_png, sd)
            stats["fallback_reason"] = e.message
    elif mode == "edit":
        stats = design_edit.run(job, rect_png, sd)
    else:
        stats = design_quant.run(job, rect_png, sd)
    if mode != "edit":
        job.state["deterministic"] = True
    job.state["design_stats"] = {k: v for k, v in stats.items() if k != "edit"}
    dump_json(sd / "design_stats.json", stats)
    job.state.setdefault("stages_ran", []).append("design")
    job.timings["design"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state


STAGE_FN = {
    "intake": intake.run, "rectify": rectify.run, "structure": structure.run, "palette": palette.run,
    "design": run_design, "production": production.run, "regularize": regularize.run, "dither": dither.run,
    "qc": qc.run, "export": export.run,
}


def run_job(job: Job, from_stage: Optional[str] = None) -> dict:
    """Run all stages (or from `from_stage` on). Returns report dict. Raises JobError on hard stops."""
    start = 0 if from_stage is None else STAGES.index(from_stage)
    if from_stage:
        job.invalidate_from(from_stage)
    # intake always re-validates (cheap) so that params are resolved for a loaded job
    if start > 0 and not job.params:
        job.load_profile()
        job.resolve()
    report = None
    for s in STAGES[start:]:
        if job.state.get("stopped"):
            job.state.pop("stopped", None)
        out = STAGE_FN[s](job)
        if s == "export":
            report = out
    job.save()
    return report
