"""Motif guard: assert_no_motif_loss(before, after, params) -> after'.

Called at the end of every stage step that changes the index map (production cleanup,
production coverage merge, regularize after each structure, regularize smoothing).
Components of `before` with area > motif_min_knots that have no match in `after`
(same index, IoU >= 0.3) are written back. Only the component is reverted, never the step.
Inside the gradient mask and for design_class == distressed the map is not checked.
Every call is appended to stages/guard.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .job import dump_json, read_json
from .metrics import _match_components, connected_components


def assert_no_motif_loss(before: np.ndarray, after: np.ndarray, params: dict, stage: str,
                         gradient_mask: Optional[np.ndarray] = None,
                         guard_json: Optional[Path] = None, iou_thr: float = 0.3) -> np.ndarray:
    motif_min = int(params.get("motif_min_knots", 10))
    design_class = params.get("design_class")
    out = after.copy()
    reverted, reverted_area = 0, 0
    total = before.size
    if design_class != "distressed":
        valid = None if gradient_mask is None else ~gradient_mask
        comps = [c for c in connected_components(before, valid) if c.area > motif_min]
        ok = _match_components(comps, out, before.shape, iou_thr)
        for c, matched in zip(comps, ok):
            if matched:
                continue
            fm = c.full_mask(before.shape)
            out[fm] = c.index
            reverted += 1
            reverted_area += c.area
    entry = {"stage": stage, "reverted_components": int(reverted),
             "reverted_area_pct": float(reverted_area / max(1, total))}
    if guard_json is not None:
        log = read_json(guard_json) if Path(guard_json).exists() else []
        log.append(entry)
        dump_json(guard_json, log)
    return out


def summarize(guard_json: Path, guard_warn_pct: float = 0.02) -> dict:
    log = read_json(guard_json) if Path(guard_json).exists() else []
    by_stage = {}
    n, area = 0, 0.0
    for e in log:
        by_stage[e["stage"]] = by_stage.get(e["stage"], 0) + int(e["reverted_components"])
        n += int(e["reverted_components"])
        area += float(e["reverted_area_pct"])
    return {"reverted_components": n, "reverted_area_pct": area, "by_stage": by_stage,
            "guard_heavy": area > guard_warn_pct}


def reset(guard_json: Path, keep_stages: tuple = ()):
    """Drop entries of stages that are about to be re-run (keeps entries listed in keep_stages)."""
    if not Path(guard_json).exists():
        return
    log = [e for e in read_json(guard_json) if e["stage"] in keep_stages]
    dump_json(guard_json, log)
