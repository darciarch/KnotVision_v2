"""design / quantize: the model-free baseline (profile default, distressed default, fallback).

rectified.png at 4 px/knot -> guided filter (r=8, eps=0.01) -> mean-shift (sp=24, sr=20) -> design_rgb.png
Also produces the edit-mode baseline (design_edit.py calls quantize_rgb + production.quick_quantize).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .job import Job
from .metrics import guided


def quantize_rgb(rectified: np.ndarray, sp: int = 24, sr: int = 20, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    g = guided(rectified, radius=radius, eps=eps)
    bgr = cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
    ms = cv2.pyrMeanShiftFiltering(bgr, sp=sp, sr=sr, maxLevel=1)
    return cv2.cvtColor(ms, cv2.COLOR_BGR2RGB)


def run(job: Job, rectified_png: Path, out_dir: Path) -> dict:
    t0 = time.time()
    rect = cv2.cvtColor(cv2.imread(str(rectified_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rgb = quantize_rgb(rect)
    cv2.imwrite(str(out_dir / "design_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {"mode": "quantize", "wall_s": round(time.time() - t0, 3)}
