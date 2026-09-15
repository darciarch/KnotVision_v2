"""Shared fixtures: small procedural designs, rendered photos, and a pipeline runner that never needs
SAM (corners_override) or any model weight. Everything is seeded and cached per session."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "synth"))

from generate_designs import Design, load_dist, make_design  # noqa: E402
from generate import render_flat, render_photo, save_pair  # noqa: E402
from knotvision.job import Job, JobError  # noqa: E402
from knotvision.run import run_job  # noqa: E402

SMALL_DIST = {
    "source": "test", "n_colors": {"min": 3, "max": 6}, "width_cm": {"min": 25, "max": 30},
    "height_cm": {"min": 32, "max": 38}, "reed_per_m": {"min": 400, "max": 400},
    "wefts_per_m": {"min": 400, "max": 400}, "photo_px_per_knot": {"p10": 2.0, "p50": 2.8, "p90": 3.5},
    "machine_colors": [6, 8, 12],
}
SHARP = {"blur_knots": 0.6, "px_per_knot": 3.0, "illum": 0.12, "shade_a": 0.08, "shade_b": 0.02,
         "fringe": False, "non_planar": False, "occlusion": False, "noise_sigma": 2.0, "jpeg_q": 92, "tilt": 0.05}


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked slow (real model inference on the CPU, minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs GPU + weights")
    config.addinivalue_line("markers", "slow: long-running")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """`gpu` tests are skipped without CUDA, `slow` tests unless --run-slow is given."""
    skip_gpu = pytest.mark.skip(reason="needs a GPU")
    skip_slow = pytest.mark.skip(reason="slow: pass --run-slow")
    has_cuda = _cuda_available()
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def dist():
    return dict(SMALL_DIST)


@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("kv")


def design_of(template: str, seed: int = 0, W: int = 100, H: int = 136, **features) -> Design:
    return make_design(seed, dict(SMALL_DIST), template=template, W=W, H=H, features=features or None)


def photo_of(design: Design, seed: int = 0, **params) -> dict:
    rng = np.random.RandomState(seed)
    return render_photo(design, rng, dict(SMALL_DIST), {**SHARP, **params})


def write_pair(design: Design, r: dict, out: Path) -> Path:
    save_pair(design, r, out)
    return out


def make_job(pair_dir: Path, runs_dir: Path, design_class: str, job_id: str | None = None, **kw) -> Job:
    meta = json.loads((pair_dir / "meta.json").read_text())
    corners = json.loads((pair_dir / "corners_gt.json").read_text())["corners"]
    return Job(job_id=job_id or f"t_{pair_dir.name}", photo_path=pair_dir / "photo.jpg", width_cm=meta["width_cm"],
               height_cm=meta["height_cm"], density=meta["reed_per_m"] * meta["wefts_per_m"],
               machine_colors=kw.pop("machine_colors", meta.get("machine_colors", 8)),
               consent_train=kw.pop("consent_train", False), design_class=design_class,
               corners_override=kw.pop("corners_override", corners), runs_dir=runs_dir,
               batch=kw.pop("batch", design_class == "auto"), **kw)


def run(job: Job, from_stage=None) -> dict:
    return run_job(job, from_stage=from_stage)


@pytest.fixture(scope="session")
def geometric_pair(tmp_root) -> dict:
    d = design_of("geometric", seed=3, border=True, repeat=True, noise=False, gradient=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=3)
    p = write_pair(d, r, tmp_root / "pairs" / "geo")
    return {"design": d, "render": r, "dir": p}


@pytest.fixture(scope="session")
def gradient_pair(tmp_root) -> dict:
    d = design_of("gradient", seed=5, border=False, repeat=False, noise=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=5)
    p = write_pair(d, r, tmp_root / "pairs" / "grad")
    return {"design": d, "render": r, "dir": p}


@pytest.fixture(scope="session")
def distressed_pair(tmp_root) -> dict:
    d = design_of("distressed", seed=7, border=False, repeat=False, gradient=False, thin_lines=False, symmetry=False)
    r = photo_of(d, seed=7)
    p = write_pair(d, r, tmp_root / "pairs" / "dist")
    return {"design": d, "render": r, "dir": p}


@pytest.fixture(scope="session")
def geometric_run(geometric_pair, tmp_root) -> dict:
    """Full modelless chain on the geometric pair (designer class)."""
    job = make_job(geometric_pair["dir"], tmp_root / "runs", "geometric", job_id="geo_run")
    rep = run(job)
    return {"job": job, "report": rep, **geometric_pair}


@pytest.fixture(scope="session")
def gradient_run(gradient_pair, tmp_root) -> dict:
    job = make_job(gradient_pair["dir"], tmp_root / "runs", "gradient", job_id="grad_run")
    rep = run(job)
    return {"job": job, "report": rep, **gradient_pair}


@pytest.fixture(scope="session")
def distressed_run(distressed_pair, tmp_root) -> dict:
    job = make_job(distressed_pair["dir"], tmp_root / "runs", "distressed", job_id="dist_run")
    rep = run(job)
    return {"job": job, "report": rep, **distressed_pair}


@pytest.fixture
def mock_agent(tmp_path, monkeypatch):
    """Write a temporary agent yaml with the mock backend and return a factory: agent(name, **mock_cfg)."""
    import yaml
    from knotvision import job as jobmod
    agents_dir = jobmod.CONFIG_DIR / "agents"
    created = []

    def factory(name: str, mock: dict | None = None, **agent_fields):
        cfg = {"backend": "mock", "edit_n_candidates": 2, "edit_max_rounds": 2,
               "select_weights": {"edge_iou": 1.0, "added_pct": -2.0, "missing_pct": -2.0, "align_px": -1.0,
                                  "palette_deltaE_p95": -0.5, "noise_1knot_pct": -0.5, "line_jaggedness": -0.3},
               "mock": mock or {"mode": "identity"}}
        cfg.update(agent_fields)
        p = agents_dir / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        created.append(p)
        return name

    yield factory
    for p in created:
        if p.exists():
            p.unlink()
