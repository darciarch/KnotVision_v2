"""Job state, parameter layering, stage cache, routing rules, setup/versions blocks.

Parameter resolution (later layers override earlier ones):
    loom profile  <-  class file (config/classes/<design_class>.yaml)
                  <-  agent file (config/agents/<edit_agent>.yaml, only edit_* fields, only when design_mode == edit)
                  <-  job override

Class files may override every field except the loom-owned ones in LOOM_ONLY_FIELDS.
The resolved set lives in job.json["resolved_params"]; its hash is part of every stage cache key.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import DESIGN_CLASSES, DESIGN_MODES, STAGES, __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

LOOM_ONLY_FIELDS = ("allowed_machine_colors", "reed_per_m", "density_definition", "retention_days", "allow_api")

# Agent file keys that are allowed to override. Everything else in the agent file is backend config
# that is passed through verbatim under resolved_params["agent"].
AGENT_OVERRIDE_PREFIX = "edit_"

# Params each stage reads. Only these enter the stage's cache key, so a config change reruns exactly
# the stages that consume it (spec: a class-file change reruns from structure on, an agent-file change
# reruns design/edit only). A stage that starts reading a new param must add it here.
STAGE_PARAM_KEYS = {
    "intake": ("allowed_machine_colors", "reed_per_m", "density_definition", "photo_long_px_rec"),
    "rectify": ("fringe_detect", "fringe_max_frac", "allow_partial", "planarity_warn", "planarity_stop",
                "illum_poly_deg", "illum_warn", "photo_px_warn", "photo_px_stop", "pile_blur_knots", "allow_low_res"),
    # design_mode / edit_thin_line_max are NOT here: routing is re-applied from layout.json on a cache hit
    "structure": ("border_frac_min", "border_frac_max", "grad_min", "grad_max", "noise_cc_min", "noise_cc_max",
                  "min_region_knots", "line_min_knots", "line_max_width", "edge_density_min", "gradient_class_frac",
                  "noise_class_frac", "class_margin_min", "planarity_warn", "grid_mismatch_tol"),
    "palette": ("area_weight_exp", "adjacency_penalty", "gain_stop_select", "merge_cost_stop",
                "motif_color_min_cc", "motif_min_knots"),
    "design_baseline": (),
    "production": ("boundary_thr", "boundary_snap_r", "noise_cc_min", "noise_cc_max", "min_region_knots",
                   "min_coverage_pct", "line_min_knots", "line_max_width", "motif_min_knots", "motif_color_min_cc",
                   "palette_model_mismatch_warn", "deltaE_warn", "design_class"),
    "regularize": ("regularize", "reg_tol_knots", "reg_min_line_knots", "reg_free_runs", "enforce_conf",
                   "force_symmetry", "force_border_repeat", "force_ground_repeat", "min_region_knots",
                   "line_min_knots", "line_max_width", "motif_min_knots", "guard_warn_pct", "design_class"),
    "dither": ("dither", "dither_pattern", "dither_pattern_override", "dither_levels"),
}


def edit_param_keys(params: dict) -> tuple:
    """Keys the edit design path reads: every edit_* field, the raw agent yaml and the class (prompt)."""
    return tuple(sorted(k for k in params if k.startswith("edit_"))) + ("agent", "design_class")

# Warnings that stop the job (hard stops) vs. warnings that become hard gates in qc.
STOP_CODES = {
    "needs_corners", "sam_unavailable", "class_unsupported", "grid_input_mismatch",
    "api_not_allowed", "partial_view_stop", "non_planar_stop", "low_res_stop",
}


class JobError(Exception):
    """Raised for validation failures and hard stops. `code` is the machine-readable reason."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


# ----------------------------------------------------------------------------- hashing helpers

def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_yaml(path: Path | str) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f)
    return d or {}


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------- parameter layering

def resolve_params(profile: dict, class_cfg: dict, job_override: dict,
                   agent_cfg: Optional[dict] = None) -> dict:
    """profile <- class <- agent(edit_* only) <- job. Loom-only fields can't be touched by class/agent."""
    params = dict(profile)
    for k, v in class_cfg.items():
        if k in LOOM_ONLY_FIELDS:
            raise JobError("class_overrides_loom_field", f"class file may not set {k}")
        params[k] = v
    # the design_mode after class layer, before job override (reported as design_mode_class_default)
    params["design_mode_class_default"] = params.get("design_mode", "quantize")
    # job override (before agent, so we know whether edit is requested and by which agent)
    for k, v in job_override.items():
        if v is None:
            continue
        params[k] = v
    if params.get("design_mode") == "edit" and agent_cfg:
        # agent sits between class and job: apply agent's edit_* then re-apply job override
        for k, v in agent_cfg.items():
            if k.startswith(AGENT_OVERRIDE_PREFIX):
                params[k] = v
        for k, v in job_override.items():
            if v is None:
                continue
            params[k] = v
        params["agent"] = {k: v for k, v in agent_cfg.items()}
    else:
        params["agent"] = None
    return params


def load_class_config(design_class: str) -> dict:
    if design_class == "auto":
        return {}
    p = CONFIG_DIR / "classes" / f"{design_class}.yaml"
    if not p.exists():
        raise JobError("class_config_missing", str(p))
    return load_yaml(p)


def load_agent_config(name: str) -> dict:
    p = CONFIG_DIR / "agents" / f"{name}.yaml"
    if not p.exists():
        raise JobError("agent_config_missing", str(p))
    return load_yaml(p)


def agent_hash(name: str) -> str:
    """Hash of the agent file + the prompt template module; a prompt change invalidates calibration."""
    p = CONFIG_DIR / "agents" / f"{name}.yaml"
    h = hashlib.sha256(p.read_bytes())
    prompts = Path(__file__).parent / "agents" / "prompts.py"
    if prompts.exists():
        h.update(prompts.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- setup / versions

def setup_block(profile: dict, params: dict, yarn_csv: Optional[Path]) -> dict:
    dither_pattern = params.get("dither_pattern")
    if dither_pattern is None:
        dither_src = "default"
    elif str(dither_pattern).startswith("archive"):
        dither_src = "archive"
    else:
        dither_src = "designer"
    return {
        "reed_per_m": "profile" if profile.get("reed_per_m") else "assumed",
        "density_definition": profile.get("density_definition"),
        "dither_pattern": dither_src,
        "yarn_library": bool(yarn_csv),
        "motif_min_knots": profile.get("motif_min_knots_source", "assumed"),
        "noise_scale": profile.get("noise_scale_source", "assumed"),
        "qc_calib": (MODELS_DIR / "qc_calib.json").exists(),
        "sam": (MODELS_DIR / "sam3.1").exists(),
        "segment_ckpt": (MODELS_DIR / "knotvision_segment.pt").exists(),
        "photo_survey": (DATA_DIR / "photo_survey.csv").exists(),
        "edit_backends": {
            "qwen_edit_2511": (MODELS_DIR / "qwen_image_edit_2511").exists(),
            "qwen_edit_2509": (MODELS_DIR / "qwen_image_edit_2509").exists(),
            "muse_image": bool(os.environ.get("MUSE_API_KEY")) and bool(profile.get("allow_api")),
        },
    }


def versions_block(profile_path: Path, design_class: str, yarn_csv: Optional[Path],
                   edit_agent: Optional[str]) -> dict:
    cls_path = CONFIG_DIR / "classes" / f"{design_class}.yaml"
    seg = MODELS_DIR / "knotvision_segment.pt"
    calib = MODELS_DIR / "qc_calib.json"
    return {
        "knotvision": __version__,
        "git_sha": git_sha(),
        "profile_hash": sha256_file(profile_path),
        "class_config_hash": sha256_file(cls_path) if cls_path.exists() else None,
        "yarn_csv_hash": sha256_file(yarn_csv) if yarn_csv else None,
        "sam_checkpoint": "facebook/sam3.1" if (MODELS_DIR / "sam3.1").exists() else None,
        "segment_ckpt_sha256": sha256_file(seg) if seg.exists() else None,
        "qc_calib_hash": sha256_file(calib) if calib.exists() else None,
        "edit_agent_hash": agent_hash(edit_agent) if edit_agent else None,
        "edit_backend_version": None,
        "edit_lora_sha256": None,
    }


# ----------------------------------------------------------------------------- Job

@dataclass
class Job:
    job_id: str
    photo_path: Path
    width_cm: float
    height_cm: float
    density: Any                      # float or [reed, wefts] for reed_x_wefts
    machine_colors: int
    consent_train: bool
    design_class: str                 # class name or "auto"
    class_source: str = "designer"    # designer | auto
    profile_name: str = "wilton_default"
    yarn_csv: Optional[Path] = None
    batch: bool = False               # --class auto is only accepted with batch/synthetic
    override: dict = field(default_factory=dict)   # job-level parameter overrides (design_mode, edit_agent, ...)
    corners_override: Optional[list] = None
    palette_override: Optional[list] = None
    no_gate: bool = False
    confirm_palette: bool = False
    runs_dir: Path = RUNS_DIR

    # runtime state
    state: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    profile_path: Path = None
    params: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    # ---- paths
    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.job_id

    @property
    def stages_dir(self) -> Path:
        return self.run_dir / "stages"

    @property
    def output_dir(self) -> Path:
        return self.run_dir / "output"

    def stage_dir(self, name: str) -> Path:
        i = STAGES.index(name)
        d = self.stages_dir / f"{i:02d}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def job_json(self) -> Path:
        return self.run_dir / "job.json"

    # ---- state helpers
    @property
    def warnings(self) -> list:
        return self.state.setdefault("warnings", [])

    def warn(self, code: str):
        if code not in self.warnings:
            self.warnings.append(code)
        stage = self.state.get("_stage")
        if stage:
            per = self.state.setdefault("warnings_by_stage", {}).setdefault(stage, [])
            if code not in per:
                per.append(code)

    def begin_stage(self, name: str):
        """Called by the orchestrator before a stage runs: warnings that this stage produced in an
        earlier run are dropped (they are restored from the stage cache if the stage is skipped)."""
        self.state["_stage"] = name
        per = self.state.setdefault("warnings_by_stage", {})
        old = per.get(name, [])
        if old:
            self.state["warnings"] = [w for w in self.warnings if w not in old]
        per[name] = []

    def restore_state(self):
        """Load state/timings persisted in job.json (needed when earlier stages are served from cache)."""
        if self.job_json.exists() and not self.state:
            d = json.loads(self.job_json.read_text())
            self.state = d.get("state", {})
            self.timings = d.get("timings_s", {})
            self.state.pop("stopped", None)

    def has_warning(self, code: str) -> bool:
        return code in self.warnings

    def stop(self, code: str, message: str = ""):
        """Hard stop unless --no-gate; in that case record the warning and continue."""
        if self.no_gate:
            self.warn(code)
            self.warn("no_gate")
            return
        self.state["stopped"] = code
        self.save()
        raise JobError(code, message)

    def p(self, key: str, default=None):
        return self.params.get(key, default)

    # ---- persistence
    def save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": self.job_id,
            "photo_path": str(self.photo_path),
            "width_cm": self.width_cm, "height_cm": self.height_cm, "density": self.density,
            "machine_colors": self.machine_colors, "consent_train": self.consent_train,
            "design_class": self.design_class, "class_source": self.class_source,
            "profile_name": self.profile_name, "yarn_csv": str(self.yarn_csv) if self.yarn_csv else None,
            "batch": self.batch, "override": self.override,
            "corners_override": self.corners_override, "palette_override": self.palette_override,
            "no_gate": self.no_gate, "confirm_palette": self.confirm_palette,
            "resolved_params": self.params,
            "timings_s": self.timings,
            "state": self.state,
        }
        tmp = self.job_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
        tmp.replace(self.job_json)

    @classmethod
    def load(cls, run_dir: Path | str) -> "Job":
        run_dir = Path(run_dir)
        d = json.loads((run_dir / "job.json").read_text())
        job = cls(
            job_id=d["job_id"], photo_path=Path(d["photo_path"]), width_cm=d["width_cm"],
            height_cm=d["height_cm"], density=d["density"], machine_colors=d["machine_colors"],
            consent_train=d["consent_train"], design_class=d["design_class"],
            class_source=d.get("class_source", "designer"), profile_name=d.get("profile_name", "wilton_default"),
            yarn_csv=Path(d["yarn_csv"]) if d.get("yarn_csv") else None, batch=d.get("batch", False),
            override=d.get("override", {}), corners_override=d.get("corners_override"),
            palette_override=d.get("palette_override"), no_gate=d.get("no_gate", False),
            confirm_palette=d.get("confirm_palette", False), runs_dir=run_dir.parent,
        )
        job.params = d.get("resolved_params", {})
        job.timings = d.get("timings_s", {})
        job.state = d.get("state", {})
        job.load_profile()
        return job

    # ---- parameter resolution
    def load_profile(self):
        self.profile_path = CONFIG_DIR / "looms" / f"{self.profile_name}.yaml"
        if not self.profile_path.exists():
            raise JobError("profile_missing", str(self.profile_path))
        self.profile = load_yaml(self.profile_path)

    def resolve(self, design_class: Optional[str] = None) -> dict:
        """(Re)resolve the parameter layers. Called by intake, and again by structure in auto mode."""
        if not self.profile:
            self.load_profile()
        dc = design_class or self.design_class
        class_cfg = load_class_config(dc) if dc in DESIGN_CLASSES else {}
        override = dict(self.override)
        tentative = {**self.profile, **class_cfg, **{k: v for k, v in override.items() if v is not None}}
        agent_cfg = None
        if tentative.get("design_mode") == "edit":
            name = tentative.get("edit_agent")
            if not name:
                raise JobError("edit_agent_missing", "design_mode: edit requires edit_agent")
            agent_cfg = load_agent_config(name)
        self.params = resolve_params(self.profile, class_cfg, override, agent_cfg)
        self.params["design_class"] = dc
        self.params["design_mode_requested"] = self.params.get("design_mode", "quantize")
        if self.params["design_mode"] not in DESIGN_MODES:
            raise JobError("bad_design_mode", str(self.params["design_mode"]))
        return self.params

    def apply_routing(self, thin_line_frac: float):
        """Routing rules, applied at the end of structure (their inputs are computed there)."""
        mode = self.params.get("design_mode", "quantize")
        seg_available = (MODELS_DIR / "knotvision_segment.pt").exists()
        if mode == "segment" and not seg_available:
            self.params["design_mode"] = "quantize"
            self.warn("segment_fallback")
        elif mode == "edit" and thin_line_frac >= float(self.params.get("edit_thin_line_max", 0.01)):
            class_default = self.params.get("design_mode_class_default", "quantize")
            fallback = class_default if class_default != "edit" else "quantize"
            if fallback == "segment" and not seg_available:
                fallback = "quantize"
                self.warn("segment_fallback")
            self.params["design_mode"] = fallback
            self.warn("edit_thin_lines")
        return self.params["design_mode"]

    @property
    def is_edit(self) -> bool:
        return self.params.get("design_mode") == "edit"

    # ---- cache
    def params_hash(self, keys: Optional[tuple] = None, exclude_prefix: Optional[str] = None) -> str:
        """Hash of the params subset `keys` (missing keys hash as None). Without `keys`: every param
        except those starting with `exclude_prefix` (legacy full-set hash)."""
        if keys is not None:
            return sha256_obj({k: self.params.get(k) for k in keys})
        p = {k: v for k, v in self.params.items()
             if not (exclude_prefix and k.startswith(exclude_prefix)) and k != "agent"}
        if exclude_prefix != "edit_" and self.params.get("agent"):
            p["agent"] = self.params["agent"]
        return sha256_obj(p)

    def cache_key(self, stage: str, inputs: list, extra: Optional[dict] = None,
                  exclude_prefix: Optional[str] = None, keys: Optional[tuple] = None) -> str:
        """Stage cache key: input file hashes + the params the stage reads (STAGE_PARAM_KEYS[stage]
        unless `keys` is given) + `extra`."""
        if keys is None:
            keys = STAGE_PARAM_KEYS.get(stage)
        parts = {"stage": stage, "params": self.params_hash(keys, exclude_prefix), "extra": extra or {},
                 "inputs": []}
        for x in inputs:
            x = Path(x)
            parts["inputs"].append(sha256_file(x) if x.exists() else f"missing:{x.name}")
        return sha256_obj(parts)

    def cached(self, stage_dir: Path, key: str) -> bool:
        f = stage_dir / "input_hash.txt"
        hit = f.exists() and f.read_text().strip() == key and (stage_dir / "done").exists()
        if hit:
            wf = stage_dir / "warnings.json"
            if wf.exists():
                for w in json.loads(wf.read_text()):
                    self.warn(w)
            stage = self.state.get("_stage")
            if stage and stage in STAGES and stage not in self.state.get("stages_ran", []):
                self.state.setdefault("stages_cached", [])
                if stage not in self.state["stages_cached"]:
                    self.state["stages_cached"].append(stage)
        return hit

    def mark_done(self, stage_dir: Path, key: str):
        (stage_dir / "input_hash.txt").write_text(key)
        stage = self.state.get("_stage")
        per = self.state.get("warnings_by_stage", {}).get(stage, []) if stage else []
        (stage_dir / "warnings.json").write_text(json.dumps(per))
        (stage_dir / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))

    def invalidate_from(self, stage: str):
        """rerun --from <stage>: drop the done markers of that stage and everything after it."""
        i = STAGES.index(stage)
        for s in STAGES[i:]:
            d = self.stages_dir / f"{STAGES.index(s):02d}_{s}"
            if (d / "done").exists():
                (d / "done").unlink()
        self.state["stages_ran"] = [s for s in self.state.get("stages_ran", []) if STAGES.index(s) < i]
        self.state["stages_cached"] = [s for s in self.state.get("stages_cached", []) if STAGES.index(s) < i]


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(path: Path | str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- retention

def apply_retention(runs_dir: Path = RUNS_DIR, now: Optional[float] = None) -> list:
    """Delete run dirs older than retention_days when consent_train is false; always delete edit
    intermediates (stages/05_design/edit/) after retention_days. Returns list of removed paths."""
    now = now or time.time()
    removed = []
    for run in sorted(Path(runs_dir).glob("*/")):
        jj = run / "job.json"
        if not jj.exists():
            continue
        try:
            d = json.loads(jj.read_text())
        except Exception:
            continue
        days = float(d.get("resolved_params", {}).get("retention_days", 30))
        age_days = (now - jj.stat().st_mtime) / 86400
        if age_days < days:
            continue
        edit_dir = run / "stages" / "05_design" / "edit"
        if edit_dir.exists():
            shutil.rmtree(edit_dir)
            removed.append(str(edit_dir))
        if not d.get("consent_train", False):
            shutil.rmtree(run)
            removed.append(str(run))
    return removed
