"""setup-check: list missing setup items and the gate each one triggers; probe edit backends.

Every field that runs on a default produces a hard gate somewhere (needs_review), never a silent pass.
"""
from __future__ import annotations

import os
from pathlib import Path

from .job import CONFIG_DIR, DATA_DIR, MODELS_DIR, load_yaml


def check(profile_name: str = "wilton_default", probe_api: bool = False) -> list[dict]:
    """Returns a list of {item, status: ok|missing|assumed, detail, gate}."""
    items = []
    pp = CONFIG_DIR / "looms" / f"{profile_name}.yaml"
    if not pp.exists():
        return [{"item": "loom_profile", "status": "missing", "detail": str(pp), "gate": "profile_missing (job cannot start)"}]
    P = load_yaml(pp)

    def add(item, ok, detail, gate, assumed=False):
        items.append({"item": item, "status": "ok" if ok else ("assumed" if assumed else "missing"),
                      "detail": detail, "gate": None if ok else gate})

    add("reed_per_m", P.get("reed_per_m") is not None, f"profile: {P.get('reed_per_m')}",
        "grid_unverified on every job (square assumption); first job's measured_reed_per_m fills the profile", assumed=True)
    add("photo_survey", (DATA_DIR / "photo_survey.csv").exists(), "data/photo_survey.csv",
        "pair-collection source decision cannot be made (tools/photo_survey.py)")
    add("dither_pattern", P.get("dither_pattern") is not None, f"profile: {P.get('dither_pattern')}",
        "dither_pattern_default on every job with a gradient mask (tools/dither_stats.py + designer confirmation)", assumed=True)
    yarns = sorted(p.name for p in (CONFIG_DIR / "yarns").glob("*.csv"))
    add("yarn_library", bool(yarns), ", ".join(yarns) or "none", "palette from photo, yarn_code empty (informational)")
    add("motif_min_knots", P.get("motif_min_knots_source") == "archive",
        f"{P.get('motif_min_knots')} ({P.get('motif_min_knots_source', 'assumed')})",
        "guard and cc_added_motif use an assumed threshold (tools/motif_stats.py)", assumed=True)
    add("noise_scale", P.get("noise_scale_source") == "archive",
        f"[{P.get('noise_cc_min')}, {P.get('noise_cc_max')}] ({P.get('noise_scale_source', 'assumed')})",
        "noise_frac / distressed detection use assumed scale (tools/noise_stats.py)", assumed=True)
    add("qc_calib", (MODELS_DIR / "qc_calib.json").exists(), "models/qc_calib.json",
        "needs_review on every job: score computed, no decision (eval/calibrate_qc.py)")
    sam_ok = any((MODELS_DIR / "sam3.1").glob("*.pt")) if (MODELS_DIR / "sam3.1").exists() else False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # sam3 decorates CUDA autocast at import time
            import torch  # noqa: F401
            import sam3  # noqa: F401
        sam_pkg = True
    except Exception:
        sam_pkg = False
    add("sam", sam_ok and sam_pkg, f"models/sam3.1/*.pt ({'found' if sam_ok else 'missing'}), sam3+torch package "
        f"({'ok' if sam_pkg else 'missing'})", "sam_unavailable -> needs_corners on every job without corners_override")
    add("segment_ckpt", (MODELS_DIR / "knotvision_segment.pt").exists(), "models/knotvision_segment.pt",
        "design_mode: segment falls back to quantize (segment_fallback) (train/train_segment.py)")
    agents = sorted(p.stem for p in (CONFIG_DIR / "agents").glob("*.yaml"))
    add("edit_agents", bool(agents), ", ".join(agents), "design_mode: edit raises edit_agent_missing")
    for name, d in (("qwen_edit_2511", "qwen_image_edit_2511"), ("qwen_edit_2509", "qwen_image_edit_2509")):
        add(f"edit_backend:{name}", _hf_snapshot_complete(MODELS_DIR / d), f"models/{d}/",
            f"{name} agents -> edit_backend_error -> edit_fallback (quantize baseline)")
    key = bool(os.environ.get("MUSE_API_KEY"))
    allow = bool(P.get("allow_api"))
    detail = f"MUSE_API_KEY={'set' if key else 'unset'}, allow_api={allow}, provider={P.get('muse_provider')}"
    if probe_api and key and allow:
        from .agents.backends.muse_image import MuseImageBackend
        ok = MuseImageBackend.test_request(P.get("muse_provider", "meta"))
        detail += f", probe={'ok' if ok else 'failed'}"
        add("edit_backend:muse_image", ok, detail, "muse_image agents -> edit_backend_error")
    else:
        add("edit_backend:muse_image", key and allow, detail,
            "muse_image agents -> api_not_allowed (allow_api false) or edit_backend_error (no key)")
    for name in ("classic", "geometric", "gradient", "distressed", "minimal"):
        add(f"class:{name}", (CONFIG_DIR / "classes" / f"{name}.yaml").exists(), f"config/classes/{name}.yaml",
            "class_config_missing for jobs of this class")
    return items


def _hf_snapshot_complete(d: Path) -> bool:
    """A diffusers snapshot counts only when model_index.json exists, every safetensors shard listed
    in the transformer / text_encoder index files is present, and no `hf download` `.incomplete`
    part is left behind (an interrupted download leaves the directory in place)."""
    if not (d / "model_index.json").exists():
        return False
    if any(d.rglob("*.incomplete")):
        return False
    import json
    for idx in d.glob("*/*.safetensors.index.json"):
        try:
            names = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except Exception:
            return False
        if any(not (idx.parent / n).exists() for n in names):
            return False
    return True


def format_report(items: list[dict]) -> str:
    lines = []
    w = max(len(i["item"]) for i in items)
    for i in items:
        mark = {"ok": "ok     ", "assumed": "ASSUMED", "missing": "MISSING"}[i["status"]]
        line = f"{mark}  {i['item']:<{w}}  {i['detail']}"
        if i["gate"]:
            line += f"\n{'':9}{'':{w}}  -> {i['gate']}"
        lines.append(line)
    n_bad = sum(1 for i in items if i["status"] != "ok")
    lines.append(f"\n{n_bad} item(s) missing or assumed; each produces the listed gate until filled.")
    return "\n".join(lines)
