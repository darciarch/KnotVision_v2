import sys
from pathlib import Path

import yaml

from knotvision.agents.loop import FEATURE_SCALE
from knotvision.job import CONFIG_DIR, agent_hash, load_yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from agents import agents_to_calibrate, pick_winners  # noqa: E402

KNOWN_BACKENDS = {"qwen_edit_2511", "qwen_edit_2509", "muse_image", "mock"}


def test_agent_files_schema():
    files = sorted((CONFIG_DIR / "agents").glob("*.yaml"))
    assert files
    for f in files:
        cfg = load_yaml(f)
        assert cfg["backend"] in KNOWN_BACKENDS, f
        assert int(cfg.get("edit_n_candidates", 1)) >= 1 and int(cfg.get("edit_max_rounds", 1)) >= 1
        assert set(cfg["select_weights"]) <= set(FEATURE_SCALE), f
        if cfg["backend"].startswith("qwen"):
            assert "num_inference_steps" in cfg and "true_cfg_scale" in cfg


def test_agent_hash_changes_with_prompt_template(monkeypatch):
    h1 = agent_hash("qwen2511_loop")
    prompts = Path(__file__).resolve().parents[1] / "knotvision" / "agents" / "prompts.py"
    orig = prompts.read_bytes()
    try:
        prompts.write_bytes(orig + b"\n# touched\n")
        h2 = agent_hash("qwen2511_loop")
    finally:
        prompts.write_bytes(orig)
    assert h1 != h2 and agent_hash("qwen2511_loop") == h1
    assert agent_hash("qwen2511_loop") != agent_hash("qwen2511_direct")


def _row(cls, agent, passed, qc, t, cost, tiling="auto"):
    return {"design_class": cls, "mode": "edit", "edit_agent": agent, "tiling": tiling, "release_pass": passed,
            "qc_score": qc, "time_s_median": t, "cost_usd_median": cost}


def test_winner_rule_and_ties():
    table = [_row("geometric", "a", True, 0.8, 100, 0.0), _row("geometric", "b", True, 0.9, 500, 0.0),
             _row("geometric", "c", False, 0.99, 10, 0.0),
             _row("gradient", "a", True, 0.7, 100, 0.0), _row("gradient", "b", True, 0.7, 50, 0.2),
             _row("classic", "a", False, 0.9, 1, 0.0)]
    w = pick_winners(table)
    assert w["geometric"]["agent"] == "b"          # highest qc_score among passing
    assert w["gradient"]["agent"] == "b"           # tie on qc -> cheaper time
    assert w["classic"] is None and w["distressed"] is None
    table2 = [_row("gradient", "a", True, 0.7, 50, 0.0), _row("gradient", "b", True, 0.7, 50, 0.2)]
    assert pick_winners(table2)["gradient"]["agent"] == "a"   # tie on time -> cheaper cost


def test_at_most_two_agents_calibrated():
    winners = {"classic": {"agent": "a"}, "geometric": {"agent": "b"}, "gradient": {"agent": "c"},
               "distressed": {"agent": "a"}, "minimal": None}
    assert agents_to_calibrate(winners) == ["a", "b"]
