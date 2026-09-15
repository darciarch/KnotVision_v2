"""Edit-mode agents: backend + prompt policy + candidate count + rounds + selection weights.

An agent is a YAML file under config/agents/. The loop (loop.py) is shared; backends live in
backends/. The critic is the set of measurements, not a language model.
"""
