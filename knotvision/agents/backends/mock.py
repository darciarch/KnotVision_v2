"""Mock backend for tests (no GPU). Behaviour is controlled by agent_cfg["mock"]:

    mode: identity | shift | add_block | remove_block | degrade | fail
    shift_px: [dx, dy]      (mode shift)
    block: [x0, y0, x1, y1] in input pixels, color: [r, g, b]   (add_block / remove_block)
    fail_times: n           (raise BackendError on the first n calls)
    round_fix: true         (from round 1 on, return identity -> loop must stop)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from ..base import BackendError


class MockBackend:
    name = "mock"
    version = "mock-1"
    max_long_px = 2048
    supports_seed = True
    api_egress = False
    price_usd = 0.0

    def __init__(self, agent_cfg: dict):
        self.cfg = dict(agent_cfg.get("mock", {}))
        self.calls = 0
        self.prompts: list[str] = []

    def edit(self, images: Sequence[Image.Image], prompt: str, negative_prompt: str, seed: int, params: dict) -> Image.Image:
        self.calls += 1
        self.prompts.append(prompt)
        fail_times = int(self.cfg.get("fail_times", 0))
        if self.calls <= fail_times:
            raise BackendError("mock failure")
        img = np.array(images[0].convert("RGB"))
        mode = self.cfg.get("mode", "identity")
        rnd = int(params.get("round", 0))
        if self.cfg.get("round_fix") and rnd > 0:
            mode = "identity"
        if mode == "shift":
            dx, dy = self.cfg.get("shift_px", [3, 0])
            out = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        elif mode in ("add_block", "remove_block"):
            out = img.copy()
            x0, y0, x1, y1 = self.cfg.get("block", [10, 10, 60, 60])
            if mode == "add_block":
                out[y0:y1, x0:x1] = np.array(self.cfg.get("color", [255, 0, 0]), np.uint8)
            else:
                # paint the block with the image's dominant colour (removes whatever motif was there)
                vals, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
                out[y0:y1, x0:x1] = vals[np.argmax(counts)]
        elif mode == "degrade":
            rng = np.random.RandomState(seed)
            noise = rng.normal(0, float(self.cfg.get("sigma", 40)), img.shape)
            out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            out = img
        return Image.fromarray(out)
