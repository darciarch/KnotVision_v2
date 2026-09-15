"""Backend protocol for edit mode.

    edit(images, prompt, negative_prompt, seed, params) -> PIL.Image
    max_long_px     : longest side the backend can handle in one pass
    supports_seed   : False for API backends without a seed (muse_image)
    name, version   : reported in versions.edit_backend_version

Backends raise BackendError on load / inference / API failures. The loop retries three times and
then falls back to the quantize baseline (edit_fallback).
"""
from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image


class BackendError(Exception):
    pass


class Backend(Protocol):
    name: str
    version: str
    max_long_px: int
    supports_seed: bool
    api_egress: bool
    price_usd: float

    def edit(self, images: Sequence[Image.Image], prompt: str, negative_prompt: str, seed: int,
             params: dict) -> Image.Image: ...


def make_backend(name: str, agent_cfg: dict, profile: dict) -> Backend:
    if name in ("qwen_edit_2511", "qwen_edit_2509"):
        from .backends.qwen_edit import QwenEditBackend
        return QwenEditBackend(name, agent_cfg)
    if name == "muse_image":
        from .backends.muse_image import MuseImageBackend
        return MuseImageBackend(agent_cfg, provider=profile.get("muse_provider", "meta"),
                                price_usd=float(profile.get("muse_price_usd", 0.01)))
    if name == "mock":
        from .backends.mock import MockBackend
        return MockBackend(agent_cfg)
    raise BackendError(f"unknown backend {name}")
