"""Muse Image 1.0 (Meta) backend via the Meta Model API or provider adapters (fal / Runware /
OpenRouter). No weights, no seed (nondeterministic), ~$0.01 per image, the customer photo leaves the
machine (api_egress: true). Requires MUSE_API_KEY and allow_api: true in the loom profile.

The exact request schema differs per provider; each adapter is a small function returning the
result image bytes. Adapters are best-effort and must be verified against the provider docs before
production use (setup-check sends a test request).
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Sequence

import requests
from PIL import Image

from ..base import BackendError

ENDPOINTS = {
    "meta": "https://api.meta.ai/v1/images/edits",
    "fal": "https://fal.run/fal-ai/muse-image/edit",
    "runware": "https://api.runware.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class MuseImageBackend:
    name = "muse_image"
    version = "muse-image-1.0"
    max_long_px = 2048
    supports_seed = False
    api_egress = True

    def __init__(self, agent_cfg: dict, provider: str = "meta", price_usd: float = 0.01, timeout: float = 120.0):
        self.cfg = agent_cfg
        self.provider = provider
        self.price_usd = price_usd
        self.timeout = timeout
        self.key = os.environ.get("MUSE_API_KEY")
        if not self.key:
            raise BackendError("MUSE_API_KEY not set")
        if provider not in ENDPOINTS:
            raise BackendError(f"unknown muse_provider {provider}")

    def _call(self, images: Sequence[Image.Image], prompt: str) -> bytes:
        url = ENDPOINTS[self.provider]
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": "muse-image-1.0", "prompt": prompt, "resolution": self.cfg.get("resolution", "2K"),
                   "images": [_b64(im) for im in images[:10]]}
        if self.provider == "openrouter":
            payload = {"model": "meta/muse-image-1.0",
                       "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}] +
                                     [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(im)}} for im in images[:10]]}],
                       "modalities": ["image"]}
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if r.status_code >= 400:
            raise BackendError(f"muse_image {self.provider} HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        # provider-specific extraction; all deliver base64 or a URL
        b64 = None
        if "data" in data and data["data"]:
            b64 = data["data"][0].get("b64_json")
            u = data["data"][0].get("url")
        elif "images" in data and data["images"]:
            u = data["images"][0].get("url")
            b64 = data["images"][0].get("b64")
        elif "choices" in data:
            imgs = data["choices"][0]["message"].get("images", [])
            u = imgs[0]["image_url"]["url"] if imgs else None
            if u and u.startswith("data:"):
                b64, u = u.split(",", 1)[1], None
        else:
            raise BackendError("muse_image: unexpected response")
        if b64:
            return base64.b64decode(b64)
        if u:
            rr = requests.get(u, timeout=self.timeout)
            if rr.status_code >= 400:
                raise BackendError(f"muse_image download HTTP {rr.status_code}")
            return rr.content
        raise BackendError("muse_image: no image in response")

    def edit(self, images: Sequence[Image.Image], prompt: str, negative_prompt: str, seed: int, params: dict) -> Image.Image:
        last = None
        for attempt in range(3):
            try:
                raw = self._call(images, prompt)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except (BackendError, requests.RequestException) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise BackendError(f"muse_image failed after 3 attempts: {last}")

    @staticmethod
    def test_request(provider: str) -> bool:
        """setup-check probe: is the key accepted? (HEAD/GET on the endpoint root)."""
        key = os.environ.get("MUSE_API_KEY")
        if not key:
            return False
        try:
            r = requests.get(ENDPOINTS[provider], headers={"Authorization": f"Bearer {key}"}, timeout=10)
            return r.status_code < 500 and r.status_code not in (401, 403)
        except requests.RequestException:
            return False
