"""Qwen-Image-Edit-2511 / 2509 backend (diffusers QwenImageEditPlusPipeline, Apache 2.0).

Weights: models/qwen_image_edit_2511/ (or _2509), downloaded from HF. Optional LoRA under
models/edit_lora/<name>.safetensors (agent: edit_lora), optional Lightning 4-step LoRA (agent:
lightning_lora: true). Multi-image input: [edit_input, palette_card, (photo on rounds > 0)].

diffusers/transformers/torch are imported lazily so that the modelless chain and the tests do not
need them. Any failure raises BackendError -> the loop retries 3x then falls back to the baseline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from ...job import MODELS_DIR
from ..base import BackendError

HF_IDS = {"qwen_edit_2511": "Qwen/Qwen-Image-Edit-2511", "qwen_edit_2509": "Qwen/Qwen-Image-Edit-2509"}
LOCAL_DIRS = {"qwen_edit_2511": MODELS_DIR / "qwen_image_edit_2511", "qwen_edit_2509": MODELS_DIR / "qwen_image_edit_2509"}


class QwenEditBackend:
    supports_seed = True
    api_egress = False
    price_usd = 0.0
    max_long_px = 1536  # single-pass budget (~1-2 MP); tiling handles larger rugs

    def __init__(self, name: str, agent_cfg: dict):
        self.name = name
        self.cfg = agent_cfg
        self.version = HF_IDS[name]
        self._pipe = None

    @staticmethod
    def available(name: str) -> bool:
        return LOCAL_DIRS[name].exists()

    def _load(self):  # pragma: no cover - needs weights + GPU
        if self._pipe is not None:
            return self._pipe
        local = LOCAL_DIRS[self.name]
        if not local.exists():
            raise BackendError(f"weights missing: {local}")
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
            pipe = QwenImageEditPlusPipeline.from_pretrained(str(local), torch_dtype=torch.bfloat16)
            lora = self.cfg.get("edit_lora")
            if lora:
                p = MODELS_DIR / "edit_lora" / f"{lora}.safetensors"
                if not p.exists():
                    raise BackendError(f"edit_lora missing: {p}")
                pipe.load_lora_weights(str(p), adapter_name="knotvision")
            if self.cfg.get("lightning_lora"):
                pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", adapter_name="lightning")
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipe
            try:
                rev = getattr(pipe.config, "_name_or_path", None)
                self.version = f"{HF_IDS[self.name]}@{rev}" if rev else HF_IDS[self.name]
            except Exception:
                pass
            return pipe
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"{self.name} load failed: {e}")

    def edit(self, images: Sequence[Image.Image], prompt: str, negative_prompt: str, seed: int, params: dict) -> Image.Image:  # pragma: no cover
        pipe = self._load()
        try:
            import torch
            gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
            steps = 4 if self.cfg.get("lightning_lora") else int(self.cfg.get("num_inference_steps", 30))
            out = pipe(image=list(images), prompt=prompt, negative_prompt=negative_prompt,
                       true_cfg_scale=float(self.cfg.get("true_cfg_scale", 4.0)),
                       num_inference_steps=steps, generator=gen, num_images_per_prompt=1)
            return out.images[0].convert("RGB")
        except Exception as e:
            raise BackendError(f"{self.name} inference failed: {e}")
