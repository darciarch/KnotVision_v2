#!/usr/bin/env python
"""Optional edit LoRA for the qwen_edit_* backends (muse_image cannot be trained).

Goal: raise the fidelity obtained with the base prompt. The LoRA-free agent race is the baseline; a LoRA
enters an agent file (edit_lora: <name>) only when the race shows a measured gain.

diffusers + peft; rank-32 LoRA on the MMDiT attention layers; encoders and VAE frozen.
Data: train/data/<pair>/{edit_input.png, flat.png, edit_weight.npy} (train/prepare.py --edit): synthetic
pairs + aligned real pairs; prompt = base prompt + class addition + palette hex list, palette card as
image 2 (the same prompts.py the pipeline uses). Rented GPU (>= 48 GB advised; 24 GB with FP8 base +
gradient checkpointing). 3000-5000 steps, AdamW 1e-4, bf16, no EMA. Every 500 steps: full pipeline on
2 samples per class with qwen2511_direct -> edit_align_px, cc_added_*, knot_deltaE (eval/run.py).
Output: models/edit_lora/<name>.safetensors; its sha256 is versions.edit_lora_sha256.

Usage: python train/train_edit_lora.py --data train/data --name kv_lora_v1 [--backend qwen_edit_2511]
       [--steps 4000] [--rank 32] [--lr 1e-4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotation only; PIL is imported lazily where used
    from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision.agents import prompts as PR  # noqa: E402
from knotvision.agents.backends.qwen_edit import HF_IDS, LOCAL_DIRS  # noqa: E402
from knotvision.color import hex_of  # noqa: E402
from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.job import MODELS_DIR, read_json  # noqa: E402


def load_records(data_dir: Path) -> list[dict]:
    recs = []
    for line in (data_dir / "train.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = Path(r["dir"])
        if (d / "edit_input.png").exists() and (d / "flat.png").exists():
            recs.append(r)
    return recs


def prompt_for(rec: dict, pairs_root: Path | None = None) -> str:
    """Base prompt + class addition + the truth palette's hex list (palette card = image 2)."""
    d = Path(rec["dir"])
    hexes = []
    # the truth palette from the pair's truth.tiff when reachable; else from flat.png's unique colours
    truth = None
    for root in ([pairs_root] if pairs_root else []) + [ROOT / "data" / "synth" / "pairs", ROOT / "data" / "real" / "pairs"]:
        t = root / rec["pair"] / "truth.tiff"
        if t.exists():
            truth = t
            break
    if truth is not None:
        _, rgb = read_indexed_tiff(truth)
        hexes = [hex_of(c) for c in rgb]
    else:
        import cv2
        flat = cv2.cvtColor(cv2.imread(str(d / "flat.png")), cv2.COLOR_BGR2RGB)
        cols = np.unique(flat.reshape(-1, 3), axis=0)[:12]
        hexes = [hex_of(c) for c in cols]
    return PR.base_prompt(hexes, rec["design_class"])


def palette_card_for(hexes_rgb: list) -> "PILImage.Image":
    from PIL import Image
    from knotvision.palette import palette_card
    entries = [{"rgb": list(c)} for c in hexes_rgb]
    return Image.fromarray(palette_card(entries))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "train" / "data"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", default="qwen_edit_2511", choices=list(HF_IDS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    recs = load_records(Path(a.data))
    if not recs:
        print("no edit records; run train/prepare.py --edit first", file=sys.stderr)
        return 1
    try:
        import torch
        from diffusers import QwenImageEditPlusPipeline  # type: ignore[import-not-found]
        from peft import LoraConfig  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"diffusers/peft/torch required: {e}", file=sys.stderr)
        return 2
    local = LOCAL_DIRS[a.backend]
    if not local.exists():
        print(f"weights missing: {local} (download {HF_IDS[a.backend]} from HF)", file=sys.stderr)
        return 2
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(local), torch_dtype=torch.bfloat16).to(device)
    transformer = pipe.transformer
    for p_ in pipe.vae.parameters():
        p_.requires_grad_(False)
    for p_ in pipe.text_encoder.parameters():
        p_.requires_grad_(False)
    for p_ in transformer.parameters():
        p_.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                       target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    transformer.enable_gradient_checkpointing()
    params = [p_ for p_ in transformer.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    rng = np.random.RandomState(a.seed)
    from PIL import Image
    out_dir = MODELS_DIR / "edit_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for step in range(a.steps):
        rec = recs[rng.randint(len(recs))]
        d = Path(rec["dir"])
        src = Image.open(d / "edit_input.png").convert("RGB")
        tgt = Image.open(d / "flat.png").convert("RGB")
        weight = np.load(d / "edit_weight.npy") if (d / "edit_weight.npy").exists() else None
        prompt = prompt_for(rec)
        # --- one flow-matching step on the target latents conditioned on [src, palette card]
        with torch.no_grad():
            x1 = pipe.vae.encode(pipe.image_processor.preprocess(tgt).to(device, torch.bfloat16)).latent_dist.sample()
            x1 = (x1 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "shift_factor") else x1 * pipe.vae.config.scaling_factor
            cond_img = pipe.image_processor.preprocess(src).to(device, torch.bfloat16)
            cond_lat = pipe.vae.encode(cond_img).latent_dist.sample()
            prompt_embeds, prompt_mask = pipe.encode_prompt(prompt, [src], device=device) if hasattr(pipe, "encode_prompt") else (None, None)
        t = torch.rand(x1.shape[0], device=device)
        noise = torch.randn_like(x1)
        xt = (1 - t.view(-1, 1, 1, 1)) * x1 + t.view(-1, 1, 1, 1) * noise
        target_v = noise - x1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            try:
                pred = transformer(hidden_states=xt, encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
                                   timestep=t * 1000, img_shapes=None, return_dict=False)[0]
            except TypeError:
                pred = transformer(xt, prompt_embeds, t * 1000).sample
        w = torch.ones_like(target_v)
        if weight is not None:
            wl = torch.from_numpy(weight).to(device, torch.bfloat16)
            wl = torch.nn.functional.interpolate(wl[None, None], size=target_v.shape[-2:], mode="nearest")
            w = w * wl
        loss = ((pred.float() - target_v.float()) ** 2 * w.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": loss.item()})
            print(json.dumps(log[-1]))
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            from safetensors.torch import save_file
            sd = {k: v.detach().cpu().contiguous() for k, v in transformer.state_dict().items() if "lora" in k}
            save_file(sd, str(out_dir / f"{a.name}.safetensors"))
            print(f"checkpoint {out_dir / (a.name + '.safetensors')} at step {step + 1}; run eval/run.py --mode edit "
                  f"--agent qwen2511_direct (edit_lora: {a.name}) on 2 samples per class for edit_align_px / cc_added_* / knot_deltaE")
    (ROOT / "train" / "runs").mkdir(exist_ok=True)
    (ROOT / "train" / "runs" / f"edit_lora_{a.name}.json").write_text(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
