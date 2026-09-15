"""design / segment: deterministic SegNet (convnext_tiny features_only + FPN, stride 4) -> per knot
Lab (3, regression) + boundary probability (1, sigmoid). Grid-aligned tiling: 512x512 px tiles,
64 px overlap, central 448x448 kept.

torch/timm are optional imports: without them (or without models/knotvision_segment.pt) the caller
falls back to quantize with warnings += segment_fallback (checkpoint absence is caught in structure).

Output: design_lab.npy (H x W x 3), boundary.npy (H x W), design_rgb.png
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import PX_PER_KNOT
from .color import lab_to_rgb
from .job import MODELS_DIR, Job, JobError

TILE, OVERLAP = 512, 64
try:  # optional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


if HAS_TORCH:
    class SegNet(nn.Module):
        """convnext_tiny (timm, features_only, out_indices 0-3) + FPN decoder to stride 4, 4 channels."""

        def __init__(self, pretrained: bool = True, fpn_ch: int = 128):
            super().__init__()
            import timm
            self.enc = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3))
            chs = self.enc.feature_info.channels()
            self.lat = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in chs])
            self.smooth = nn.ModuleList([nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1) for _ in chs])
            self.head = nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.GELU(), nn.Conv2d(fpn_ch, 4, 1))

        def forward(self, x):
            feats = self.enc(x)
            p = self.lat[-1](feats[-1])
            for i in range(len(feats) - 2, -1, -1):
                p = F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](feats[i])
                p = self.smooth[i](p)
            out = self.head(p)  # stride 4 == one knot at 4 px/knot
            lab = out[:, :3]
            lab = torch.stack([lab[:, 0] * 100.0, lab[:, 1] * 128.0, lab[:, 2] * 128.0], dim=1)
            boundary = torch.sigmoid(out[:, 3:4])
            return lab, boundary


def load_model(ckpt: Path = MODELS_DIR / "knotvision_segment.pt"):
    if not HAS_TORCH:
        raise JobError("segment_fallback", "torch not installed")
    if not ckpt.exists():
        raise JobError("segment_fallback", f"missing {ckpt}")
    model = SegNet(pretrained=False)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model


def infer_tiled(model, rect_rgb: np.ndarray, W: int, H: int, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Tiles of TILE px (multiples of 4), OVERLAP px overlap, central region kept. Returns Lab (H,W,3), boundary (H,W)."""
    ph, pw = rect_rgb.shape[:2]
    keep = TILE - 2 * OVERLAP
    x = torch.from_numpy(rect_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)
    lab = np.zeros((H, W, 3), np.float32)
    bnd = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, ph - TILE + 1), keep)) + [max(0, ph - TILE)]
    xs = list(range(0, max(1, pw - TILE + 1), keep)) + [max(0, pw - TILE)]
    with torch.no_grad():
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                tile = x[:, :, y0:y0 + TILE, x0:x0 + TILE]
                th, tw = tile.shape[-2:]
                pad_h, pad_w = TILE - th, TILE - tw
                if pad_h or pad_w:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
                l, b = model(tile)
                l = l[0].permute(1, 2, 0).cpu().numpy()[: th // PX_PER_KNOT, : tw // PX_PER_KNOT]
                b = b[0, 0].cpu().numpy()[: th // PX_PER_KNOT, : tw // PX_PER_KNOT]
                gy0, gx0 = y0 // PX_PER_KNOT, x0 // PX_PER_KNOT
                # central crop unless at the image border
                cy0 = 0 if y0 == 0 else OVERLAP // PX_PER_KNOT
                cx0 = 0 if x0 == 0 else OVERLAP // PX_PER_KNOT
                cy1 = l.shape[0] if y0 + TILE >= ph else l.shape[0] - OVERLAP // PX_PER_KNOT
                cx1 = l.shape[1] if x0 + TILE >= pw else l.shape[1] - OVERLAP // PX_PER_KNOT
                lab[gy0 + cy0:gy0 + cy1, gx0 + cx0:gx0 + cx1] += l[cy0:cy1, cx0:cx1]
                bnd[gy0 + cy0:gy0 + cy1, gx0 + cx0:gx0 + cx1] += b[cy0:cy1, cx0:cx1]
                cnt[gy0 + cy0:gy0 + cy1, gx0 + cx0:gx0 + cx1] += 1
    cnt = np.maximum(cnt, 1)
    return lab / cnt[..., None], bnd / cnt


def run(job: Job, rectified_png: Path, out_dir: Path) -> dict:
    t0 = time.time()
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    model = load_model()
    rect = cv2.cvtColor(cv2.imread(str(rectified_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    model.to(device)
    lab, bnd = infer_tiled(model, rect, W, H, device)
    np.save(out_dir / "design_lab.npy", lab)
    np.save(out_dir / "boundary.npy", bnd)
    rgb = lab_to_rgb(lab)
    rgb = cv2.resize(rgb, (W * PX_PER_KNOT, H * PX_PER_KNOT), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "design_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    unc = float(((bnd > 0.3) & (bnd < 0.7)).mean())
    return {"mode": "segment", "wall_s": round(time.time() - t0, 3), "model_uncertainty": unc}
