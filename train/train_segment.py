#!/usr/bin/env python
"""SegNet training (convnext_tiny features_only + FPN, stride 4, 4 channels: Lab regression + boundary).

Input 4 px/knot, 512x512 tiles (4 px aligned), batch 16, AdamW 3e-4, cosine, 40 epochs, bf16. Tile
sampling balanced by design_class. Loss = loss_mask-masked L1(Lab) + lambda * BCE(boundary), lambda 1.
Augmentation: flips (pair together), colour jitter (input only, +-10 %), scale 0.9-1.1 (grid aligned),
online pile shading / fibre blur / camera noise / JPEG, resolution augmentation (downscale to 1-4 px/knot
then back to 4; distribution from data/photo_survey.csv when present). Val every 2 epochs: loss + the
full pipeline on 2 samples per class (eval metrics). Output: models/knotvision_segment.pt.

torch / timm are required here (not for the pipeline). Usage:
    python train/train_segment.py --data train/data [--epochs 40] [--batch 16] [--out models/knotvision_segment.pt]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision import PX_PER_KNOT  # noqa: E402
from knotvision.job import DATA_DIR, MODELS_DIR  # noqa: E402

TILE_PX = 512
TILE_KNOTS = TILE_PX // PX_PER_KNOT


# ----------------------------------------------------------------------------- data

def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def resolution_dist() -> np.ndarray:
    """px/knot values to sample the resolution augmentation from (survey when present)."""
    s = DATA_DIR / "photo_survey.csv"
    if s.exists():
        with open(s, newline="") as f:
            v = [float(r["px_per_knot_geo"]) for r in csv.DictReader(f) if r.get("px_per_knot_geo")]
        if v:
            return np.clip(np.asarray(v), 1.0, 4.0)
    return np.linspace(1.0, 4.0, 13)


def augment_input(rgb: np.ndarray, rng: np.random.RandomState, ppk_choices: np.ndarray) -> np.ndarray:
    """Online photo-like degradation of the rectified tile (input only)."""
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]
    # colour jitter +-10 %
    img *= rng.uniform(0.9, 1.1, (1, 1, 3))
    # pile shading: linear ramp
    yy, xx = np.mgrid[0:h, 0:w]
    theta = rng.uniform(0, 2 * np.pi)
    ramp = (xx / w - 0.5) * np.cos(theta) + (yy / h - 0.5) * np.sin(theta)
    img *= (1 + rng.uniform(0.0, 0.25) * ramp)[..., None]
    # fibre blur 0.3-3 knots
    sigma = rng.uniform(0.3, 3.0) * PX_PER_KNOT / 2.355
    img = cv2.GaussianBlur(img, (0, 0), max(0.3, sigma))
    # resolution augmentation: downscale to ppk then back to 4
    ppk = float(rng.choice(ppk_choices))
    if ppk < PX_PER_KNOT:
        s = ppk / PX_PER_KNOT
        small = cv2.resize(img, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    # camera noise + JPEG
    img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    q = int(rng.randint(75, 96))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class TileSampler:
    """Class-balanced tile sampling over the prepared pairs (tiles with skip=True are never drawn)."""

    def __init__(self, recs: list[dict], seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.by_class = {}
        for r in recs:
            tiles = [t for t in r["tiles"] if not t["skip"]]
            if tiles:
                self.by_class.setdefault(r["design_class"], []).append((r, tiles))
        self.classes = sorted(self.by_class)
        self.cache = {}

    def _load(self, rec):
        d = Path(rec["dir"])
        if rec["pair"] not in self.cache:
            self.cache[rec["pair"]] = {
                "rgb": cv2.cvtColor(cv2.imread(str(d / "rectified.png")), cv2.COLOR_BGR2RGB),
                "lab": np.load(d / "target_lab.npy"), "bnd": np.load(d / "target_boundary.npy"),
                "mask": np.load(d / "loss_mask.npy")}
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[rec["pair"]]

    def sample(self, ppk_choices):
        cls = self.classes[self.rng.randint(len(self.classes))]
        rec, tiles = self.by_class[cls][self.rng.randint(len(self.by_class[cls]))]
        t = tiles[self.rng.randint(len(tiles))]
        data = self._load(rec)
        y, x = t["y"], t["x"]
        # scale 0.9-1.1, grid aligned: pick a knot window of TILE_KNOTS * s and resize to TILE
        s = self.rng.uniform(0.9, 1.1)
        k = int(round(TILE_KNOTS * s))
        H, W = data["lab"].shape[:2]
        k = min(k, H, W)
        y = min(y, H - k); x = min(x, W - k)
        rgb = data["rgb"][y * 4:(y + k) * 4, x * 4:(x + k) * 4]
        lab = data["lab"][y:y + k, x:x + k]; bnd = data["bnd"][y:y + k, x:x + k]; msk = data["mask"][y:y + k, x:x + k]
        rgb = cv2.resize(rgb, (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC)
        lab = cv2.resize(lab, (TILE_KNOTS, TILE_KNOTS), interpolation=cv2.INTER_LINEAR)
        bnd = cv2.resize(bnd, (TILE_KNOTS, TILE_KNOTS), interpolation=cv2.INTER_NEAREST)
        msk = cv2.resize(msk.astype(np.uint8), (TILE_KNOTS, TILE_KNOTS), interpolation=cv2.INTER_NEAREST) > 0
        if self.rng.uniform() < 0.5:
            rgb, lab, bnd, msk = rgb[:, ::-1], lab[:, ::-1], bnd[:, ::-1], msk[:, ::-1]
        if self.rng.uniform() < 0.5:
            rgb, lab, bnd, msk = rgb[::-1], lab[::-1], bnd[::-1], msk[::-1]
        rgb = augment_input(np.ascontiguousarray(rgb), self.rng, ppk_choices)
        return rgb, np.ascontiguousarray(lab), np.ascontiguousarray(bnd), np.ascontiguousarray(msk)


# ----------------------------------------------------------------------------- training

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "train" / "data"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=str(MODELS_DIR / "knotvision_segment.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("torch is required for training (pip install torch timm)", file=sys.stderr)
        return 2
    from knotvision.design_seg import SegNet
    torch.manual_seed(a.seed)
    train = load_jsonl(Path(a.data) / "train.jsonl")
    val = load_jsonl(Path(a.data) / "val.jsonl")
    if not train:
        print("no training records", file=sys.stderr)
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    sampler = TileSampler(train, a.seed)
    vsampler = TileSampler(val, a.seed + 1) if val else None
    ppk = resolution_dist()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    use_bf16 = device == "cuda"

    def batch(smp):
        rgbs, labs, bnds, msks = zip(*[smp.sample(ppk) for _ in range(a.batch)])
        x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = (x - mean) / std
        y_lab = torch.from_numpy(np.stack(labs)).permute(0, 3, 1, 2).float().to(device)
        y_b = torch.from_numpy(np.stack(bnds)).unsqueeze(1).float().to(device)
        m = torch.from_numpy(np.stack(msks)).unsqueeze(1).float().to(device)
        return x, y_lab, y_b, m

    def loss_fn(x, y_lab, y_b, m):
        lab, bnd = model(x)
        l1 = (torch.abs(lab - y_lab) * m).sum() / (m.sum() * 3 + 1e-6)
        bce = F.binary_cross_entropy(bnd.clamp(1e-5, 1 - 1e-5), y_b, weight=m, reduction="sum") / (m.sum() + 1e-6)
        return l1 + a.lam * bce, l1.item(), bce.item()

    step = 0
    log = []
    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tl = []
        for _ in range(a.steps_per_epoch):
            x, y_lab, y_b, m = batch(sampler)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, l1, bce = loss_fn(x, y_lab, y_b, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tl.append(loss.item())
        rec = {"epoch": ep, "train_loss": float(np.mean(tl)), "time_s": time.time() - t0}
        if vsampler and ep % 2 == 1:
            model.eval()
            vl = []
            with torch.no_grad():
                for _ in range(20):
                    x, y_lab, y_b, m = batch(vsampler)
                    loss, _, _ = loss_fn(x, y_lab, y_b, m)
                    vl.append(loss.item())
            rec["val_loss"] = float(np.mean(vl))
            if rec["val_loss"] < best:
                best = rec["val_loss"]
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": best}, a.out)
        log.append(rec)
        print(json.dumps(rec))
    if best == float("inf"):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": a.epochs - 1}, a.out)
    (ROOT / "train" / "runs").mkdir(exist_ok=True)
    (ROOT / "train" / "runs" / f"segment_{time.strftime('%Y%m%d_%H%M')}.json").write_text(json.dumps(log, indent=2))
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
