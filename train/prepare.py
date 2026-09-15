#!/usr/bin/env python
"""Training preparation (train/prepare.py).

1. Every pair runs intake -> structure (real pairs: manual corners from corners.json, class from meta;
   synthetic pairs: corners_gt.json, class from meta).
2. Segment targets, at grid resolution:
   target_lab.npy       truth palette Lab; inside the gradient mask (truth dither U photo gradient mask)
                        the 4x4-knot box mean.
   target_boundary.npy  knots whose index changes, dilated by 1 knot; 0 inside the gradient mask.
   loss_mask.npy        align_mask AND NOT noise_mask.
   Tiles (128x128 knots = 512 px), not pairs, are the skip unit: a tile whose noise_mask share exceeds
   tile_skip_noise_frac (0.2) is skipped; distressed pairs stay in training with their other tiles.
3. Augmentation is online (train_segment.py).
4. train/data/{train,val}.jsonl: one line per pair (design_class, tile list).
5. Edit LoRA targets (optional, --edit): edit_input.png = the pair's rectified.png at the edit
   working resolution (design step 1, 16 px multiple; never the raw photo), flat.png = truth rendered
   flat in the rectified space (real pairs: after H_refined/H_local, align_mask==0 blocks weighted 0).
   layout_changed pairs are excluded.

Usage: python train/prepare.py --pairs data/synth/pairs [--pairs data/real/pairs ...] [--splits data/real/splits.json]
       [--out train/data] [--runs runs] [--edit]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from knotvision import PX_PER_KNOT  # noqa: E402
from knotvision.color import rgb_to_lab  # noqa: E402
from knotvision.design_edit import working_resolution  # noqa: E402
from knotvision.export import read_indexed_tiff  # noqa: E402
from knotvision.job import CONFIG_DIR, RUNS_DIR, Job, JobError, dump_json, load_yaml, read_json  # noqa: E402
from knotvision.metrics import boundary_map  # noqa: E402
from knotvision.run import STAGE_FN  # noqa: E402

TILE_KNOTS = 128
TILE_SKIP_NOISE_FRAC = 0.2


def run_to_structure(pair: Path, runs_dir: Path, profile: str = "wilton_default") -> Job | None:
    meta = read_json(pair / "meta.json")
    photo = next((pair / n for n in ("photo.jpg", "photo.png") if (pair / n).exists()), None)
    corners = None
    for cf in ("corners_gt.json", "corners.json"):
        if (pair / cf).exists():
            corners = read_json(pair / cf)["corners"]
            break
    if photo is None or corners is None:
        return None
    density = meta.get("density") or (meta.get("reed_per_m", 400) * meta.get("wefts_per_m", meta.get("reed_per_m", 400)))
    job = Job(job_id=f"prep_{pair.name}", photo_path=photo, width_cm=meta["width_cm"], height_cm=meta["height_cm"],
              density=density, machine_colors=meta.get("machine_colors", 8), consent_train=True,
              design_class=meta.get("design_class", "auto"), profile_name=profile, batch=True,
              corners_override=corners, runs_dir=runs_dir, override={"allow_low_res": True, "allow_partial": True})
    job.restore_state()
    try:
        for s in ("intake", "rectify", "structure"):
            job.begin_stage(s)
            STAGE_FN[s](job)
    except JobError:
        return None
    job.save()
    return job


def segment_targets(truth: np.ndarray, truth_rgb: np.ndarray, gmask: np.ndarray, align_mask: np.ndarray | None,
                    noise_mask: np.ndarray | None) -> dict:
    H, W = truth.shape
    lab = rgb_to_lab(truth_rgb)[truth].astype(np.float32)
    if gmask.any():
        box = cv2.blur(lab, (4, 4))
        lab = np.where(gmask[..., None], box, lab)
    b = boundary_map(truth)
    b = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    b[gmask] = False
    am = np.ones((H, W), bool) if align_mask is None else align_mask.astype(bool)
    nm = np.zeros((H, W), bool) if noise_mask is None else noise_mask.astype(bool)
    return {"target_lab": lab, "target_boundary": b.astype(np.float32), "loss_mask": (am & ~nm), "noise_mask": nm}


def tile_list(H: int, W: int, noise_mask: np.ndarray, skip_frac: float = TILE_SKIP_NOISE_FRAC, tile: int = TILE_KNOTS) -> list:
    tiles = []
    ys = list(range(0, max(1, H - tile + 1), tile - 16)) + ([max(0, H - tile)] if H > tile else [])
    xs = list(range(0, max(1, W - tile + 1), tile - 16)) + ([max(0, W - tile)] if W > tile else [])
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            nm = noise_mask[y:y + tile, x:x + tile]
            tiles.append({"y": int(y), "x": int(x), "skip": bool(nm.mean() > skip_frac), "noise_frac": float(nm.mean())})
    return tiles


def prepare_pair(pair: Path, job: Job, out_dir: Path, edit: bool = False) -> dict | None:
    truth, truth_rgb = read_indexed_tiff(pair / "truth.tiff")
    meta = read_json(pair / "meta.json")
    if meta.get("layout_changed"):
        return None
    st = job.stage_dir("structure")
    g = job.state["grid"]
    if truth.shape != (g["H"], g["W"]):
        truth = cv2.resize(truth.astype(np.int32), (g["W"], g["H"]), interpolation=cv2.INTER_NEAREST)
    gm_photo = np.load(st / "gradient_mask.npy")
    gm_truth = np.load(pair / "gradient_mask.npy") if (pair / "gradient_mask.npy").exists() else np.zeros_like(gm_photo)
    if gm_truth.shape != gm_photo.shape:
        gm_truth = cv2.resize(gm_truth.astype(np.uint8), (gm_photo.shape[1], gm_photo.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    gmask = gm_photo | gm_truth.astype(bool)
    am = np.load(pair / "align_mask.npy") if (pair / "align_mask.npy").exists() else None
    nm = np.load(pair / "noise_mask.npy") if (pair / "noise_mask.npy").exists() else None
    if nm is not None and nm.shape != truth.shape:
        nm = cv2.resize(nm.astype(np.uint8), (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if am is not None and am.shape != truth.shape:
        am = cv2.resize(am.astype(np.uint8), (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    t = segment_targets(truth, truth_rgb, gmask, am, nm)
    od = out_dir / pair.name
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / "target_lab.npy", t["target_lab"])
    np.save(od / "target_boundary.npy", t["target_boundary"])
    np.save(od / "loss_mask.npy", t["loss_mask"])
    rect_png = job.stage_dir("rectify") / "rectified.png"
    (od / "rectified.png").write_bytes(rect_png.read_bytes())
    tiles = tile_list(truth.shape[0], truth.shape[1], t["noise_mask"])
    rec = {"pair": pair.name, "design_class": job.design_class, "H": int(truth.shape[0]), "W": int(truth.shape[1]),
           "tiles": tiles, "n_tiles_used": sum(1 for x in tiles if not x["skip"]), "dir": str(od)}
    if edit:
        P = load_yaml(CONFIG_DIR / "looms" / f"{job.profile_name}.yaml")
        wr = working_resolution(truth.shape[1], truth.shape[0], P, 1536)
        rect = cv2.cvtColor(cv2.imread(str(rect_png)), cv2.COLOR_BGR2RGB)
        ein = cv2.resize(rect, (wr["w"], wr["h"]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(od / "edit_input.png"), cv2.cvtColor(ein, cv2.COLOR_RGB2BGR))
        # flat target in the rectified space (4W x 4H), residual-aligned for real pairs
        flat = cv2.resize(truth_rgb[truth], (rect.shape[1], rect.shape[0]), interpolation=cv2.INTER_NEAREST)
        if (pair / "flat_aligned.png").exists():
            flat = cv2.cvtColor(cv2.imread(str(pair / "flat_aligned.png")), cv2.COLOR_BGR2RGB)
        flat_small = cv2.resize(flat, (wr["w"], wr["h"]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(od / "flat.png"), cv2.cvtColor(flat_small, cv2.COLOR_RGB2BGR))
        weight = np.ones((wr["h"], wr["w"]), np.float32)
        if (pair / "H_local.json").exists() and am is not None:
            grid = 4
            for by in range(grid):
                for bx in range(grid):
                    blk = am[by * am.shape[0] // grid:(by + 1) * am.shape[0] // grid, bx * am.shape[1] // grid:(bx + 1) * am.shape[1] // grid]
                    if blk.size and blk.mean() < 0.5:
                        weight[by * wr["h"] // grid:(by + 1) * wr["h"] // grid, bx * wr["w"] // grid:(bx + 1) * wr["w"] // grid] = 0.0
        np.save(od / "edit_weight.npy", weight)
        rec["edit"] = {"w": wr["w"], "h": wr["h"], "px_per_knot": wr["px_per_knot"], "tiled_train": wr["tiled"]}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="append", required=True)
    ap.add_argument("--splits", default=str(ROOT / "data" / "real" / "splits.json"))
    ap.add_argument("--out", default=str(ROOT / "train" / "data"))
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--edit", action="store_true")
    ap.add_argument("--profile", default="wilton_default")
    a = ap.parse_args(argv)
    splits = read_json(a.splits) if Path(a.splits).exists() else {"train": [], "val": [], "test": []}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = {"train": [], "val": []}
    for pd in a.pairs:
        for pair in sorted(p for p in Path(pd).iterdir() if (p / "truth.tiff").exists()):
            if pair.name in splits.get("test", []):
                continue
            job = run_to_structure(pair, Path(a.runs), a.profile)
            if job is None:
                print("skip (pipeline stop)", pair.name)
                continue
            rec = prepare_pair(pair, job, out, edit=a.edit)
            if rec is None:
                print("skip (layout_changed)", pair.name)
                continue
            split = "val" if pair.name in splits.get("val", []) else "train"
            recs[split].append(rec)
            print(split, pair.name, rec["design_class"], f"tiles {rec['n_tiles_used']}/{len(rec['tiles'])}")
    for split, rs in recs.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    print(f"train: {len(recs['train'])} val: {len(recs['val'])} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
