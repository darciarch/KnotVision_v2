#!/usr/bin/env python
"""Design -> photo render chain (synthetic pairs).

    design render -> pile texture (grid-locked) -> yarn deviation -> fibre blur (blur_knots)
    -> pile-direction shading -> illumination -> perspective (+15% TPS non-planar) -> background
    -> fringe (30%) -> occlusion -> camera -> JPEG (q 75-95)

flat.png is the truth design rendered flat in the rectified space (4W x 4H): no texture, no shading,
no perspective. It is pixel-aligned with what `rectify` produces because perspective is added later
in the chain (edit LoRA target).

Usage:  python data/synth/generate.py --designs data/synth/designs --out data/synth/pairs [--seed 0]
API:    render_photo(design, rng, dist, params=None) -> dict(photo (uint8 RGB), corners, meta)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knotvision import PX_PER_KNOT  # noqa: E402
from generate_designs import Design, load_dist, make_design, save_design  # noqa: E402


def render_flat(indexed: np.ndarray, palette_rgb: np.ndarray, px: int = PX_PER_KNOT) -> np.ndarray:
    H, W = indexed.shape
    rgb = palette_rgb[indexed]
    return cv2.resize(rgb, (W * px, H * px), interpolation=cv2.INTER_NEAREST)


def _sample_ppk(rng, dist) -> float:
    d = dist["photo_px_per_knot"]
    u = rng.uniform()
    if u < 0.1:
        return float(rng.uniform(max(0.6, d["p10"] * 0.7), d["p10"]))
    if u < 0.5:
        return float(rng.uniform(d["p10"], d["p50"]))
    if u < 0.9:
        return float(rng.uniform(d["p50"], d["p90"]))
    return float(rng.uniform(d["p90"], d["p90"] * 1.4))


def pile_texture(img: np.ndarray, px: int, rng, yarn_dev: float = 0.03) -> np.ndarray:
    """Per-knot brightness ripple locked to the grid + per-knot yarn colour deviation."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ripple = 1.0 + 0.06 * np.cos(2 * np.pi * (xx % px) / px) * np.cos(2 * np.pi * (yy % px) / px)
    H, W = h // px, w // px
    dev = rng.normal(0, yarn_dev, (H, W, 3)).astype(np.float32)
    dev = cv2.resize(dev, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.astype(np.float32) * ripple[..., None] * (1.0 + dev)
    return np.clip(out, 0, 255)


def pile_shading(img: np.ndarray, rng, a: float, b: float) -> tuple[np.ndarray, dict]:
    """m(x,y) = 1 + a * (d . p) + b * bulge(p); d = pile direction, p = normalised position."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([yy / max(1, h - 1) - 0.5, xx / max(1, w - 1) - 0.5], axis=-1)
    theta = rng.uniform(0, 2 * np.pi)
    d = np.array([np.sin(theta), np.cos(theta)])
    bulge = 1.0 - 4.0 * (p[..., 0] ** 2 + p[..., 1] ** 2)
    m = 1.0 + a * (p @ d) + b * bulge
    return img * m[..., None], {"a": a, "b": b, "theta": float(theta)}


def illumination_field(h: int, w: int, rng, strength: float) -> np.ndarray:
    """Low-frequency multiplicative field (2nd-degree polynomial in normalised coords)."""
    yy, xx = np.mgrid[0:h, 0:w]
    yn, xn = yy / max(1, h - 1) - 0.5, xx / max(1, w - 1) - 0.5
    c = rng.uniform(-1, 1, 5)
    f = c[0] * xn + c[1] * yn + c[2] * xn * yn + c[3] * xn ** 2 + c[4] * yn ** 2
    f = f / (np.abs(f).max() + 1e-9) * strength
    return np.exp(f)


def _tps_displace(corners_dst: np.ndarray, rng, amount_frac: float, out_w: int, out_h: int, img: np.ndarray):
    """Out-of-plane bend: TPS with the 4 corners fixed and interior points displaced."""
    src = list(corners_dst.astype(np.float32))
    dst = list(corners_dst.astype(np.float32))
    cx, cy = corners_dst.mean(0)
    span = min(np.ptp(corners_dst[:, 0]), np.ptp(corners_dst[:, 1]))
    for fy in (0.3, 0.5, 0.7):
        for fx in (0.3, 0.5, 0.7):
            p = corners_dst[0] + fx * (corners_dst[1] - corners_dst[0]) + fy * (corners_dst[3] - corners_dst[0])
            src.append(p.astype(np.float32))
            dst.append((p + rng.uniform(-1, 1, 2) * amount_frac * span).astype(np.float32))
    tps = cv2.createThinPlateSplineShapeTransformer()
    s = np.array(src).reshape(1, -1, 2)
    d = np.array(dst).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src))]
    tps.estimateTransformation(d, s, matches)
    return tps.warpImage(img)


def render_photo(design: Design, rng: np.random.RandomState, dist: dict, params: dict | None = None) -> dict:
    """Returns dict(photo uint8 RGB, corners [[x,y]x4] (tl,tr,br,bl), meta)."""
    P = dict(params or {})
    idx, pal = design.indexed, design.palette_rgb
    H, W = idx.shape
    px = PX_PER_KNOT
    img = render_flat(idx, pal, px).astype(np.float32)
    img = pile_texture(img, px, rng, yarn_dev=P.get("yarn_dev", 0.03))
    blur_knots = float(P.get("blur_knots", rng.uniform(0.3, 3.5)))
    sigma = blur_knots * px / 2.355
    img = cv2.GaussianBlur(img, (0, 0), max(0.3, sigma))
    a = float(P.get("shade_a", rng.uniform(0.05, 0.25)))
    b = float(P.get("shade_b", rng.uniform(0.0, 0.1)))
    img, shade = pile_shading(img, rng, a, b)
    illum_strength = float(P.get("illum", rng.uniform(0.05, 0.35)))
    field = illumination_field(img.shape[0], img.shape[1], rng, illum_strength)
    img = np.clip(img * field[..., None], 0, 255)
    # camera resolution
    ppk = float(P.get("px_per_knot", _sample_ppk(rng, dist)))
    pw, ph = max(16, int(round(W * ppk))), max(16, int(round(H * ppk)))
    img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    # fringe (short sides)
    fringe = bool(P.get("fringe", rng.uniform() < 0.3))
    fringe_px = 0
    if fringe:
        fringe_px = int(max(4, round(rng.uniform(0.03, 0.08) * min(pw, ph))))
        strip = np.full((fringe_px, pw, 3), 215, np.float32)
        strip *= rng.uniform(0.6, 1.0, (fringe_px, pw, 1))
        # vertical thread streaks
        for x in range(0, pw, max(1, int(rng.randint(2, 4)))):
            strip[:, x] *= rng.uniform(0.5, 0.9)
        img = np.vstack([strip, img, strip[::-1]])
    # perspective onto a background
    margin = int(0.15 * max(pw, ph))
    out_w, out_h = pw + 2 * margin, img.shape[0] + 2 * margin
    bg = np.full((out_h, out_w, 3), 0, np.float32)
    bg[:] = np.array(rng.uniform(40, 200, 3), np.float32)
    bg += rng.normal(0, 6, bg.shape)
    tilt = float(P.get("tilt", rng.uniform(0.0, 0.12)))
    jit = np.array([[rng.uniform(-tilt, tilt) * pw, rng.uniform(-tilt, tilt) * ph] for _ in range(4)])
    src = np.float32([[0, 0], [pw, 0], [pw, img.shape[0]], [0, img.shape[0]]])
    dst = np.float32([[margin, margin], [margin + pw, margin], [margin + pw, margin + img.shape[0]], [margin, margin + img.shape[0]]]) + jit.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones(img.shape[:2], np.float32), M, (out_w, out_h), flags=cv2.INTER_NEAREST)
    photo = np.where(mask[..., None] > 0, warped, bg)
    # pile-edge corners (exclude fringe) in photo coordinates
    pile_src = np.float32([[0, fringe_px], [pw, fringe_px], [pw, fringe_px + ph], [0, fringe_px + ph]])
    corners = cv2.perspectiveTransform(pile_src.reshape(1, -1, 2), M)[0]
    non_planar = bool(P.get("non_planar", rng.uniform() < 0.15))
    if non_planar:
        photo = _tps_displace(corners, rng, 0.03, out_w, out_h, photo)
    # occlusion
    occl = bool(P.get("occlusion", rng.uniform() < 0.1))
    if occl:
        oy, ox = int(rng.randint(0, out_h - out_h // 6)), int(rng.randint(0, out_w - out_w // 6))
        photo[oy:oy + out_h // 8, ox:ox + out_w // 8] = rng.uniform(20, 240, 3)
    # camera noise + JPEG
    noise_sigma = float(P.get("noise_sigma", rng.uniform(1.0, 6.0)))
    photo = np.clip(photo + rng.normal(0, noise_sigma, photo.shape), 0, 255).astype(np.uint8)
    q = int(P.get("jpeg_q", rng.randint(75, 96)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    photo = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    meta = {"px_per_knot": ppk, "blur_knots": blur_knots, "shade": shade, "illum_strength": illum_strength,
            "illum_range_ratio": float(field.max() / field.min()), "fringe": fringe, "fringe_px": fringe_px,
            "non_planar": non_planar, "occlusion": occl, "noise_sigma": noise_sigma, "jpeg_q": q,
            "photo_long_px": int(max(photo.shape[:2])), "tilt": tilt}
    # low-res illumination field for illum_err (in knot grid)
    illum_small = cv2.resize(field.astype(np.float32), (min(W, 64), min(H, 64)), interpolation=cv2.INTER_AREA)
    return {"photo": photo, "corners": corners.tolist(), "meta": meta, "illum": illum_small}


def save_pair(design: Design, r: dict, out: Path):
    save_design(design, out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "photo.jpg"), cv2.cvtColor(r["photo"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / "flat.png"), cv2.cvtColor(render_flat(design.indexed, design.palette_rgb), cv2.COLOR_RGB2BGR))
    (out / "corners_gt.json").write_text(json.dumps({"corners": r["corners"]}))
    np.save(out / "illum_gt.npy", r["illum"])
    meta = {**design.meta, "photo": r["meta"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", help="dir of saved designs (truth.tiff + meta.json); if omitted, designs are generated")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth" / "pairs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist", default=None)
    a = ap.parse_args(argv)
    dist = load_dist(Path(a.dist)) if a.dist else load_dist()
    rng = np.random.RandomState(a.seed + 7919)
    if a.designs:
        from knotvision.export import read_indexed_tiff
        dirs = sorted(p for p in Path(a.designs).iterdir() if (p / "truth.tiff").exists())
        for p in dirs[:a.n]:
            idx, pal = read_indexed_tiff(p / "truth.tiff")
            meta = json.loads((p / "meta.json").read_text())
            layout = json.loads((p / "layout_gt.json").read_text()) if (p / "layout_gt.json").exists() else {}
            d = Design(idx, pal, meta, layout, np.load(p / "gradient_mask.npy") if (p / "gradient_mask.npy").exists() else np.zeros(idx.shape, bool),
                       np.load(p / "noise_mask.npy") if (p / "noise_mask.npy").exists() else np.zeros(idx.shape, bool))
            r = render_photo(d, rng, dist)
            save_pair(d, r, Path(a.out) / p.name)
            print(p.name, r["meta"]["px_per_knot"], r["meta"]["blur_knots"])
    else:
        for i in range(a.n):
            d = make_design(a.seed + i, dist)
            r = render_photo(d, rng, dist)
            save_pair(d, r, Path(a.out) / f"proc_{a.seed + i:05d}")
            print(f"proc_{a.seed + i:05d}", d.meta["template"], round(r["meta"]["px_per_knot"], 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
