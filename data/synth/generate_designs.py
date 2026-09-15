#!/usr/bin/env python
"""Procedural designs with freely combined class features (no archive needed).

Every design starts from a base template (geometric | gradient | distressed | minimal, by quota;
`classic` is a symmetric medallion + border proxy for tests, not a class_acc target) and receives,
with independent probabilities: border (with 1-D period + corner blocks), lattice repeat, gradient
regions (1-3, dithered with flat colours the way a designer does), simplex noise (noise_frac in
[0.05, 0.25]), symmetry, thin 1-knot lines (thin_line_frac in [0, 0.03]).

    pure  : no feature beyond the base template  -> class_acc is measured on these
    mixed : gradient_frac >= 0.25 and noise_frac >= 0.08 -> class_mixed recall is measured on these

Usage:  python data/synth/generate_designs.py --n 100 --out data/synth/designs --seed 0
API:    make_design(seed, dist=None, template=None, W=None, H=None, features=None) -> Design
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from knotvision.color import delta_e2000, rgb_to_lab  # noqa: E402
from knotvision.metrics import connected_components, is_thin  # noqa: E402

TEMPLATES = ("geometric", "gradient", "distressed", "minimal", "classic")
QUOTA = {"geometric": 0.3, "gradient": 0.25, "distressed": 0.2, "minimal": 0.15, "classic": 0.1}
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64)


@dataclass
class Design:
    indexed: np.ndarray            # (H, W) int32
    palette_rgb: np.ndarray        # (K, 3) uint8
    meta: dict
    layout_gt: dict
    gradient_mask: np.ndarray      # (H, W) bool
    noise_mask: np.ndarray         # (H, W) bool
    seed: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def W(self):
        return self.indexed.shape[1]

    @property
    def H(self):
        return self.indexed.shape[0]


# ----------------------------------------------------------------------------- helpers

def load_dist(path: Path | None = None) -> dict:
    p = path or (ROOT / "data" / "synth" / "dist.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


def _sample_range(rng, d: dict) -> float:
    return float(rng.uniform(d["min"], d["max"]))


def make_palette(rng: np.random.RandomState, n: int, min_de: float = 14.0, tries: int = 400) -> np.ndarray:
    """n distinct sRGB colours with pairwise deltaE2000 >= min_de (relaxed if it cannot be met)."""
    cols = []
    labs = []
    de = min_de
    while len(cols) < n:
        for _ in range(tries):
            c = rng.randint(20, 236, size=3)
            lab = rgb_to_lab(c / 255.0)
            if all(float(delta_e2000(lab, l2)) >= de for l2 in labs):
                cols.append(c)
                labs.append(lab)
                break
        else:
            de *= 0.8
    return np.array(cols, dtype=np.uint8)


def _snoise(rng, H, W, scale: float) -> np.ndarray:
    """Simplex noise field in [-1, 1] via the `noise` package (fallback: filtered gaussian)."""
    try:
        from noise import snoise2
        ox, oy = rng.uniform(0, 1000, 2)
        ys, xs = np.mgrid[0:H, 0:W]
        f = np.vectorize(lambda y, x: snoise2((x + ox) / scale, (y + oy) / scale, octaves=2))
        return f(ys, xs)
    except Exception:  # pragma: no cover
        import cv2
        g = rng.normal(0, 1, (H, W)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), scale / 3)
        return g / (np.abs(g).max() + 1e-9)


# ----------------------------------------------------------------------------- feature painters

def paint_border(idx, rng, ground, border_col, motif_col, thickness, period, corner_col=None):
    H, W = idx.shape
    t = thickness
    idx[:t, :] = border_col
    idx[-t:, :] = border_col
    idx[:, :t] = border_col
    idx[:, -t:] = border_col
    # periodic motif along the strips (excluding corner blocks): a diamond every `period` knots
    r = max(1, t // 3)
    cy = t // 2
    for x0 in range(t + period // 2, W - t, period):
        for dy in range(-r, r + 1):
            span = r - abs(dy)
            idx[cy + dy, max(t, x0 - span):min(W - t, x0 + span + 1)] = motif_col
            idx[H - 1 - cy - dy, max(t, x0 - span):min(W - t, x0 + span + 1)] = motif_col
    for y0 in range(t + period // 2, H - t, period):
        for dx in range(-r, r + 1):
            span = r - abs(dx)
            idx[max(t, y0 - span):min(H - t, y0 + span + 1), cy + dx] = motif_col
            idx[max(t, y0 - span):min(H - t, y0 + span + 1), W - 1 - cy - dx] = motif_col
    # corner blocks: a distinct square motif that breaks the period
    cc = motif_col if corner_col is None else corner_col
    q = max(1, t // 4)
    for (ys, xs) in ((slice(q, t - q), slice(q, t - q)), (slice(q, t - q), slice(W - t + q, W - q)),
                     (slice(H - t + q, H - q), slice(q, t - q)), (slice(H - t + q, H - q), slice(W - t + q, W - q))):
        idx[ys, xs] = cc
    return {"thickness_knots": t, "period_knots": period, "corner_block_knots": t}


def paint_lattice(idx, rng, region, v1, v2, motif_col, ground_col, shape="diamond", size=None):
    """Repeat a small motif on the lattice k*v1 + l*v2 inside `region` (bool mask)."""
    H, W = idx.shape
    size = size or max(2, int(0.3 * min(np.hypot(*v1), np.hypot(*v2))))
    n = int(2 * max(H, W) / max(1, min(np.hypot(*v1), np.hypot(*v2)))) + 2
    ys0, xs0 = np.nonzero(region)
    oy, ox = int(ys0.min()) + size, int(xs0.min()) + size
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cy = oy + i * v1[0] + j * v2[0]
            cx = ox + i * v1[1] + j * v2[1]
            if not (0 <= cy < H and 0 <= cx < W) or not region[cy, cx]:
                continue
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    y, x = cy + dy, cx + dx
                    if not (0 <= y < H and 0 <= x < W) or not region[y, x]:
                        continue
                    if shape == "diamond" and abs(dy) + abs(dx) <= size:
                        idx[y, x] = motif_col
                    elif shape == "square" and max(abs(dy), abs(dx)) <= size:
                        idx[y, x] = motif_col
                    elif shape == "ring" and size - 1 <= abs(dy) + abs(dx) <= size:
                        idx[y, x] = motif_col
    return {"v1": [int(v1[0]), int(v1[1])], "v2": [int(v2[0]), int(v2[1])], "motif": shape, "size": size}


def paint_gradient(idx, rng, region, ia, ib, direction, levels=4):
    """Designer-style dither between indices ia and ib along `direction` inside `region`."""
    H, W = idx.shape
    ys, xs = np.nonzero(region)
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    proj = ys * d[0] + xs * d[1]
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    tq = np.round(t * levels) / levels
    thr = (BAYER4[ys % 4, xs % 4] + 0.5) / 16.0
    idx[ys, xs] = np.where(tq > thr, ib, ia)
    return {"ia": int(ia), "ib": int(ib), "direction": d.tolist(), "area_knots": int(len(ys))}


def paint_noise(idx, rng, region, worn_col, frac_target, cc_max=6, single_knot=False):
    """Multi-knot speckles (component areas mostly in [2, cc_max]) covering ~frac_target of the region."""
    H, W = idx.shape
    if region.sum() < 16:
        return np.zeros((H, W), dtype=bool)
    field_ = _snoise(rng, H, W, scale=2.2)
    import scipy.ndimage as ndi
    vals = field_[region]
    keep = rng.uniform(0, 1, (H, W)) < 0.45
    q = 100 * (1 - frac_target)
    m = np.zeros((H, W), dtype=bool)
    for _ in range(8):
        thr = np.percentile(vals, q)
        m = (field_ > thr) & region
        # break large blobs with a random keep mask so that component areas stay in [2, cc_max]
        lab, n = ndi.label(m)
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        big = np.isin(lab, np.nonzero(sizes > cc_max)[0] + 1)
        m = (m & ~big) | (big & keep)
        if not single_knot:
            lab, n = ndi.label(m)
            sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
            m &= ~np.isin(lab, np.nonzero(sizes < 2)[0] + 1)
        cov = m.sum() / max(1, region.sum())
        if cov >= frac_target * 0.9 or q <= 5:
            break
        q = max(5, q - 100 * (frac_target - cov) * 1.5)
    idx[m] = worn_col
    return m


def paint_thin_lines(idx, rng, region, line_col, frac_target, min_len=8):
    """1-knot-wide straight lines (h / v / diagonal) inside `region` until ~frac_target of cells."""
    H, W = idx.shape
    target = int(frac_target * idx.size)
    painted = np.zeros_like(region)
    tries = 0
    while painted.sum() < target and tries < 200:
        tries += 1
        kind = rng.choice(["h", "v", "d"])
        L = int(rng.randint(min_len, max(min_len + 1, min(H, W) // 3)))
        y0, x0 = int(rng.randint(0, H)), int(rng.randint(0, W))
        pts = []
        for s in range(L):
            y, x = (y0, x0 + s) if kind == "h" else (y0 + s, x0) if kind == "v" else (y0 + s, x0 + s)
            if not (0 <= y < H and 0 <= x < W) or not region[y, x]:
                break
            pts.append((y, x))
        if len(pts) < min_len:
            continue
        # keep lines 2 knots apart so that they stay 1 knot wide
        ok = True
        for y, x in pts:
            if painted[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                ok = False
                break
        if not ok:
            continue
        for y, x in pts:
            idx[y, x] = line_col
            painted[y, x] = True
    return painted


def paint_medallion(idx, rng, cy, cx, ry, rx, col, inner_col=None):
    H, W = idx.shape
    ys, xs = np.mgrid[0:H, 0:W]
    e = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
    m = e <= 1.0
    idx[m] = col
    if inner_col is not None:
        idx[e <= 0.45] = inner_col
    return {"bbox": [int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)]}


def make_symmetric(idx, kind):
    H, W = idx.shape
    if kind in ("h", "hv"):
        idx[:, W - W // 2:] = idx[:, :W // 2][:, ::-1]
    if kind in ("v", "hv"):
        idx[H - H // 2:, :] = idx[:H // 2, :][::-1, :]
    return idx


# ----------------------------------------------------------------------------- main generator

def make_design(seed: int, dist: dict | None = None, template: str | None = None, W: int | None = None,
                H: int | None = None, features: dict | None = None, n_colors: int | None = None) -> Design:
    """features: dict of forced feature flags {border, repeat, gradient, noise, symmetry, thin_lines}
    (True/False); missing keys are sampled."""
    rng = np.random.RandomState(seed)
    dist = dist or load_dist()
    if template is None:
        names, probs = zip(*QUOTA.items())
        template = str(rng.choice(names, p=np.array(probs) / sum(probs)))
    width_cm = _sample_range(rng, dist["width_cm"])
    height_cm = _sample_range(rng, dist["height_cm"])
    reed = _sample_range(rng, dist["reed_per_m"])
    wefts = _sample_range(rng, dist["wefts_per_m"])
    if W is None:
        W = int(round(width_cm / 100 * reed))
    else:
        width_cm = W / reed * 100
    if H is None:
        H = int(round(height_cm / 100 * wefts))
    else:
        height_cm = H / wefts * 100
    W, H = max(32, W), max(32, H)
    mc = int(rng.choice(dist.get("machine_colors", [5, 6, 8, 12])))
    n = n_colors or int(min(mc, rng.randint(dist["n_colors"]["min"], dist["n_colors"]["max"] + 1)))
    if template == "minimal":
        n = min(n, 3)
    n = max(2, n)
    pal = make_palette(rng, n)
    idx = np.zeros((H, W), dtype=np.int32)
    ground = 0
    feats = dict(features or {})

    def want(name, p):
        if name in feats:
            return bool(feats[name])
        return bool(rng.uniform() < p)

    layout = {"template": template, "border": None, "repeat": None, "gradient_regions": [], "medallion": None,
              "symmetry": "none", "corner_blocks": False}
    gmask = np.zeros((H, W), dtype=bool)
    nmask = np.zeros((H, W), dtype=bool)
    thin_painted = np.zeros((H, W), dtype=bool)
    cols = list(range(1, n))
    rng.shuffle(cols)
    col_iter = iter(cols * 6)

    def nxt():
        return next(col_iter)

    inner = np.ones((H, W), dtype=bool)
    # --- base template
    has_border = want("border", {"geometric": 0.6, "gradient": 0.3, "distressed": 0.4, "minimal": 0.2, "classic": 1.0}[template])
    if has_border and n >= 3 and min(H, W) >= 48:
        t = int(np.clip(rng.uniform(0.05, 0.14) * min(H, W), 4, 40))
        period = int(np.clip(rng.randint(8, 24), 8, max(8, (min(H, W) - 2 * t) // 3)))
        layout["border"] = paint_border(idx, rng, ground, nxt(), nxt(), t, period)
        layout["corner_blocks"] = True
        inner[:] = False
        inner[t:H - t, t:W - t] = True
    has_repeat = want("repeat", {"geometric": 0.8, "gradient": 0.2, "distressed": 0.3, "minimal": 0.0, "classic": 0.2}[template])
    if has_repeat:
        p1 = int(rng.randint(10, max(11, min(H, W) // 4)))
        v1 = (0, p1)
        v2 = (p1, 0) if rng.uniform() < 0.6 else (p1, p1 // 2)
        layout["repeat"] = paint_lattice(idx, rng, inner, v1, v2, nxt(), ground,
                                         shape=str(rng.choice(["diamond", "square", "ring"])))
    if template == "geometric" and not has_repeat:
        # straight stripes / chevrons: lines locked to allowed slopes
        c = nxt()
        step = int(rng.randint(8, 16))
        for k in range(0, W + H, step):
            for s in range(min(H, W)):
                y, x = s, k - s
                if 0 <= x < W and 0 <= y < H and inner[y, x]:
                    idx[y, x] = c
                    if x + 1 < W and inner[y, x + 1]:
                        idx[y, x + 1] = c
    if template == "classic" or (template != "minimal" and want("medallion", 0.15)):
        ry, rx = int(0.18 * H), int(0.22 * W)
        layout["medallion"] = paint_medallion(idx, rng, H // 2, W // 2, ry, rx, nxt(), nxt() if n >= 4 else None)
    if template == "minimal":
        # two or three big flat shapes, few edges
        c = nxt()
        y0, y1 = int(H * rng.uniform(0.2, 0.4)), int(H * rng.uniform(0.6, 0.85))
        x0, x1 = int(W * rng.uniform(0.2, 0.4)), int(W * rng.uniform(0.6, 0.85))
        idx[y0:y1, x0:x1] = c
    # --- gradient regions
    n_grad = 0
    if template == "gradient":
        n_grad = int(rng.randint(1, 4))
    elif want("gradient", 0.15):
        n_grad = 1
    for g in range(n_grad):
        ia, ib = (ground, nxt()) if g == 0 else (nxt(), nxt())
        if ia == ib:
            continue
        if template == "gradient" and g == 0:
            region = inner.copy()
        else:
            gh, gw = int(H * rng.uniform(0.2, 0.5)), int(W * rng.uniform(0.2, 0.5))
            gy, gx = int(rng.randint(0, max(1, H - gh))), int(rng.randint(0, max(1, W - gw)))
            region = np.zeros((H, W), dtype=bool)
            region[gy:gy + gh, gx:gx + gw] = True
            region &= inner
        if region.sum() < 64:
            continue
        direction = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))]
        if abs(direction[0]) + abs(direction[1]) < 0.3:
            direction = [0.0, 1.0]
        info = paint_gradient(idx, rng, region, ia, ib, direction, levels=int(rng.choice([8, 12, 16])))
        ys, xs = np.nonzero(region)
        info["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        layout["gradient_regions"].append(info)
        gmask |= region
    # --- noise
    noise_target = 0.0
    if template == "distressed":
        noise_target = float(rng.uniform(0.10, 0.25))
    elif want("noise", 0.15):
        noise_target = float(rng.uniform(0.05, 0.15))
    if noise_target > 0:
        region = inner & ~gmask
        worn = nxt()
        nmask = paint_noise(idx, rng, region, worn, noise_target)
    # --- thin lines
    thin_target = 0.0
    if want("thin_lines", {"geometric": 0.4, "classic": 0.4, "gradient": 0.1, "distressed": 0.1, "minimal": 0.05}[template]):
        thin_target = float(rng.uniform(0.005, 0.03))
        thin_painted = paint_thin_lines(idx, rng, inner & ~gmask & ~nmask, nxt(), thin_target)
    # --- symmetry
    sym = "none"
    if template == "classic":
        sym = "hv"
    elif want("symmetry", {"geometric": 0.3, "gradient": 0.1, "distressed": 0.1, "minimal": 0.2}[template]):
        sym = str(rng.choice(["h", "v", "hv"]))
    if sym != "none":
        make_symmetric(idx, sym)
        gmask = make_symmetric(gmask.astype(np.int32), sym).astype(bool)
        nmask = make_symmetric(nmask.astype(np.int32), sym).astype(bool)
        thin_painted = make_symmetric(thin_painted.astype(np.int32), sym).astype(bool)
    layout["symmetry"] = sym
    # --- merge indices with identical RGB (none by construction) and drop unused entries
    used = np.unique(idx)
    remap = {int(u): i for i, u in enumerate(used)}
    idx = np.vectorize(remap.get)(idx).astype(np.int32)
    pal = pal[used]
    for r in layout["gradient_regions"]:
        r["ia"], r["ib"] = remap[r["ia"]], remap[r["ib"]]

    gradient_frac = float(gmask.mean())
    noise_frac = float(nmask.mean())
    # thin_line_frac is measured with the single-home rule (metrics.is_thin) on the final design, so
    # that thin motif parts (rings, border diamonds) count exactly as the pipeline will count them
    thin_frac = 0.0
    for c in connected_components(idx):
        if c.area >= 6 and is_thin(c.full_mask(idx.shape), 6, 1):
            thin_frac += c.area
    thin_frac = float(thin_frac / idx.size)
    base_only = (not has_border or template == "classic") and not (has_repeat and template != "geometric") \
        and (n_grad == 0 or template == "gradient") and (noise_target == 0 or template == "distressed") \
        and thin_target == 0 and (sym == "none" or template == "classic")
    meta = {
        "seed": seed, "template": template, "design_class": template, "W": W, "H": H,
        "width_cm": round(width_cm, 1), "height_cm": round(height_cm, 1), "reed_per_m": round(reed, 1),
        "wefts_per_m": round(wefts, 1), "density": round(reed * wefts, 1), "machine_colors": mc,
        "n_colors": int(len(pal)), "gradient_frac": gradient_frac, "noise_frac": noise_frac,
        "thin_line_frac": thin_frac, "has_border": bool(layout["border"]), "has_repeat": bool(layout["repeat"]),
        "symmetry": sym, "pure": bool(base_only),
        "mixed": bool(gradient_frac >= 0.25 and noise_frac >= 0.08),
    }
    return Design(indexed=idx, palette_rgb=pal, meta=meta, layout_gt=layout, gradient_mask=gmask,
                  noise_mask=nmask, seed=seed)


def save_design(d: Design, out: Path):
    from knotvision.export import write_indexed_tiff
    out.mkdir(parents=True, exist_ok=True)
    write_indexed_tiff(out / "truth.tiff", d.indexed, d.palette_rgb, d.meta["reed_per_m"], d.meta["wefts_per_m"])
    np.save(out / "gradient_mask.npy", d.gradient_mask)
    np.save(out / "noise_mask.npy", d.noise_mask)
    (out / "meta.json").write_text(json.dumps(d.meta, indent=2))
    (out / "layout_gt.json").write_text(json.dumps(d.layout_gt, indent=2))
    with open(out / "palette.csv", "w") as f:
        f.write("index,R,G,B\n")
        for i, c in enumerate(d.palette_rgb):
            f.write(f"{i},{c[0]},{c[1]},{c[2]}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth" / "designs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist", default=None)
    ap.add_argument("--ids-file", help="write the generated ids here (e.g. eval/synth_modern_ids.txt)")
    a = ap.parse_args(argv)
    dist = load_dist(Path(a.dist)) if a.dist else load_dist()
    ids = []
    for i in range(a.n):
        d = make_design(a.seed + i, dist)
        did = f"proc_{a.seed + i:05d}"
        save_design(d, Path(a.out) / did)
        ids.append(did)
        print(did, d.meta["template"], d.W, d.H, "pure" if d.meta["pure"] else "mixed" if d.meta["mixed"] else "combo")
    if a.ids_file:
        Path(a.ids_file).write_text("\n".join(ids) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
