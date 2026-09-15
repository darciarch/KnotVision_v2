"""palette: chosen from the photo BEFORE the model. Upper bound machine_colors, lower bound 1.

1. rectified -> guided filter -> Lab -> long side 400 px -> k-means (k = 4 x machine_colors, fixed seed).
   Cluster weight = count ** area_weight_exp; gradient-mask pixels weigh 0; each gradient region's two
   Lab endpoints are mandatory candidates.
2. Adjacency matrix (shared boundary pixels at 400 px; gradient-mask boundaries excluded).
3. Selection: palette_override | yarn library (greedy, stop at gain_stop_select) |
   greedy cluster merge (cost = weighted deltaE increase + adjacency_penalty * shared/total; down to
   machine_colors then continue while cost < merge_cost_stop). Gradient endpoints are never merged.
4. Motif-colour protection: clusters with small coverage but >= motif_color_min_cc components or a
   largest component >= motif_min_knots are merged last.
5. palette.json, palette_proposed.json, palette_card.png
Output: palette.json, palette_proposed.json, palette_card.png
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

from .color import delta_e2000, hex_of, lab_to_rgb, rgb_to_lab
from .job import Job, dump_json, read_json
from .metrics import guided


def _resize_long(img, long_px):
    h, w = img.shape[:2]
    s = long_px / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- clustering

def cluster_photo(rect_rgb: np.ndarray, gmask_grid: Optional[np.ndarray], machine_colors: int,
                  area_weight_exp: float, gradient_regions: list, seed: int = 0) -> dict:
    g = guided(rect_rgb, radius=8, eps=0.01)
    small = _resize_long(g, 400)
    h, w = small.shape[:2]
    lab = rgb_to_lab(small)
    if gmask_grid is not None and gmask_grid.any():
        gm = cv2.resize(gmask_grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        gm = np.zeros((h, w), dtype=bool)
    k = min(4 * machine_colors, max(2, (~gm).sum() // 50))
    flat = lab.reshape(-1, 3)
    valid = ~gm.reshape(-1)
    X = flat[valid] if valid.any() else flat
    km = KMeans(n_clusters=min(k, len(X)), random_state=seed, n_init=4).fit(X)
    labels_img = np.full(h * w, -1, dtype=np.int32)
    labels_img[valid] = km.labels_
    labels_img = labels_img.reshape(h, w)
    centers = km.cluster_centers_.copy()
    counts = np.bincount(km.labels_, minlength=len(centers)).astype(np.float64)
    weights = counts ** area_weight_exp
    forced = np.zeros(len(centers), dtype=bool)
    endpoint_of = [None] * len(centers)
    # mandatory gradient endpoints as extra clusters
    for r in gradient_regions:
        for end in ("lab_lo", "lab_hi"):
            centers = np.vstack([centers, np.array(r[end], dtype=np.float64)])
            weights = np.append(weights, weights.max() if len(weights) else 1.0)
            counts = np.append(counts, 0.0)
            forced = np.append(forced, True)
            endpoint_of.append((r["id"], end))
    # adjacency (400 px): shared boundary pixels between clusters, excluding gradient-mask boundaries
    n = len(centers)
    adj = np.zeros((n, n), dtype=np.float64)
    a, b = labels_img[:, :-1], labels_img[:, 1:]
    m = (a != b) & (a >= 0) & (b >= 0) & ~gm[:, :-1] & ~gm[:, 1:]
    np.add.at(adj, (a[m], b[m]), 1)
    a, b = labels_img[:-1, :], labels_img[1:, :]
    m = (a != b) & (a >= 0) & (b >= 0) & ~gm[:-1, :] & ~gm[1:, :]
    np.add.at(adj, (a[m], b[m]), 1)
    adj = adj + adj.T
    return {"centers": centers, "weights": weights, "counts": counts, "forced": forced,
            "endpoint_of": endpoint_of, "adj": adj, "labels_img": labels_img, "lab_small": lab, "gm_small": gm}


# ----------------------------------------------------------------------------- motif-colour protection

def motif_protected(cl: dict, motif_color_min_cc: int, motif_min_knots: int, W: int, H: int) -> np.ndarray:
    """Clusters whose coverage is small but that form >= motif_color_min_cc components or one
    component >= motif_min_knots (measured on the 400 px label image, scaled to knots)."""
    labels_img = cl["labels_img"]
    h, w = labels_img.shape
    knots_per_px = (W * H) / float(h * w)
    n = len(cl["centers"])
    prot = np.zeros(n, dtype=bool)
    total = labels_img.size
    for i in range(n):
        m = labels_img == i
        cov = m.sum() / total
        if cov == 0 or cov > 0.05:
            continue
        lab, ncc = ndi.label(m)
        if ncc >= motif_color_min_cc:
            prot[i] = True
            continue
        if ncc:
            largest = ndi.sum(m, lab, index=np.arange(1, ncc + 1)).max() * knots_per_px
            if largest >= motif_min_knots:
                prot[i] = True
    return prot


# ----------------------------------------------------------------------------- selection paths

def merge_clusters(cl: dict, machine_colors: int, adjacency_penalty: float, merge_cost_stop: float,
                   protected: np.ndarray) -> list[list[int]]:
    """Greedy merge; returns groups of original cluster ids. Cost = weighted deltaE increase +
    adjacency_penalty * shared / total boundary. Forced (gradient endpoints) are never merged."""
    centers, weights, adj, forced = cl["centers"], cl["weights"], cl["adj"], cl["forced"]
    groups = [[i] for i in range(len(centers))]
    gcent = [centers[i].copy() for i in range(len(centers))]
    gw = [float(weights[i]) for i in range(len(centers))]
    gforced = [bool(forced[i]) for i in range(len(centers))]
    gprot = [bool(protected[i]) for i in range(len(centers))]
    total_adj = adj.sum() / 2 + 1e-9

    def group_adj(ga, gb):
        return sum(adj[i, j] for i in ga for j in gb)

    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if gforced[a] and gforced[b]:
                    continue
                if gforced[a] or gforced[b]:
                    # a forced endpoint may absorb a free cluster only if very close
                    de = float(delta_e2000(gcent[a], gcent[b]))
                    if de > 2.0:
                        continue
                de = float(delta_e2000(gcent[a], gcent[b]))
                wa, wb = gw[a], gw[b]
                cost = de * min(wa, wb) / (wa + wb) + adjacency_penalty * group_adj(groups[a], groups[b]) / total_adj
                if gprot[a] or gprot[b]:
                    cost += 1e3  # protected clusters are merged last
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        if best is None:
            break
        cost, a, b = best
        if len(groups) <= machine_colors and cost >= merge_cost_stop:
            break
        # merge b into a (weighted centre; forced centre wins)
        if gforced[b] and not gforced[a]:
            a, b = b, a
        if not gforced[a]:
            gcent[a] = (gcent[a] * gw[a] + gcent[b] * gw[b]) / (gw[a] + gw[b])
        gw[a] += gw[b]
        groups[a] = groups[a] + groups[b]
        gprot[a] = gprot[a] and gprot[b]
        del groups[b], gcent[b], gw[b], gforced[b], gprot[b]
    return groups


def select_from_library(cl: dict, yarn_lab: np.ndarray, yarns: list, machine_colors: int,
                        gain_stop_select: float) -> list[int]:
    """Greedy: at each step add the yarn that most reduces weighted total deltaE2000; stop when the
    gain < gain_stop_select. Gradient endpoints' nearest yarns are mandatory."""
    centers, weights, forced = cl["centers"], cl["weights"], cl["forced"]
    D = delta_e2000(centers[:, None, :], yarn_lab[None, :, :])  # (n_clusters, n_yarns)
    chosen = []
    for i in np.nonzero(forced)[0]:
        j = int(np.argmin(D[i]))
        if j not in chosen:
            chosen.append(j)
    cur = np.full(len(centers), np.inf) if not chosen else D[:, chosen].min(axis=1)
    while len(chosen) < machine_colors:
        best_gain, best_j = -1.0, None
        cur_cost = np.where(np.isinf(cur), 1e3, cur)
        for j in range(len(yarn_lab)):
            if j in chosen:
                continue
            new = np.minimum(cur_cost, D[:, j])
            gain = float(((cur_cost - new) * weights).sum() / (weights.sum() + 1e-9))
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None or (chosen and best_gain < gain_stop_select):
            break
        chosen.append(best_j)
        cur = D[:, chosen].min(axis=1)
    return chosen


# ----------------------------------------------------------------------------- palette card

def palette_card(entries: list, cell: int = 128) -> np.ndarray:
    n = len(entries)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    pad = 8
    label_h = 28
    Wc = cols * (cell + pad) + pad
    Hc = rows * (cell + label_h + pad) + pad
    img = np.full((Hc, Wc, 3), 255, np.uint8)
    for i, e in enumerate(entries):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (cell + pad), pad + r * (cell + label_h + pad)
        img[y0:y0 + cell, x0:x0 + cell] = np.array(e["rgb"], np.uint8)
        cv2.putText(img, hex_of(e["rgb"]), (x0, y0 + cell + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("palette")
    rd, st = job.stage_dir("rectify"), job.stage_dir("structure")
    rect_png = rd / "rectified.png"
    inputs = [rect_png, st / "gradient_mask.npy", st / "gradient_regions.json"]
    idir = job.stage_dir("intake")
    if (idir / "yarn_lab.npy").exists():
        inputs.append(idir / "yarn_lab.npy")
    key = job.cache_key("palette", inputs, extra={"override": job.palette_override}, exclude_prefix="edit_")
    if job.cached(sd, key) and (sd / "palette.json").exists():
        job.timings["palette"] = 0.0
        return job.state

    P = job.params
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    rect = cv2.cvtColor(cv2.imread(str(rect_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gmask = np.load(st / "gradient_mask.npy")
    gregions = read_json(st / "gradient_regions.json")
    mc = int(job.machine_colors)

    cl = cluster_photo(rect, gmask, mc, float(P.get("area_weight_exp", 0.5)), gregions)
    entries = []
    if job.palette_override:
        for i, e in enumerate(job.palette_override):
            rgb = [int(x) for x in e["rgb"]]
            entries.append({"index": i, "rgb": rgb, "lab": rgb_to_lab(np.array(rgb) / 255.0).tolist(),
                            "yarn_code": e.get("yarn_code", ""), "name": e.get("name", ""),
                            "gradient_endpoint": None})
        source = "override"
    elif (idir / "yarn_lab.npy").exists():
        yarn_lab = np.load(idir / "yarn_lab.npy")
        yarns = read_json(idir / "yarns.json")
        chosen = select_from_library(cl, yarn_lab, yarns, mc, float(P.get("gain_stop_select", 0.3)))
        for i, j in enumerate(chosen):
            ep = None
            for ci in np.nonzero(cl["forced"])[0]:
                if int(np.argmin(delta_e2000(cl["centers"][ci][None], yarn_lab))) == j:
                    ep = cl["endpoint_of"][ci]
            entries.append({"index": i, "rgb": yarns[j]["rgb"], "lab": yarn_lab[j].tolist(),
                            "yarn_code": yarns[j]["yarn_code"], "name": yarns[j]["name"], "gradient_endpoint": ep})
        source = "library"
    else:
        prot = motif_protected(cl, int(P.get("motif_color_min_cc", 5)), int(P.get("motif_min_knots", 10)), W, H)
        groups = merge_clusters(cl, mc, float(P.get("adjacency_penalty", 2.0)), float(P.get("merge_cost_stop", 0.5)), prot)
        # order: coverage descending, lowest original id on ties
        def cov(g):
            return float(sum(cl["counts"][i] for i in g))
        groups = sorted(groups, key=lambda g: (-cov(g), min(g)))
        for i, g in enumerate(groups):
            forced_ids = [j for j in g if cl["forced"][j]]
            if forced_ids:
                lab = cl["centers"][forced_ids[0]]
                ep = cl["endpoint_of"][forced_ids[0]]
            else:
                w = np.array([cl["weights"][j] for j in g])
                lab = (cl["centers"][g] * w[:, None]).sum(0) / w.sum()
                ep = None
            rgb = lab_to_rgb(lab).tolist()
            entries.append({"index": i, "rgb": rgb, "lab": np.asarray(lab).tolist(), "yarn_code": "",
                            "name": "", "gradient_endpoint": ep})
        source = "photo"
    if len(entries) > mc:
        entries = entries[:mc]
    for i, e in enumerate(entries):
        e["index"] = i
    pal = {"source": source, "machine_colors": mc, "entries": entries}
    dump_json(sd / "palette.json", pal)
    dump_json(sd / "palette_proposed.json", pal)
    cv2.imwrite(str(sd / "palette_card.png"), cv2.cvtColor(palette_card(entries), cv2.COLOR_RGB2BGR))
    job.state["colors"]["palette_used"] = len(entries)
    job.state["palette_confirmed"] = bool(job.confirm_palette or job.palette_override)
    job.state.setdefault("stages_ran", []).append("palette")
    job.timings["palette"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
