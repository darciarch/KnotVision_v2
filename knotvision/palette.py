"""palette: chosen from the photo BEFORE the model. Upper bound machine_colors, lower bound 1.

1. rectified -> guided filter -> Lab -> long side 400 px -> k-means (k = 4 x machine_colors, fixed seed).
   Cluster weight = count ** area_weight_exp; gradient-mask pixels weigh 0; each gradient region's two
   Lab endpoints are mandatory candidates.
2. Adjacency matrix (shared boundary pixels at 400 px; gradient-mask boundaries excluded).
3. Selection: palette_override | yarn library (greedy, stop at gain_stop_select) |
   greedy cluster merge (cost = weighted deltaE increase + adjacency_penalty * shared/total; down to
   machine_colors then continue while cost < merge_cost_stop). Gradient endpoints are never merged.
4. Motif-colour protection: clusters with small coverage but >= motif_color_min_cc components or a
   largest component >= motif_min_knots are merged last. Protection is evaluated after k-means
   duplicates (deltaE < DUP_DE) are merged, otherwise fragments of one flat colour look like motifs;
   pile-shading variants (deltaE < SHADE_DE meeting along boundaries without a colour edge) and
   transition mixes (a small cluster on the Lab segment between its two neighbours) merge likewise.
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
from .metrics import canny_lab, guided


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
    # pixels within 2 px of an edge are pile-blur transition colours, not yarns: excluded from the fit.
    # Colour edges: two yarns of equal lightness (a common two-colour rug) have no L edge, and their
    # transition band would otherwise enter the k-means as a chroma ramp of fake yarns.
    band = cv2.dilate(canny_lab(small).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    k = min(4 * machine_colors, max(2, (~gm).sum() // 50))
    flat = lab.reshape(-1, 3)
    valid = (~gm & ~band).reshape(-1)
    if valid.sum() < 100:
        valid = ~gm.reshape(-1)
    X = flat[valid] if valid.any() else flat
    km = KMeans(n_clusters=min(k, len(X)), random_state=seed, n_init=4).fit(X)
    labels_img = np.full(h * w, -1, dtype=np.int32)
    labels_img[valid] = km.labels_
    labels_img = labels_img.reshape(h, w)
    raw_labels = labels_img.copy()   # -1 in the edge band: adjacency here is a soft boundary
    # fill the band with the nearest labelled pixel so that adjacency is measured across it
    if (labels_img < 0).any() and (labels_img >= 0).any():
        _, (iy, ix) = ndi.distance_transform_edt(labels_img < 0, return_indices=True)
        filled = labels_img[iy, ix]
        labels_img = np.where(gm, -1, filled)
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
    # adjacency (400 px): shared boundary pixels between clusters, excluding gradient-mask boundaries.
    # adj: across the filled edge band (motif boundaries); adj_soft: only pairs where neither pixel
    # lies in the band, i.e. boundaries with no colour edge (pile shading inside one yarn)
    n = len(centers)

    def adjacency(li):
        m_ = np.zeros((n, n), dtype=np.float64)
        a, b = li[:, :-1], li[:, 1:]
        m = (a != b) & (a >= 0) & (b >= 0) & ~gm[:, :-1] & ~gm[:, 1:]
        np.add.at(m_, (a[m], b[m]), 1)
        a, b = li[:-1, :], li[1:, :]
        m = (a != b) & (a >= 0) & (b >= 0) & ~gm[:-1, :] & ~gm[1:, :]
        np.add.at(m_, (a[m], b[m]), 1)
        return m_ + m_.T

    adj = adjacency(labels_img)
    adj_soft = adjacency(raw_labels)
    return {"centers": centers, "weights": weights, "counts": counts, "forced": forced,
            "endpoint_of": endpoint_of, "adj": adj, "adj_soft": adj_soft, "labels_img": labels_img,
            "lab_small": lab, "gm_small": gm}


# ----------------------------------------------------------------------------- motif-colour protection

DUP_DE = 2.5   # clusters closer than this are the same yarn split by k-means; they merge before anything else
SHADE_DE = 8.0   # up to this deltaE two clusters whose common boundary is mostly soft (no colour edge) are
                 # pile-shading variants of one yarn and merge at no cost like duplicates; the same 8 units
                 # as structure.coarse_labels' merge_de (centres closer than that are one yarn)


def _group_label_image(cl: dict, groups: list[list[int]]) -> np.ndarray:
    lut = np.full(len(cl["centers"]) + 1, -1, dtype=np.int32)
    for gi, g in enumerate(groups):
        for i in g:
            lut[i] = gi
    li = cl["labels_img"]
    return np.where(li >= 0, lut[np.clip(li, 0, len(lut) - 1)], -1)


def motif_protected(cl: dict, motif_color_min_cc: int, motif_min_knots: int, W: int, H: int,
                    groups: Optional[list[list[int]]] = None, cc_min_knots: int = 7) -> np.ndarray:
    """Colours whose coverage is small but that form >= motif_color_min_cc components or one
    component >= motif_min_knots (measured on the 400 px label image, scaled to knots). Evaluated per
    cluster, or per group when `groups` is given (k-means fragments of one flat colour look like many
    components; dedup first, then protect). Only components of >= cc_min_knots knots count towards
    motif_color_min_cc: pile-shading speckle of a big colour is many knot-scale blobs (noise_cc_max
    and below, the spec's noise scale), a motif colour is drawn in larger pieces."""
    labels_img = cl["labels_img"] if groups is None else _group_label_image(cl, groups)
    h, w = labels_img.shape
    knots_per_px = (W * H) / float(h * w)
    n = len(cl["centers"]) if groups is None else len(groups)
    prot = np.zeros(n, dtype=bool)
    total = labels_img.size
    for i in range(n):
        m = labels_img == i
        cov = m.sum() / total
        if cov == 0 or cov > 0.05:
            continue
        lab, ncc = ndi.label(m)
        if ncc == 0:
            continue
        sizes = ndi.sum(m, lab, index=np.arange(1, ncc + 1)) * knots_per_px
        if int((sizes >= cc_min_knots).sum()) >= motif_color_min_cc or sizes.max() >= motif_min_knots:
            prot[i] = True
    return prot


# ----------------------------------------------------------------------------- selection paths

class _Groups:
    """Greedy agglomeration state over the original clusters (weighted Lab centres, forced flags)."""

    def __init__(self, cl: dict, groups: Optional[list[list[int]]] = None):
        centers, weights, forced = cl["centers"], cl["weights"], cl["forced"]
        self.adj = cl["adj"]
        self.adj_soft = cl.get("adj_soft", np.zeros_like(self.adj))
        self.total_adj = self.adj.sum() / 2 + 1e-9
        self.groups = [list(g) for g in groups] if groups is not None else [[i] for i in range(len(centers))]
        self.cent, self.w, self.forced = [], [], []
        for g in self.groups:
            f = [i for i in g if forced[i]]
            wsum = float(sum(weights[i] for i in g))
            if f:
                self.cent.append(centers[f[0]].copy())
            else:
                self.cent.append(sum(centers[i] * weights[i] for i in g) / max(wsum, 1e-9))
            self.w.append(wsum)
            self.forced.append(bool(f))

    def shared(self, a: int, b: int) -> float:
        return float(sum(self.adj[i, j] for i in self.groups[a] for j in self.groups[b]))

    def soft_share(self, a: int, b: int) -> float:
        """Share of the common boundary that carries no colour edge (0 when there is no boundary)."""
        tot = self.shared(a, b)
        if tot <= 0:
            return 0.0
        return float(sum(self.adj_soft[i, j] for i in self.groups[a] for j in self.groups[b])) / tot

    def merge(self, a: int, b: int):
        if self.forced[b] and not self.forced[a]:
            a, b = b, a
        if not self.forced[a]:
            self.cent[a] = (self.cent[a] * self.w[a] + self.cent[b] * self.w[b]) / (self.w[a] + self.w[b])
        self.w[a] += self.w[b]
        self.groups[a] = self.groups[a] + self.groups[b]
        del self.groups[b], self.cent[b], self.w[b], self.forced[b]
        return a, b


def dedup_groups(cl: dict, dup_de: float = DUP_DE) -> list[list[int]]:
    """Merge clusters that are the same colour (deltaE < dup_de). Two forced (gradient endpoint)
    clusters never merge; a forced cluster may absorb a free duplicate and keeps its centre."""
    G = _Groups(cl)
    while len(G.groups) > 1:
        best = None
        for a in range(len(G.groups)):
            for b in range(a + 1, len(G.groups)):
                if G.forced[a] and G.forced[b]:
                    continue
                de = float(delta_e2000(G.cent[a], G.cent[b]))
                if de < dup_de and (best is None or de < best[0]):
                    best = (de, a, b)
        if best is None:
            break
        G.merge(best[1], best[2])
    return G.groups


def merge_clusters(cl: dict, machine_colors: int, adjacency_penalty: float, merge_cost_stop: float,
                   protected: Optional[np.ndarray] = None, groups: Optional[list[list[int]]] = None) -> list[list[int]]:
    """Greedy merge; returns groups of original cluster ids.

    Duplicates (deltaE < DUP_DE) merge first at no cost. Then cost = weighted deltaE increase +
    adjacency_penalty * shared / total boundary; merging continues down to machine_colors and after
    that while the cost stays below merge_cost_stop. Forced (gradient endpoints) never merge with each
    other. Protected (motif) colours carry a +1e3 cost against a different colour, i.e. they merge last;
    `protected` is per group when `groups` is given, else per cluster (a group is protected when every
    member is)."""
    if groups is None:
        groups = dedup_groups(cl)
        if protected is not None and len(protected) == len(cl["centers"]):
            protected = np.array([all(bool(protected[i]) for i in g) for g in groups])
    G = _Groups(cl, groups)
    gprot = [bool(x) for x in protected] if protected is not None and len(protected) == len(groups) else [False] * len(groups)
    _fold_transitions(G, gprot)
    while len(G.groups) > 1:
        best = None
        for a in range(len(G.groups)):
            for b in range(a + 1, len(G.groups)):
                if G.forced[a] and G.forced[b]:
                    continue
                de = float(delta_e2000(G.cent[a], G.cent[b]))
                if (G.forced[a] or G.forced[b]) and de > DUP_DE:
                    continue   # a forced endpoint may absorb a free cluster only if it is the same colour
                if de < DUP_DE or (de < SHADE_DE and not (G.forced[a] or G.forced[b]) and G.soft_share(a, b) >= 0.5):
                    # merged centres drift: two groups that became the same colour are still duplicates;
                    # two close colours meeting mostly along soft boundaries are shading of one yarn.
                    # Both merge at no cost (the adjacency penalty is for motif boundaries)
                    cost = 0.0
                else:
                    wa, wb = G.w[a], G.w[b]
                    cost = de * min(wa, wb) / (wa + wb) + adjacency_penalty * G.shared(a, b) / G.total_adj
                    if (gprot[a] or gprot[b]) and de >= SHADE_DE:
                        cost += 1e3   # protection only keeps a motif colour apart from a *different* colour
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        if best is None:
            break
        cost, a, b = best
        if len(G.groups) <= machine_colors and cost >= merge_cost_stop:
            break
        pa, pb = gprot[a], gprot[b]
        a, b = G.merge(a, b)
        gprot[a] = pa and pb
        del gprot[b]
    return G.groups


def _fold_transitions(G: "_Groups", gprot: list, max_share: float = 0.05, band: float = 0.35) -> None:
    """Fold pile-blur transition colours: a small, unprotected, unforced group whose centre lies on the
    Lab segment between its two most adjacent larger neighbours (within `band` of the segment length)
    is the mix of those two yarns, not a yarn (the counterpart of structure._absorb_transition_bands
    at palette level; a chroma ramp between two isoluminant yarns would otherwise survive the L-edge
    band and the cost merge as a run of fake yarns). Smallest group first, lowest id on ties; repeats
    until stable. Merges into the nearer neighbour at no cost."""
    changed = True
    while changed and len(G.groups) > 2:
        changed = False
        total_w = sum(G.w) + 1e-9
        order = sorted(range(len(G.groups)), key=lambda i: (G.w[i], min(G.groups[i])))
        for g in order:
            if G.forced[g] or gprot[g] or G.w[g] / total_w > max_share:
                continue
            nb = sorted((i for i in range(len(G.groups)) if i != g and G.w[i] > G.w[g]),
                        key=lambda i: (-G.shared(g, i), i))
            if len(nb) < 2 or G.shared(g, nb[1]) <= 0:
                continue
            a, b = nb[0], nb[1]
            ab = G.cent[b] - G.cent[a]
            L2 = float(ab @ ab) + 1e-9
            t = float(((G.cent[g] - G.cent[a]) @ ab) / L2)
            perp = float(np.linalg.norm(G.cent[g] - (G.cent[a] + t * ab)))
            if not (0.0 <= t <= 1.0 and perp < band * np.sqrt(L2)):
                continue
            da, db = float(delta_e2000(G.cent[g], G.cent[a])), float(delta_e2000(G.cent[g], G.cent[b]))
            keep = a if da <= db else b
            pk = gprot[keep]
            keep, gone = G.merge(keep, g)
            gprot[keep] = pk
            del gprot[gone]
            changed = True
            break


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
    mc = int(job.machine_colors)
    key = job.cache_key("palette", inputs, extra={"override": job.palette_override, "mc": mc})
    if job.cached(sd, key) and (sd / "palette.json").exists():
        job.timings["palette"] = 0.0
        return job.state

    P = job.params
    W, H = job.state["grid"]["W"], job.state["grid"]["H"]
    rect = cv2.cvtColor(cv2.imread(str(rect_png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gmask = np.load(st / "gradient_mask.npy")
    gregions = read_json(st / "gradient_regions.json")

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
        groups0 = dedup_groups(cl)
        prot = motif_protected(cl, int(P.get("motif_color_min_cc", 5)), int(P.get("motif_min_knots", 10)), W, H, groups=groups0,
                               cc_min_knots=int(P.get("noise_cc_max", 6)) + 1)
        groups = merge_clusters(cl, mc, float(P.get("adjacency_penalty", 2.0)), float(P.get("merge_cost_stop", 0.5)), prot, groups=groups0)
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
