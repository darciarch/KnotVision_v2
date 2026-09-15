"""regularize: deterministic, model-free. Repeat / symmetry merge (majority vote, template first),
boundary smoothing (Douglas-Peucker, angle lock, arc fit, free runs), re-rasterisation, guard.

Untouched in every step: the gradient mask and regions == 4 (corner blocks). Mode-independent.
regularize: off -> indexed = indexed_raw, statistics still computed.

Output: indexed.npy, regularize_stats.json, stages/guard.json (appended)
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_polygon
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize

from .guard import assert_no_motif_loss, summarize
from .job import Job, dump_json, read_json
from .metrics import connected_components, is_thin, line_jaggedness, repeat_mismatch, symmetry_mismatch
from .production import cleanup
from .structure import REGION_BORDER, REGION_CORNER, REGION_GROUND, REGION_MEDALLION

ALLOWED_SLOPES = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1), (1, 3), (3, 1), (1, -3), (3, -1)]
_ALLOWED_ANGLES = sorted(set(np.degrees(np.arctan2(dy, dx)) % 180 for dy, dx in ALLOWED_SLOPES))


# ----------------------------------------------------------------------------- majority votes

def _vote(stack: np.ndarray) -> np.ndarray:
    """Majority vote along axis 0; ties -> lowest index. stack values < 0 are ignored."""
    n, H, W = stack.shape
    k = int(stack.max()) + 1 if stack.size else 1
    counts = np.zeros((k, H, W), dtype=np.int32)
    for i in range(n):
        s = stack[i]
        valid = s >= 0
        np.add.at(counts, (s[valid], *np.nonzero(valid)), 1)
    return counts.argmax(axis=0).astype(np.int32)


def _shift(a: np.ndarray, dy: int, dx: int, fill: int = -1) -> np.ndarray:
    out = np.full_like(a, fill)
    H, W = a.shape
    if abs(dy) >= H or abs(dx) >= W:
        return out
    ys = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, dx), min(W, W + dx))
    out[ys, xs] = a[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx]
    return out


def enforce_symmetry(indexed: np.ndarray, kind: str, region: np.ndarray) -> np.ndarray:
    H, W = indexed.shape
    src = np.where(region, indexed, -1)
    views = [src]
    if kind in ("h", "hv"):
        views.append(src[:, ::-1])
    if kind in ("v", "hv"):
        views.append(src[::-1, :])
    if kind == "hv":
        views.append(src[::-1, ::-1])
    voted = _vote(np.stack(views))
    out = indexed.copy()
    out[region] = voted[region]
    # axis column/row is its own mirror -> unchanged by construction when W/H odd
    return out


def enforce_repeat(indexed: np.ndarray, v1, v2, region: np.ndarray, reach: int = 2) -> np.ndarray:
    """Majority vote across lattice translations (k*v1 + l*v2, |k|,|l| <= reach) inside `region`."""
    v1 = (int(round(v1[0])), int(round(v1[1])))
    v2 = (int(round(v2[0])), int(round(v2[1]))) if v2 is not None else None
    src = np.where(region, indexed, -1)
    views = []
    for k in range(-reach, reach + 1):
        for l in (range(-reach, reach + 1) if v2 else [0]):
            dy = k * v1[0] + (l * v2[0] if v2 else 0)
            dx = k * v1[1] + (l * v2[1] if v2 else 0)
            views.append(_shift(src, dy, dx))
    voted = _vote(np.stack(views))
    out = indexed.copy()
    out[region] = voted[region]
    return out


def enforce_border_period(indexed: np.ndarray, regions: np.ndarray, period_knots: dict) -> np.ndarray:
    """1-D period along each border strip (top/bottom share a period, left/right share a period)."""
    out = indexed.copy()
    H, W = indexed.shape
    strip = regions == REGION_BORDER
    rows = np.nonzero(strip.any(axis=1))[0]
    cols = np.nonzero(strip.any(axis=0))[0]
    for axis, p in (("h", period_knots.get("h")), ("v", period_knots.get("v"))):
        if not p or p < 2:
            continue
        p = int(round(p))
        if axis == "h":
            # top and bottom strips: shift along x
            top = strip & (np.arange(H)[:, None] < H // 2)
            bot = strip & ~top
            for part in (top, bot):
                if part.any():
                    src = np.where(part, out, -1)
                    views = [_shift(src, 0, k * p) for k in range(-3, 4)]
                    voted = _vote(np.stack(views))
                    out[part] = voted[part]
        else:
            left = strip & (np.arange(W)[None, :] < W // 2)
            right = strip & ~left
            for part in (left, right):
                if part.any():
                    src = np.where(part, out, -1)
                    views = [_shift(src, k * p, 0) for k in range(-3, 4)]
                    voted = _vote(np.stack(views))
                    out[part] = voted[part]
    return out


# ----------------------------------------------------------------------------- vectorisation

def _snap_angle(theta: float) -> float:
    theta = theta % 180
    best = min(_ALLOWED_ANGLES, key=lambda a: min(abs(a - theta), 180 - abs(a - theta)))
    return best


def _circle_fit(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa algebraic circle fit. Returns (centre, radius, rms residual)."""
    x, y = pts[:, 1], pts[:, 0]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(1e-9, sol[2] + cx ** 2 + cy ** 2))
    res = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return np.array([cy, cx]), float(r), float(res)


def _presmooth(contour: np.ndarray, size: int = 5) -> np.ndarray:
    """Moving average (size 5, wrap) along a closed contour before DP: +-1 knot jitter on a staircase
    shrinks below the DP tolerance while an ideal staircase becomes its underlying straight line
    (the corner rounding this causes stays below the tolerance and the angle lock re-snaps)."""
    if len(contour) < size + 2:
        return contour
    return ndi.uniform_filter1d(contour, size=size, axis=0, mode="wrap")


def _free_run(orig: np.ndarray, tol: float) -> np.ndarray:
    """Free segment: run-length median (size 5) -> piecewise-constant monotone steps; endpoints and
    total extent preserved; falls back to the original if any point moves more than `tol`."""
    if len(orig) <= 3:
        return orig
    sm = ndi.median_filter(orig, size=(5, 1), mode="nearest")
    sm[0], sm[-1] = orig[0], orig[-1]
    dev = np.hypot(*(sm - orig).T).max()
    return sm if dev <= tol else orig


def _arc_run(contour: np.ndarray, idx: list, i: int, tol: float, min_len: float = 8.0) -> tuple:
    """Longest run of DP vertices idx[i..j] (j >= i+3) whose contour points fit one circle with rms
    residual <= tol, arc length >= min_len knots, sagitta > tol and no single line within tol
    (spec: equal residuals -> line; a diamond side or a stepped polygon is described by its lines).
    (j, centre, r) or (None,)*3."""
    best = (None, None, None)
    for j in range(i + 3, len(idx)):
        pts = contour[idx[i]:idx[j] + 1]
        if len(pts) < 4 or _length(pts) < min_len:
            continue
        c, r, res = _circle_fit(pts)
        chord = float(np.hypot(*(pts[-1] - pts[0])))
        sagitta = r - np.sqrt(max(0.0, r * r - (chord / 2) ** 2))
        if res <= tol and 2.0 <= r < 1e4 and sagitta > tol and _line_rms(pts) > tol:
            best = (j, c, r)
        else:
            break
    return best


def _line_rms(pts: np.ndarray) -> float:
    """rms distance of pts to their least-squares line."""
    if len(pts) < 3:
        return 0.0
    c = pts.mean(0)
    u, s_, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[1]
    return float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))


def _line_run(contour: np.ndarray, idx: list, i: int, tol: float) -> int:
    """Longest j > i such that contour[idx[i]:idx[j]+1] lies on one line (rms <= tol, max <= 2 tol)."""
    best = i + 1
    for j in range(i + 1, len(idx)):
        pts = contour[idx[i]:idx[j] + 1]
        if len(pts) < 2:
            continue
        a, b = pts[0], pts[-1]
        ab = b - a
        L = float(np.hypot(*ab))
        if L < 1e-9:
            break
        rel = pts - a
        dmax = float(np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]).max() / L)
        if _line_rms(pts) <= tol and dmax <= 2 * tol:
            best = j
        else:
            break
    return best


def _ls_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through pts: (centroid, unit direction oriented from pts[0] to pts[-1])."""
    c = pts.mean(0)
    if len(pts) < 3:
        d = pts[-1] - pts[0]
    else:
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
    n = float(np.hypot(*d))
    d = d / n if n > 1e-9 else np.array([0.0, 1.0])
    if np.dot(d, pts[-1] - pts[0]) < 0:
        d = -d
    return c, d


def _project(line: tuple, p: np.ndarray) -> np.ndarray:
    c, d = line
    return c + d * float(np.dot(d, p - c))


def _resolve_vertex(prev_line, next_line, vpos: np.ndarray, tol: float) -> np.ndarray:
    """Position of the vertex shared by two runs. Two lines meet at their intersection (a diamond tip
    is the meeting of its two locked sides) unless they are nearly parallel or the intersection is
    further than 2 tol from the vertex, then at the mean of the two projections; a line and a
    non-line meet on the line; otherwise the vertex position is kept."""
    if prev_line is not None and next_line is not None:
        (c1, d1), (c2, d2) = prev_line, next_line
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) >= 0.2:
            w = c2 - c1
            t = float(w[0] * d2[1] - w[1] * d2[0]) / cross
            p = c1 + d1 * t
            if np.hypot(*(p - vpos)) <= 2.0 * tol:
                return p
        return 0.5 * (_project(prev_line, vpos) + _project(next_line, vpos))
    if prev_line is not None:
        return _project(prev_line, vpos)
    if next_line is not None:
        return _project(next_line, vpos)
    return vpos.copy()


def regularize_contour(contour: np.ndarray, tol: float, min_line: int, free_runs: bool) -> tuple[np.ndarray, dict]:
    """contour: (N, 2) closed polyline (row, col). Returns (new polyline, segment class lengths).

    1. moving-average pre-smoothing, 2. Douglas-Peucker at `tol` (segmentation only),
    3. greedy runs over DP vertices: the longest of {line run (LS rms <= tol), arc run (circle rms <=
    tol, >= 8 knots, sagitta > tol, not a line within tol)} wins, a line on ties; a line run >= min_line
    is angle-locked (direction snapped, line through the centroid of the run's points, accepted when
    both end vertices stay within tol of it), a shorter one is a free segment (run-length median),
    4. vertices between runs: two lines meet at their intersection, a line and a non-line on the line.
    Placing every line through its centroid (not through one jittered vertex) makes the boundary of
    a region and of its neighbour resolve to the same line."""
    orig = np.asarray(contour, dtype=np.float64)
    if len(orig) < 16:
        # a motif this small is pixel art: every knot was placed by hand, nothing to regularise
        return orig, {"line": 0.0, "arc": 0.0, "free": float(_length(orig))}
    # the smoothed contour decides the segmentation (runs, classes) and the line fits; the points
    # that are re-rasterised in arcs / free runs come from the original contour
    contour = _presmooth(orig)
    approx = approximate_polygon(contour, tolerance=tol)
    if len(approx) < 4:
        return orig, {"line": 0.0, "arc": 0.0, "free": float(_length(orig))}
    stats = {"line": 0.0, "arc": 0.0, "free": 0.0}
    idx = [int(np.argmin(np.hypot(*(contour - v).T))) for v in approx]
    for k in range(1, len(idx)):
        if idx[k] <= idx[k - 1]:
            idx[k] = idx[k - 1] + 1
    idx = [min(i, len(contour) - 1) for i in idx]
    idx[-1] = len(contour) - 1
    # vertex positions: a DP vertex that turns sharply (a convex corner, a diamond tip) keeps its
    # original position (the moving average pulls it inward); a vertex inside a gently bending or
    # jittered run takes the smoothed position (the original one carries the +-1 knot jitter)
    vpos = _vertex_positions(orig, contour, idx, corner_deg=30.0)
    # pass 1: runs. Each run: (i, j, kind, line, data); line = (centroid, unit dir) for lines
    runs = []
    i = 0
    while i < len(idx) - 1:
        jl = _line_run(contour, idx, i, tol)
        ja, c, r = _arc_run(contour, idx, i, tol)
        if ja is not None and ja > jl:
            runs.append((i, ja, "arc", None, (c, r)))
            i = ja
            continue
        pts_s = contour[idx[i]:idx[jl] + 1]
        L = float(np.hypot(*(vpos[jl] - vpos[i])))
        if L >= min_line:
            cen, d_ls = _ls_line(pts_s)
            theta = np.degrees(np.arctan2(d_ls[0], d_ls[1]))
            snap = np.radians(_snap_angle(theta))
            d = np.array([np.sin(snap), np.cos(snap)])
            if np.dot(d, d_ls) < 0:
                d = -d
            locked = (cen, d)
            dev = max(np.hypot(*(_project(locked, vpos[i]) - vpos[i])), np.hypot(*(_project(locked, vpos[jl]) - vpos[jl])))
            runs.append((i, jl, "line", locked if dev <= tol else (cen, d_ls), None))
        else:
            runs.append((i, jl, "free", None, None))
        i = jl
    # pass 2: vertices (closed contour: vertex 0 is shared by the last and the first run)
    n = len(runs)
    verts = []
    for k in range(n + 1):
        prev_run = runs[(k - 1) % n]
        next_run = runs[k % n]
        v = runs[k][0] if k < n else runs[-1][1]
        verts.append(_resolve_vertex(prev_run[3], next_run[3], vpos[v], tol))
    out = [verts[0]]
    for k, (i, j, kind, line, data) in enumerate(runs):
        a, b = verts[k], verts[k + 1]
        pts = orig[idx[i]:idx[j] + 1]
        if kind == "line":
            out.append(b)
            stats["line"] += float(np.hypot(*(b - a)))
        elif kind == "arc":
            c, r = data
            ang = np.arctan2(pts[:, 0] - c[0], pts[:, 1] - c[1])
            snapped = np.column_stack([c[0] + r * np.sin(ang), c[1] + r * np.cos(ang)])
            snapped[0], snapped[-1] = a, b
            out.extend(snapped[1:].tolist())
            stats["arc"] += float(_length(pts))
        else:
            pts = pts.copy()
            pts[0], pts[-1] = a, b
            fr = _free_run(pts, tol) if free_runs else pts
            out.extend(np.asarray(fr)[1:].tolist())
            stats["free"] += float(_length(pts))
    out = np.array(out, dtype=np.float64)
    return out, stats


def _vertex_positions(orig: np.ndarray, smooth: np.ndarray, idx: list, corner_deg: float) -> np.ndarray:
    """Per DP vertex: the original contour point at a corner (turn >= corner_deg between the incoming
    and outgoing DP segments), the smoothed point elsewhere. First and last vertex (same point on a
    closed contour) follow the corner test of that point."""
    n = len(idx)
    pos = np.array([smooth[k] for k in idx], dtype=np.float64)
    for v in range(n):
        prev = idx[v - 1] if v > 0 else idx[n - 2]
        nxt = idx[v + 1] if v < n - 1 else idx[1]
        a = smooth[idx[v]] - smooth[prev]
        b = smooth[nxt] - smooth[idx[v]]
        na, nb = np.hypot(*a), np.hypot(*b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        if np.degrees(np.arccos(cosang)) >= corner_deg:
            pos[v] = orig[idx[v]]
    return pos


def _length(pl: np.ndarray) -> float:
    if len(pl) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pl, axis=0).T).sum())


def smooth_boundaries(indexed: np.ndarray, protected: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Step 3: every component (largest first), boundary polylines -> regularised -> re-rasterised.
    Cells that fall in no polygon keep their indexed_raw index."""
    H, W = indexed.shape
    tol = float(params.get("reg_tol_knots", 0.7))
    min_line = int(params.get("reg_min_line_knots", 4))
    free_runs = bool(params.get("reg_free_runs", True))
    lmk, lmw = int(params.get("line_min_knots", 6)), int(params.get("line_max_width", 1))
    comps = connected_components(indexed, ~protected)
    comps.sort(key=lambda c: (-c.area, c.index))
    out = indexed.copy()
    stats = {"line": 0.0, "arc": 0.0, "free": 0.0}
    for c in comps:
        if c.area < 4:
            continue
        fm = c.full_mask(indexed.shape)
        thin = is_thin(fm, lmk, lmw)
        if thin:
            continue  # thin lines: keep as is (skeleton connectivity must survive)
        pad = np.pad(fm.astype(np.float32), 1)
        contours = find_contours(pad, 0.5)
        if not contours:
            continue
        contours = [ct - 1.0 for ct in contours]
        # outer = largest |area|
        def sarea(ct):
            y, x = ct[:, 0], ct[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        contours.sort(key=sarea, reverse=True)
        fill = np.zeros(indexed.shape, dtype=bool)
        for k, ct in enumerate(contours):
            new, st = regularize_contour(ct, tol, min_line, free_runs)
            for key in stats:
                stats[key] += st[key]
            # +0.01 row shift: a deterministic tie-break for cell centres lying exactly on a diagonal edge
            rr, cc = draw_polygon(new[:, 0] + 0.01, new[:, 1], shape=indexed.shape)
            if k == 0:
                fill[rr, cc] = True
            else:
                fill[rr, cc] = False
        fill &= ~protected
        # only cells that moved: write index where the polygon says so; cells that left the
        # component keep whatever was there (indexed_raw index) -> handled by not touching them
        out[fill] = c.index
    total = stats["line"] + stats["arc"] + stats["free"]
    seg = {k: (v / total if total else 0.0) for k, v in stats.items()}
    seg["free_len_share"] = seg["free"]
    return out, seg


def _thin_connectivity_ok(before: np.ndarray, after: np.ndarray, params: dict) -> np.ndarray:
    """Revert thin components whose skeleton broke into more pieces."""
    lmk, lmw = int(params.get("line_min_knots", 6)), int(params.get("line_max_width", 1))
    out = after.copy()
    for c in connected_components(before):
        fm = c.full_mask(before.shape)
        if not is_thin(fm, lmk, lmw):
            continue
        n_before = ndi.label(skeletonize(fm))[1]
        am = (after == c.index) & ndi.binary_dilation(fm, iterations=1)
        n_after = ndi.label(skeletonize(am))[1] if am.any() else 0
        if n_after > n_before or n_after == 0:
            out[fm] = c.index
    return out


# ----------------------------------------------------------------------------- enforcement decision

def _force(setting: str, conf: float, design_class: str, enforce_conf: float, detected: bool) -> bool:
    if setting == "off":
        return False
    if setting == "on":
        return detected
    return detected and conf >= enforce_conf and design_class in ("classic", "geometric")


# ----------------------------------------------------------------------------- stage

def run(job: Job) -> dict:
    t0 = time.time()
    sd = job.stage_dir("regularize")
    pr, st = job.stage_dir("production"), job.stage_dir("structure")
    inputs = [pr / "indexed_raw.npy", st / "layout.json", st / "regions.npy", st / "gradient_mask.npy"]
    key = job.cache_key("regularize", inputs)
    if job.cached(sd, key) and (sd / "indexed.npy").exists():
        job.timings["regularize"] = 0.0
        return job.state

    P = dict(job.params)
    P["design_class"] = job.design_class
    raw = np.load(pr / "indexed_raw.npy")
    layout = read_json(st / "layout.json")
    regions = np.load(st / "regions.npy")
    gmask = np.load(st / "gradient_mask.npy")
    protected = gmask | (regions == REGION_CORNER)
    guard_json = job.stages_dir / "guard.json"
    from .guard import reset
    reset(guard_json, keep_stages=("production_cleanup", "production_merge"))
    H, W = raw.shape

    setting = job.override.get("regularize") or P.get("regularize", "auto")
    enabled = setting != "off"
    sym = layout["symmetry"]
    rep = layout["ground_repeat"]
    period_px = layout.get("border_period", {})
    scale = layout.get("scale_512", 1.0)
    knots_per_px512 = max(W, H) / (512.0 if max(W, H) * 4 > 512 else max(W, H) * 4)
    period_knots = {}
    for axis, edges in (("h", ("top", "bottom")), ("v", ("left", "right"))):
        ps = [period_px[e]["period_px"] for e in edges if period_px.get(e, {}).get("period_px")]
        period_knots[axis] = float(np.mean(ps)) * knots_per_px512 if ps else None
    ground = np.isin(regions, [REGION_GROUND, REGION_MEDALLION]) & ~protected
    whole = ~protected

    stats = {"applied": False, "changed_pct": 0.0}
    stats["line_jaggedness_before"] = line_jaggedness(raw, ~gmask)
    stats["repeat_mismatch_raw"] = repeat_mismatch(raw, rep["v1"], rep["v2"], ground) if rep.get("v1") else None
    stats["symmetry_mismatch_raw"] = symmetry_mismatch(raw, sym["kind"], whole) if sym["kind"] != "none" else None
    stats["structures"] = []

    cur = raw.copy()
    if enabled:
        ec = float(P.get("enforce_conf", 0.85))
        fs = job.override.get("force_symmetry") or P.get("force_symmetry", "auto")
        fb = job.override.get("force_border_repeat") or P.get("force_border_repeat", "auto")
        fg = job.override.get("force_ground_repeat") or P.get("force_ground_repeat", "auto")
        # 2. structures: template (vote) -> smoothing happens in step 3 -> vote again keeps copies identical
        if _force(fb, max((period_px.get(e, {}).get("conf", 0.0) for e in period_px), default=0.0), job.design_class, ec,
                  layout.get("border_accepted", False) and any(period_knots.values())):
            new = enforce_border_period(cur, regions, period_knots)
            cur = assert_no_motif_loss(cur, new, P, "regularize_border", gmask, guard_json)
            stats["structures"].append("border_repeat")
        if _force(fg, rep.get("conf", 0.0), job.design_class, ec, bool(rep.get("v1"))):
            new = enforce_repeat(cur, rep["v1"], rep["v2"], ground)
            cur = assert_no_motif_loss(cur, new, P, "regularize_repeat", gmask, guard_json)
            stats["structures"].append("ground_repeat")
        if _force(fs, sym.get("conf", 0.0), job.design_class, ec, sym["kind"] != "none"):
            new = enforce_symmetry(cur, sym["kind"], whole)
            cur = assert_no_motif_loss(cur, new, P, "regularize_symmetry", gmask, guard_json)
            stats["structures"].append("symmetry")
        # 3. boundary smoothing
        smoothed, seg = smooth_boundaries(cur, protected, P)
        smoothed, _ = cleanup(smoothed, int(P.get("min_region_knots", 2)), protected, rounds=1)
        smoothed = _thin_connectivity_ok(cur, smoothed, P)
        # re-stamp structures so copies stay identical after smoothing
        if "symmetry" in stats["structures"]:
            smoothed = enforce_symmetry(smoothed, sym["kind"], whole)
        if "ground_repeat" in stats["structures"]:
            smoothed = enforce_repeat(smoothed, rep["v1"], rep["v2"], ground)
        cur = assert_no_motif_loss(cur, smoothed, P, "regularize_smooth", gmask, guard_json)
        stats["segment_classes"] = seg
        stats["applied"] = True
    else:
        stats["segment_classes"] = None

    stats["line_jaggedness_after"] = line_jaggedness(cur, ~gmask)
    stats["repeat_mismatch"] = repeat_mismatch(cur, rep["v1"], rep["v2"], ground) if rep.get("v1") else None
    stats["symmetry_mismatch"] = symmetry_mismatch(cur, sym["kind"], whole) if sym["kind"] != "none" else None
    stats["changed_pct"] = float((cur != raw).mean())
    stats["regularize_changed_pct"] = stats["changed_pct"]
    np.save(sd / "indexed.npy", cur.astype(np.int32))
    dump_json(sd / "regularize_stats.json", stats)
    job.state["regularize"] = stats
    job.state["guard"] = summarize(guard_json, float(P.get("guard_warn_pct", 0.02)))
    if job.state["guard"]["guard_heavy"]:
        job.warn("guard_heavy")
    job.state.setdefault("stages_ran", []).append("regularize")
    job.timings["regularize"] = round(time.time() - t0, 3)
    job.mark_done(sd, key)
    job.save()
    return job.state
