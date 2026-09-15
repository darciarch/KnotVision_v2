#!/usr/bin/env python
"""QC calibration: eval/score_sheet.csv + runs/<job_id>/output/qc.json -> models/qc_calib.json.

Protocol (spec, "Kalibrasyon protokolü"):
1. >= 100 outputs with class quotas (classic 15, geometric 15, gradient 10, distressed 10, minimal 5);
   a class under quota falls to `other` (no decision in that class). Edit configurations need >= 10
   outputs per (agent, class) for at most two agents and edit_gate_source: eval.
2. Intra-rater consistency: designer_score_2 on >= 10 outputs; quadratic weighted kappa < 0.5 -> the
   whole set is scored twice and the mean is the label.
3. Core feature set; n_features = min(10, floor(n / 15)) in priority order; > 50 % null -> dropped.
4. L2 logistic regression, C from {0.1, 0.3, 1, 3} by inner 5-fold CV; 5-fold stratified CV AUC
   overall / per class / per config. Overall < 0.75 -> refused. Class < 0.65 -> class_unreliable.
   Config (edit + agent) < 0.65 -> config_unreliable (dropped from calibrated_configs).
5. qc_accept: lowest threshold with precision >= 0.9 for score >= target; qc_reject: highest
   threshold that rejects >= 90 % of score <= 2.
6. Recalibration: forced on model/agent hash change; else every 200 jobs a 30-output drift check.

Usage: python eval/calibrate_qc.py --sheet eval/score_sheet.csv [--runs runs] [--out models/qc_calib.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knotvision.job import MODELS_DIR, RUNS_DIR, agent_hash, read_json, sha256_file  # noqa: E402
from knotvision.qc import CORE_FEATURES  # noqa: E402

CLASS_QUOTA = {"classic": 15, "geometric": 15, "gradient": 10, "distressed": 10, "minimal": 5}
CLASS_LEVEL_MIN = 10
EDIT_QUOTA = 10
MAX_EDIT_AGENTS = 2
C_GRID = (0.1, 0.3, 1.0, 3.0)


# ----------------------------------------------------------------------------- data

def load_sheet(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            r["designer_score"] = float(r["designer_score"]) if r.get("designer_score") else None
            r["designer_score_2"] = float(r["designer_score_2"]) if r.get("designer_score_2") else None
            r["fix_minutes"] = float(r["fix_minutes"]) if r.get("fix_minutes") else None
            r["regularize"] = str(r.get("regularize", "true")).lower() in ("1", "true", "on", "yes")
            r["edit_agent"] = r.get("edit_agent") or None
            rows.append(r)
    return rows


def attach_qc(rows: list[dict], runs_dir: Path) -> list[dict]:
    out = []
    for r in rows:
        qj = runs_dir / r["job_id"] / "output" / "qc.json"
        if not qj.exists():
            qj = runs_dir / r["job_id"] / "stages" / "08_qc" / "qc.json"
        if not qj.exists():
            continue
        q = read_json(qj)
        r = dict(r)
        r["features"] = q["features"]
        out.append(r)
    return out


# ----------------------------------------------------------------------------- helpers

def kappa_quadratic(a, b) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) < 2:
        return None
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def labels_from_rows(rows: list[dict], kappa_min: float = 0.5) -> tuple[np.ndarray, float | None]:
    s1 = np.array([r["designer_score"] for r in rows], float)
    both = [(r["designer_score"], r["designer_score_2"]) for r in rows if r.get("designer_score_2") is not None]
    kappa = kappa_quadratic([a for a, _ in both], [b for _, b in both]) if len(both) >= 2 else None
    if kappa is not None and kappa < kappa_min:
        # the whole set must be scored twice; where a second score exists, the mean becomes the label
        s = np.array([(r["designer_score"] + r["designer_score_2"]) / 2 if r.get("designer_score_2") is not None
                      else r["designer_score"] for r in rows])
        return s, kappa
    return s1, kappa


def choose_target(scores: np.ndarray, min_each: int = 15) -> int | None:
    for t in (5, 4, 3):
        if (scores >= t).sum() >= min_each and (scores < t).sum() >= min_each:
            return t
    return None


def select_features(rows: list[dict], n: int, extra_priority: list | None = None) -> tuple[list, dict]:
    names = list(CORE_FEATURES)
    for e in extra_priority or []:
        if e not in names:
            names.append(e)
    dropped = {}
    kept = []
    for f in names:
        vals = [r["features"].get(f) for r in rows]
        null_share = sum(v is None for v in vals) / max(1, len(vals))
        if null_share > 0.5:
            dropped[f] = "null_share"
            continue
        kept.append(f)
    n_feat = min(n, len(kept))
    for f in kept[n_feat:]:
        dropped[f] = "n_cap"
    return kept[:n_feat], dropped


def design_matrix(rows, features, class_levels, mode_levels, mean=None, std=None):
    X = []
    for r in rows:
        x = []
        for f in features:
            v = r["features"].get(f)
            x.append(np.nan if v is None else float(v))
        X.append(x)
    X = np.array(X, float).reshape(len(rows), len(features))
    if mean is None:
        mean = np.nanmean(X, axis=0) if len(features) else np.zeros(0)
        std = np.nanstd(X, axis=0) if len(features) else np.zeros(0)
        std = np.where(std < 1e-9, 1.0, std)
        mean = np.nan_to_num(mean)
    Z = (X - mean) / std
    Z = np.nan_to_num(Z, nan=0.0)   # null -> standardized 0
    cls = np.array([[1.0 if (r["design_class"] if r["design_class"] in class_levels else "other") == lvl else 0.0
                     for lvl in class_levels] for r in rows]).reshape(len(rows), len(class_levels))
    mode = np.array([[1.0 if r["mode"] == lvl else 0.0 for lvl in mode_levels] for r in rows]).reshape(len(rows), len(mode_levels))
    return np.hstack([Z, cls, mode]), mean, std


def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    best_c, best = None, -1
    n_splits = min(5, int(min(y.sum(), (1 - y).sum())))
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000)
        if n_splits >= 2:
            s = cross_val_score(m, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0), scoring="roc_auc").mean()
        else:
            s = 0.0
        if s > best:
            best, best_c = s, C
    m = LogisticRegression(penalty="l2", C=best_c, max_iter=2000).fit(X, y)
    return m, best_c


def cv_auc(X, y, groups: dict[str, np.ndarray], strat) -> tuple[float, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    n_splits = min(5, int(min(y.sum(), (1 - y).sum())))
    if n_splits < 2:
        return 0.0, {k: None for k in groups}
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, strat):
        m = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    overall = float(roc_auc_score(y, pred))
    per = {}
    for k, mask in groups.items():
        if mask.sum() >= 6 and 0 < y[mask].sum() < mask.sum():
            per[k] = float(roc_auc_score(y[mask], pred[mask]))
        else:
            per[k] = None
    return overall, per


def thresholds(prob: np.ndarray, y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """qc_accept: lowest threshold with precision >= 0.9 for the positive class; qc_reject: highest
    threshold below which >= 90 % of score <= 2 outputs fall."""
    grid = np.linspace(0.05, 0.99, 95)
    accept = 0.99
    for t in grid:
        sel = prob >= t
        if sel.sum() and y[sel].mean() >= 0.9:
            accept = float(t)
            break
    bad = scores <= 2
    reject = 0.0
    if bad.any():
        for t in grid[::-1]:
            if (prob[bad] < t).mean() >= 0.9:
                reject = float(t)
                break
    reject = min(reject, accept)
    return accept, reject


# ----------------------------------------------------------------------------- calibration

def calibrate(rows: list[dict], profile: dict | None = None, kappa_min: float = 0.5, auc_min: float = 0.75,
              class_auc_min: float = 0.65, extra_features: list | None = None) -> dict:
    profile = profile or {}
    rows = [r for r in rows if r.get("designer_score") is not None and r.get("features") is not None]
    n = len(rows)
    result = {"n_samples": n, "date": time.strftime("%Y-%m-%d"), "status": "ok", "notes": []}
    # class levels + quota
    hist = {}
    for r in rows:
        hist[r["design_class"]] = hist.get(r["design_class"], 0) + 1
    result["class_hist"] = hist
    class_levels = [c for c in CLASS_QUOTA if hist.get(c, 0) >= CLASS_LEVEL_MIN] + ["other"]
    under_quota = [c for c, q in CLASS_QUOTA.items() if hist.get(c, 0) < q]
    if under_quota:
        result["notes"].append(f"class quota not met: {under_quota}")
    if n < 100:
        result["notes"].append(f"n_samples {n} < 100")
    # labels
    scores, kappa = labels_from_rows(rows, kappa_min)
    result["intra_rater_kappa"] = kappa
    result["score_hist"] = {str(int(s)): int((np.round(scores) == s).sum()) for s in (1, 2, 3, 4, 5)}
    fix_hist = {}
    for r in rows:
        if r.get("fix_type"):
            fix_hist[r["fix_type"]] = fix_hist.get(r["fix_type"], 0) + 1
    result["fix_type_hist"] = fix_hist
    target = choose_target(scores)
    if target is None:
        result.update({"status": "refused", "reason": "no target with >= 15 samples on both sides"})
        return result
    y = (scores >= target).astype(int)
    # features
    n_feat = min(10, n // 15)
    features, dropped = select_features(rows, n_feat, extra_features)
    mode_levels = ["segment", "quantize", "edit"]
    X, mean, std = design_matrix(rows, features, class_levels, mode_levels)
    model, C = fit_logistic(X, y)
    # CV AUC overall / per class / per config
    cls_of = np.array([r["design_class"] if r["design_class"] in class_levels else "other" for r in rows])
    cfg_of = np.array([f"{r['design_class']}/{r['mode']}/{r['edit_agent']}" for r in rows])
    groups = {f"class:{c}": cls_of == c for c in class_levels}
    for cfg in sorted(set(cfg_of)):
        if "/edit/" in cfg:
            groups[f"config:{cfg}"] = cfg_of == cfg
    overall, per = cv_auc(X, y, groups, cls_of)
    result["cv_auc"] = overall
    result["cv_auc_by_class"] = {k.split(":", 1)[1]: v for k, v in per.items() if k.startswith("class:")}
    result["cv_auc_by_config"] = {k.split(":", 1)[1]: v for k, v in per.items() if k.startswith("config:")}
    if overall < auc_min:
        result.update({"status": "refused", "reason": f"cv_auc {overall:.3f} < {auc_min}"})
    result["class_unreliable"] = [c for c, v in result["cv_auc_by_class"].items() if v is not None and v < class_auc_min]
    result["config_unreliable"] = [c for c, v in result["cv_auc_by_config"].items() if v is not None and v < class_auc_min]
    # thresholds
    prob = model.predict_proba(X)[:, 1]
    accept, reject = thresholds(prob, y, scores)
    # p5 model
    model_p5 = None
    y5 = (scores >= 5).astype(int)
    if y5.sum() >= 15 and (1 - y5).sum() >= 15:
        m5, C5 = fit_logistic(X, y5)
        model_p5 = {"type": "logistic_l2", "C": C5, "coef": m5.coef_[0].tolist(), "intercept": float(m5.intercept_[0])}
    # calibrated configs: each (class, mode, agent, regularize) with enough samples, class in levels
    cfg_count = {}
    for r in rows:
        key = (r["design_class"], r["mode"], r["edit_agent"], bool(r["regularize"]))
        cfg_count[key] = cfg_count.get(key, 0) + 1
    edit_agents = sorted({k[2] for k in cfg_count if k[1] == "edit" and k[2]},
                         key=lambda a: -sum(v for k, v in cfg_count.items() if k[2] == a))[:MAX_EDIT_AGENTS]
    gate_ok = profile.get("edit_gate_source") == "eval"
    calibrated = []
    for (cls, mode, agent, reg), cnt in sorted(cfg_count.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]), kv[0][3])):
        if cls not in class_levels or cls == "other":
            continue
        if mode == "edit":
            if agent not in edit_agents or cnt < EDIT_QUOTA or not gate_ok:
                continue
            if f"{cls}/{mode}/{agent}" in result["config_unreliable"]:
                continue
        elif cnt < 5:
            continue
        calibrated.append([cls, mode, agent, reg])
    # fix_type: motif share for edit configs vs segment (> 2x -> config dropped, added motif gate stays 0)
    def motif_share(pred):
        sel = [r for r in rows if pred(r)]
        return (sum(1 for r in sel if r.get("fix_type") == "motif") / len(sel)) if sel else None
    seg_share = motif_share(lambda r: r["mode"] == "segment")
    for cfg in list(calibrated):
        if cfg[1] == "edit":
            es = motif_share(lambda r, c=cfg: r["mode"] == "edit" and r["edit_agent"] == c[2] and r["design_class"] == c[0])
            if es is not None and seg_share is not None and es > 2 * seg_share and seg_share > 0:
                calibrated.remove(cfg)
                result["notes"].append(f"{cfg[0]}/edit/{cfg[2]} dropped: fix_type motif share {es:.2f} > 2 x segment {seg_share:.2f}")
    seg = MODELS_DIR / "knotvision_segment.pt"
    result.update({
        "features": features, "dropped_features": dropped, "class_levels": class_levels, "mode_levels": mode_levels,
        "calibrated_configs": calibrated,
        "agent_hashes": {a: agent_hash(a) for a in edit_agents if (ROOT / "config" / "agents" / f"{a}.yaml").exists()},
        "model": {"type": "logistic_l2", "C": C, "mean": mean.tolist(), "std": std.tolist(),
                  "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])},
        "model_p5": model_p5, "target": int(target), "qc_accept": accept, "qc_reject": reject,
        "model_hashes": {"segment": sha256_file(seg) if seg.exists() else None, "quantize": "meanshift-sp24-sr20",
                         "qwen_edit_2511": None, "edit_lora": None},
    })
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(ROOT / "eval" / "score_sheet.csv"))
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--out", default=str(MODELS_DIR / "qc_calib.json"))
    ap.add_argument("--profile", default="wilton_default")
    ap.add_argument("--force", action="store_true", help="write even when refused (status recorded)")
    a = ap.parse_args(argv)
    from knotvision.job import CONFIG_DIR, load_yaml
    P = load_yaml(CONFIG_DIR / "looms" / f"{a.profile}.yaml")
    rows = attach_qc(load_sheet(Path(a.sheet)), Path(a.runs))
    res = calibrate(rows, P)
    print(json.dumps({k: v for k, v in res.items() if k not in ("model", "model_p5")}, indent=2))
    if res["status"] == "ok" or a.force:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"written {a.out}")
        return 0
    print("calibration refused; nothing written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
