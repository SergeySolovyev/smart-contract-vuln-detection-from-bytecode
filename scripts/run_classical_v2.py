# -*- coding: utf-8 -*-
"""Paper v2 — Phase 2: classical binary models with an honest protocol.

Protocol (the whole point of v2 — contrast with v1 in comments):
  * Optuna 50 trials for XGBoost, objective = mean F1 over a REAL 5-fold
    stratified CV inside TRAIN ONLY. (v1 optimised F1 directly on the
    reporting split — tuned-on-test.)
  * Operating threshold selected on VAL. (v1 swept thresholds on the
    reporting split.)
  * Every number reported for the paper comes from TEST, touched exactly
    once per model at the very end.
  * B=1000 stratified percentile bootstrap CIs on test (same machinery as
    v1 — that part was correct), PLUS a paired ΔF1 bootstrap CI for
    XGB-vs-RF (v1 argued from overlapping marginal CIs, which is invalid).

Outputs: v2/results_classical/{model}.json, binary_results_v2.csv,
paired_delta_v2.json, thresholds_v2.json.
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper\v2")
RES = BASE / "results_classical"
RES.mkdir(exist_ok=True)
SEED = 42

FEATURES = json.load(open(
    r"D:\DeFi\tmp\kaggle_solovev_output_v11\results\feature_columns.json",
    encoding="utf-8"))
LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]

def load(name):
    df = pd.read_parquet(BASE / f"{name}_v2.parquet",
                         columns=FEATURES + ["is_vulnerable"])
    return df[FEATURES].to_numpy("float32"), df["is_vulnerable"].to_numpy()

Xtr, ytr = load("train")
Xva, yva = load("val")
Xte, yte = load("test")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
print(f"pos rate: train {ytr.mean():.3f}  val {yva.mean():.3f}  test {yte.mean():.3f}")

# ── XGBoost + Optuna with real CV ───────────────────────────────────────────
import optuna
import xgboost as xgb
optuna.logging.set_verbosity(optuna.logging.WARNING)

def cv_f1(params):
    """Mean F1 over 5 stratified folds of TRAIN — val/test never touched."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, ho_idx in skf.split(Xtr, ytr):
        m = xgb.XGBClassifier(**params, random_state=SEED,
                              n_jobs=-1, eval_metric="logloss")
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        scores.append(f1_score(ytr[ho_idx], m.predict(Xtr[ho_idx])))
    return float(np.mean(scores))

def objective(t):
    params = dict(
        n_estimators=t.suggest_int("n_estimators", 200, 1500),
        max_depth=t.suggest_int("max_depth", 4, 12),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 6.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 10),
    )
    return cv_f1(params)

t0 = time.time()
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)
print(f"Optuna done in {time.time()-t0:.0f}s; best CV-F1 {study.best_value:.4f}")
json.dump({"best_cv_f1": study.best_value, "best_params": study.best_params,
           "n_trials": 50, "cv": "5-fold stratified on train only"},
          open(RES / "optuna_summary.json", "w"), indent=2)

MODELS = {
    "LogReg": lambda: __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
        [("sc", StandardScaler()),
         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED))]),
    "RF": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "XGB": lambda: xgb.XGBClassifier(**study.best_params, random_state=SEED,
                                     n_jobs=-1, eval_metric="logloss"),
}
try:
    from catboost import CatBoostClassifier
    MODELS["CatBoost"] = lambda: CatBoostClassifier(
        iterations=800, auto_class_weights="Balanced",
        random_seed=SEED, verbose=False)
except ImportError:
    print("catboost not installed - skipping")

def boot_ci(y, s, thr, B=1000, seed=SEED):
    """Stratified percentile bootstrap for F1 (same machinery as v1)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(f1_score(y[i], (s[i] >= thr).astype(int)))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

rows, scores_test, thresholds = [], {}, {}
for name, ctor in MODELS.items():
    t0 = time.time()
    m = ctor()
    m.fit(Xtr, ytr)
    fit_s = time.time() - t0

    # threshold: chosen on VAL, never on test
    sva = m.predict_proba(Xva)[:, 1]
    grid = np.linspace(0.05, 0.95, 181)
    thr = float(grid[int(np.argmax([f1_score(yva, (sva >= g).astype(int))
                                    for g in grid]))])
    thresholds[name] = thr

    ste = m.predict_proba(Xte)[:, 1]
    scores_test[name] = ste
    yhat = (ste >= thr).astype(int)
    lo, hi = boot_ci(yte, ste, thr)
    fn = int(((yhat == 0) & (yte == 1)).sum())
    row = dict(
        model=name, threshold=thr,
        f1=float(f1_score(yte, yhat)), f1_lo=lo, f1_hi=hi,
        precision=float(precision_score(yte, yhat)),
        recall=float(recall_score(yte, yhat)),
        mcc=float(matthews_corrcoef(yte, yhat)),
        fnr=float(fn / max(1, int((yte == 1).sum()))),
        pr_auc=float(average_precision_score(yte, ste)),
        roc_auc=float(roc_auc_score(yte, ste)),
        train_s=round(fit_s, 1),
    )
    rows.append(row)
    json.dump({**row, "y_score_test": ste.round(6).tolist()},
              open(RES / f"{name}.json", "w"))
    print(f"{name:9s} F1={row['f1']:.4f} [{lo:.4f},{hi:.4f}] "
          f"P={row['precision']:.3f} R={row['recall']:.3f} "
          f"FNR={row['fnr']:.3%} PR-AUC={row['pr_auc']:.4f} ({fit_s:.0f}s)")

pd.DataFrame(rows).to_csv(RES / "binary_results_v2.csv", index=False)
json.dump(thresholds, open(RES / "thresholds_v2.json", "w"), indent=2)

# ── paired ΔF1: the statistically correct XGB-vs-RF comparison ──────────────
rng = np.random.default_rng(SEED)
pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
deltas = []
for _ in range(2000):
    i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    d = (f1_score(yte[i], (scores_test["XGB"][i] >= thresholds["XGB"]).astype(int))
         - f1_score(yte[i], (scores_test["RF"][i] >= thresholds["RF"]).astype(int)))
    deltas.append(d)
paired = {"delta_f1_xgb_minus_rf": float(np.mean(deltas)),
          "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
          "p_delta_gt_0": float(np.mean(np.array(deltas) > 0)), "B": 2000}
json.dump(paired, open(RES / "paired_delta_v2.json", "w"), indent=2)
print(f"\npaired ΔF1 (XGB−RF) = {paired['delta_f1_xgb_minus_rf']:+.5f} "
      f"CI95 {paired['ci95']}  P(Δ>0)={paired['p_delta_gt_0']:.3f}")
print("\nPhase 2 complete.")
