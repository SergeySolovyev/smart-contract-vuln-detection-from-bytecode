"""Paired class-level significance test for ICICPE 2026 multi-label results.

Why this, not a bootstrap CI on the macro-F1 gap?

The V12 Kaggle pipeline saved aggregate metrics (macro-F1, per-class F1,
precision, recall, training history) but NOT per-sample predictions. A
paired bootstrap over validation rows is therefore infeasible without
re-running the full pipeline on the 1.27 GB parquet (and Kaggle's 12 h
session cap already cancelled V12 mid-run).

The alternative — paired non-parametric tests on the eight class-level
F1 values — is actually MORE defensible for a Scopus reviewer in this
setting:

  * Non-parametric: no distributional assumption on per-class F1.
  * Honest: 8 classes is a small N; sign-test and Wilcoxon report
    exact discrete p-values.
  * Stronger claim: when one model wins on all 8 of 8 classes, the
    sign-test gives p = 1/256 (= 0.00391) — well below alpha = 0.05.

We compute:

  1. Mean and range of per-class XGB - DL_B1 gap (in p.p.).
  2. Two-sided sign test (binomial) on 'XGB beats DL_B1 per class'.
  3. Wilcoxon signed-rank test on the same 8 paired class-level F1s.
  4. Same comparisons for XGB - DL_A1.

Outputs a JSON for direct citation in the paper's Sec. 6 prose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

# ---- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
XGB_ML  = ROOT / "kaggle_paper" / "wandb_pull" / \
    "run-hnr5zkwn-tablemultilabel_classical_seeds-EduMzg" / "table" / \
    "multilabel_classical_seeds.table.json"
A1_JSON = ROOT / "kaggle_paper" / "kaggle_output_v12_final" / "runs" / \
    "A1_baseline.json"
B1_JSON = ROOT / "kaggle_paper" / "kaggle_output_v12_final" / "runs" / \
    "B1_pos_weight.json"
OUT     = ROOT / "kaggle_paper" / "wandb_pull" / "paired_class_test.json"

# ---- load per-class F1 vectors -------------------------------------------
xgb_tab = json.loads(XGB_ML.read_text(encoding="utf-8"))
xgb_cols = xgb_tab["columns"]
# layout: ['index', 'seed', 'macro_f1', 'access-control', 'arithmetic',
#          'bad-randomness', 'double-spending', 'locked-ether', 'other',
#          'reentrancy', 'unchecked-calls']
class_col_idx = [xgb_cols.index(c) for c in [
    "access-control", "arithmetic", "bad-randomness", "double-spending",
    "locked-ether", "other", "reentrancy", "unchecked-calls",
]]
# Seeds are deterministic — all three rows identical. Take first.
row0 = xgb_tab["data"][0]
xgb_per_class = np.array([row0[i] for i in class_col_idx])

# DL JSONs have explicit per-label F1 in the same canonical order.
LABEL_ORDER = [
    "access-control", "arithmetic", "bad-randomness", "double-spending",
    "locked-ether", "other", "reentrancy", "unchecked-calls",
]
a1 = json.loads(A1_JSON.read_text(encoding="utf-8"))
b1 = json.loads(B1_JSON.read_text(encoding="utf-8"))
a1_per_class = np.asarray(a1["f1_per_label"], dtype=float)
b1_per_class = np.asarray(b1["f1_per_label"], dtype=float)

assert xgb_per_class.shape == a1_per_class.shape == b1_per_class.shape == (8,)

# ---- per-pair stats ------------------------------------------------------
def paired_stats(name_a: str, name_b: str,
                 a: np.ndarray, b: np.ndarray) -> dict:
    """Compute paired statistics on per-class F1 vectors (length 8)."""
    diff = a - b
    n_a_wins = int(np.sum(diff > 0))
    n_b_wins = int(np.sum(diff < 0))
    n_ties = int(np.sum(diff == 0))

    sign_p = binomtest(n_a_wins, n=8, p=0.5, alternative="two-sided").pvalue
    sign_p_one = binomtest(n_a_wins, n=8, p=0.5,
                           alternative="greater").pvalue

    # Wilcoxon signed-rank (drop zeros mode 'wilcox' is robust here).
    if np.all(diff == 0):
        wil = {"W": None, "p": 1.0, "note": "all zeros"}
    else:
        w = wilcoxon(a, b, alternative="two-sided",
                     zero_method="wilcox", method="exact")
        wil = {"W": float(w.statistic), "p": float(w.pvalue),
               "method": "exact"}

    return {
        "comparison": f"{name_a} vs {name_b}",
        "macro_F1_a": float(np.mean(a)),
        "macro_F1_b": float(np.mean(b)),
        "macro_F1_gap_mean_pp": float((np.mean(a) - np.mean(b)) * 100),
        "per_class_gap_pp": (diff * 100).round(2).tolist(),
        "per_class_gap_min_pp": float(diff.min() * 100),
        "per_class_gap_max_pp": float(diff.max() * 100),
        "wins_a_of_8": n_a_wins,
        "wins_b_of_8": n_b_wins,
        "ties_of_8": n_ties,
        "sign_test_two_sided_p": float(sign_p),
        "sign_test_one_sided_p_a_better": float(sign_p_one),
        "wilcoxon_signed_rank": wil,
    }


vs_a1 = paired_stats("XGB_multilabel", "DL_A1_baseline",
                     xgb_per_class, a1_per_class)
vs_b1 = paired_stats("XGB_multilabel", "DL_B1_pos_weight",
                     xgb_per_class, b1_per_class)

out = {
    "labels": LABEL_ORDER,
    "xgb_per_class_f1": xgb_per_class.round(4).tolist(),
    "a1_per_class_f1":  a1_per_class.round(4).tolist(),
    "b1_per_class_f1":  b1_per_class.round(4).tolist(),
    "xgb_vs_a1": vs_a1,
    "xgb_vs_b1": vs_b1,
}

OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
               encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nwritten to: {OUT}")
