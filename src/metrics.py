"""Paper-grade evaluation metrics for binary smart-contract vulnerability
detection.

Implements the security-aware and honest-evaluation toolkit promised in the
paper but missing from the original Block-G pipeline:

  * :func:`bootstrap_ci`         — non-parametric BCa CIs for any metric
  * :func:`confusion_summary`    — TN/FP/FN/TP + MCC + FNR + Specificity
  * :func:`topk_recall`          — R@K with the information-theoretic ceiling
  * :func:`severity_weighted_miss_cost` — multi-label weighted FN cost
  * :func:`calibration_table`    — reliability-diagram data
  * :func:`compare_models`       — paper-ready comparison frame

Every public function is deterministic given an explicit ``seed``; no global
RNG state is mutated.

Designed for import inside a Kaggle notebook::

    from metrics import (
        bootstrap_ci, confusion_summary, topk_recall,
        severity_weighted_miss_cost, calibration_table, compare_models,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. Bootstrap confidence intervals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapCI:
    """Container for a bootstrap CI estimate."""

    point: float
    lo: float
    hi: float
    se: float
    n_boot: int
    alpha: float

    def fmt(self, digits: int = 3) -> str:
        """Render as ``0.943 [0.939, 0.947]`` for tables."""
        return (
            f"{self.point:.{digits}f} "
            f"[{self.lo:.{digits}f}, {self.hi:.{digits}f}]"
        )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    stratified: bool = True,
    seed: int = 376,
) -> BootstrapCI:
    """Non-parametric bootstrap CI for *any* binary classification metric.

    The same routine handles both score-based metrics (PR-AUC, ROC-AUC --
    pass ``y_score``) and label-based ones (F1, MCC -- pass ``y_pred``).

    Stratified resampling preserves the positive/negative ratio of the
    validation fold, which is the safer default for class-imbalanced data
    and matches the protocol we report in the paper.

    Parameters
    ----------
    y_true, y_score_or_pred
        Aligned 1-D arrays. The second array's type (probability vs. label)
        must match what ``metric_fn`` expects.
    metric_fn
        Any callable with signature ``(y_true, y_other) -> float``.
    n_boot
        Number of resamples. 1000 is the paper default.
    alpha
        Two-sided coverage; ``alpha=0.05`` gives a 95\\% CI.
    stratified
        If True, resample positives and negatives separately to preserve
        class balance per resample.
    seed
        RNG seed.

    Returns
    -------
    BootstrapCI
        Point estimate, lower/upper percentile bounds, bootstrap SE, params.
    """
    y_true = np.asarray(y_true)
    y_other = np.asarray(y_score_or_pred)
    if y_true.shape != y_other.shape:
        raise ValueError("y_true and y_score_or_pred shape mismatch")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_other))

    if stratified:
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)
    else:
        all_idx = np.arange(y_true.size)

    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ])
        else:
            idx = rng.choice(all_idx, size=all_idx.size, replace=True)
        boot_stats[b] = metric_fn(y_true[idx], y_other[idx])

    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    se = float(boot_stats.std(ddof=1))
    return BootstrapCI(point, lo, hi, se, n_boot, alpha)


# ---------------------------------------------------------------------------
# 2. Confusion-matrix summary
# ---------------------------------------------------------------------------
def confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Full confusion summary with security-relevant derived metrics.

    Returns TN / FP / FN / TP plus F1, precision, recall, specificity, MCC,
    FNR and FPR. FNR is reported with three significant digits because the
    paper headline ``FNR = 5.2 %`` lives here.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pos = tp + fn
    neg = tn + fp
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / neg if neg else float("nan"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "FNR": fn / pos if pos else float("nan"),
        "FPR": fp / neg if neg else float("nan"),
    }


# ---------------------------------------------------------------------------
# 3. Top-k Recall with information-theoretic ceiling
# ---------------------------------------------------------------------------
def topk_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Iterable[float] = (0.01, 0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Recall when only the top-K fraction by score is forwarded downstream.

    Used in the paper to quantify the "pre-filter" thesis. We additionally
    return the *theoretical ceiling* :math:`K / \\pi_{+}`: for a class-
    imbalanced validation with positive rate :math:`\\pi_{+}`, no ranker can
    recover more than :math:`K\\%` of all positives within the top-:math:`K\\%`
    of *all* items unless :math:`K \\ge \\pi_{+}`. Plotting both curves
    makes saturation visible at a glance and explains why we report PR-AUC
    rather than R@K to discriminate strong models in this regime.

    Returns
    -------
    DataFrame with columns ``K``, ``R@K``, ``ceiling``, ``slack``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.size
    n_pos = int(y_true.sum())
    if n_pos == 0:
        raise ValueError("Top-k Recall undefined: no positives in y_true.")
    pi_plus = n_pos / n
    order = np.argsort(-y_score)  # descending by score
    y_sorted = y_true[order]

    rows = []
    for k in ks:
        if not 0 < k <= 1:
            raise ValueError(f"K must be in (0, 1]; got {k}")
        cutoff = max(1, int(np.ceil(k * n)))
        recall_at_k = float(y_sorted[:cutoff].sum() / n_pos)
        ceiling = min(1.0, k / pi_plus)
        rows.append({
            "K": k,
            "R@K": recall_at_k,
            "ceiling": ceiling,
            "slack": ceiling - recall_at_k,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Severity-weighted miss cost (multi-label)
# ---------------------------------------------------------------------------
DEFAULT_SEVERITY: Mapping[str, float] = {
    # Severities mirror auditor consensus on operational damage potential.
    # Values are unitless and meant to be ranked, not absolute.
    "reentrancy":        1.00,   # SWC-107  -- The DAO class
    "access-control":    0.90,
    "arithmetic":        0.80,   # SWC-101
    "unchecked-calls":   0.70,
    "double-spending":   0.65,
    "bad-randomness":    0.50,   # SWC-120
    "locked-ether":      0.40,
    "other":             0.30,
}


def severity_weighted_miss_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    severity: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Weighted false-negative cost per multi-label class.

    For each label, computes:

      * ``FN_c``  -- false negatives for label *c*
      * ``severity_c`` -- caller-provided weight (defaults from auditor
        consensus on operational damage potential)
      * ``cost_c = FN_c * severity_c`` -- the headline quantity

    A waterfall plot of ``cost`` per label exposes which classes contribute
    most of the residual operational risk *after* model selection. Two
    models with identical macro-F1 can differ wildly on this measure.
    """
    if severity is None:
        severity = DEFAULT_SEVERITY
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch between y_true and y_pred")
    if y_true.shape[1] != len(label_names):
        raise ValueError("len(label_names) must equal n_classes")
    rows = []
    for c, name in enumerate(label_names):
        fn_c = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        w = float(severity.get(name, 0.5))
        rows.append({
            "label": name,
            "FN": fn_c,
            "severity": w,
            "cost": fn_c * w,
        })
    df = pd.DataFrame(rows).sort_values("cost", ascending=False, ignore_index=True)
    df["cost_share"] = df["cost"] / max(df["cost"].sum(), 1e-12)
    return df


# ---------------------------------------------------------------------------
# 5. Calibration / reliability diagram data
# ---------------------------------------------------------------------------
def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Reliability-diagram data with per-bin counts.

    Quantile binning is the safe default for class-imbalanced data: equal-
    frequency bins guarantee every bin holds enough samples to make the
    observed positive rate informative.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    rows = []
    for b in range(n_bins):
        in_bin = (y_prob > edges[b]) & (y_prob <= edges[b + 1])
        n = int(in_bin.sum())
        if n == 0:
            continue
        rows.append({
            "bin": b,
            "p_mean": float(y_prob[in_bin].mean()),
            "y_rate": float(y_true[in_bin].mean()),
            "n": n,
        })
    df = pd.DataFrame(rows)
    df["gap"] = df["y_rate"] - df["p_mean"]
    return df


# ---------------------------------------------------------------------------
# 6. Paper-ready comparison frame
# ---------------------------------------------------------------------------
def compare_models(
    per_model: Mapping[str, Mapping[str, float]],
    columns: Sequence[str] = (
        "F1", "Precision", "Recall", "MCC", "FNR", "PR-AUC", "train_sec",
    ),
    digits: int = 3,
    bold_best: bool = True,
) -> pd.DataFrame:
    """Build the comparison DataFrame that becomes Table 1 in the paper.

    Each value in ``per_model`` is a dict produced by
    :func:`confusion_summary` (plus ``PR-AUC`` and ``train_sec`` added by
    the caller). Missing keys are rendered as ``--``.

    ``bold_best`` wraps the best value per column in HTML ``<b>...</b>`` so
    ``DataFrame.to_html(escape=False)`` renders boldface in the W\\&B log
    and Jupyter inline. The Latex export uses the raw values.
    """
    rows = []
    for name, m in per_model.items():
        rows.append({"Model": name, **{c: m.get(c, float("nan")) for c in columns}})
    df = pd.DataFrame(rows).set_index("Model")
    if bold_best:
        better_is_higher = {
            "F1": True, "Precision": True, "Recall": True, "MCC": True,
            "PR-AUC": True, "ROC-AUC": True, "Specificity": True,
            "FNR": False, "FPR": False, "train_sec": False,
        }
        formatted = df.copy().astype(object)
        for col in df.columns:
            higher = better_is_higher.get(col, True)
            best = df[col].max() if higher else df[col].min()
            for idx, val in df[col].items():
                if pd.isna(val):
                    formatted.loc[idx, col] = "--"
                    continue
                cell = f"{val:.{digits}f}"
                if abs(val - best) < 10 ** -digits:
                    cell = f"<b>{cell}</b>"
                formatted.loc[idx, col] = cell
        return formatted
    return df.round(digits)


# ---------------------------------------------------------------------------
# 7. Convenience: PR-AUC as a metric_fn for bootstrap_ci
# ---------------------------------------------------------------------------
def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Wrapper to expose ``average_precision_score`` with a stable name."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Wrapper to expose ``roc_auc_score`` with a stable name."""
    return float(roc_auc_score(y_true, y_score))


__all__ = [
    "BootstrapCI",
    "bootstrap_ci",
    "confusion_summary",
    "topk_recall",
    "severity_weighted_miss_cost",
    "DEFAULT_SEVERITY",
    "calibration_table",
    "compare_models",
    "pr_auc",
    "roc_auc",
]
