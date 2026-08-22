"""Full 14-run DL ablation analysis from real W&B Colab L4 run hk57ndy1.

Aggregates per-class F1 from 10 Conv-Transformer configurations
(A1, B1-B5, C1-C4) and 3 XGBoost multi-label seeds. Computes:
  * Paired class-level sign test + Wilcoxon for XGB vs each DL config.
  * Best-of-DL summary for the paper.
  * Publication-quality heatmap (13 columns x 8 SWC classes).

Writes:
  - kaggle_paper/wandb_pull/full_ablation_stats.json (citable numbers)
  - icicpe_paper/figures/perlabel_f1_heatmap.pdf and .png (vector)
"""
from __future__ import annotations
import json, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DL_DIR = ROOT / "kaggle_paper" / "wandb_pull" / "dl_ablation"
OUT_JSON = ROOT / "kaggle_paper" / "wandb_pull" / "full_ablation_stats.json"
FIG_PDF = ROOT / "icicpe_paper" / "figures" / "perlabel_f1_heatmap.pdf"
FIG_PNG = ROOT / "icicpe_paper" / "figures" / "perlabel_f1_heatmap.png"

LABELS_ORDER = [
    "access-control", "arithmetic", "bad-randomness", "double-spending",
    "locked-ether", "other", "reentrancy", "unchecked-calls",
]

# ---- Load per-class F1 from each downloaded table -------------------------
def _exp_name(dir_name: str) -> str:
    # extract experiment name from W&B artifact dir name
    # e.g. run-hk57ndy1-block_cC2_dmodel_256per_label_f1_table-qY1d-Q
    s = dir_name
    s = s.replace("run-hk57ndy1-", "")
    s = s.split("per_label_f1_table")[0]
    for prefix in ("block_a", "block_b", "block_c", "block_f", "block_ml"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _family(name: str) -> str:
    """Classify experiment by family: classical | dl."""
    if any(s in name for s in ("XGB", "LogReg", "RandomForest", "ML0",
                                "ML1", "ML2")):
        return "classical"
    return "dl"

def _load_table(p: Path) -> tuple[str, np.ndarray]:
    """Return (experiment_name, per_label_f1_vector_in_LABELS_ORDER)."""
    name = _exp_name(p.name)
    inner = list(p.glob("**/per_label_f1_table.table.json")) or \
            list(p.glob("**/*.table.json"))
    if not inner:
        raise FileNotFoundError(p)
    d = json.loads(inner[0].read_text(encoding="utf-8"))
    cols = d["columns"]
    rows = d["data"]
    # Tables can be in two shapes:
    # A) row-per-label: cols = ['label', 'F1'] -> 8 rows
    # B) row-per-row:   cols = ['index','label1','label2',...] -> 1 row
    if "label" in cols and any("f1" in c.lower() for c in cols):
        # shape A
        label_idx = cols.index("label")
        f1_idx = next(i for i, c in enumerate(cols) if "f1" in c.lower())
        d_map = {r[label_idx]: r[f1_idx] for r in rows}
        vec = np.array([d_map[l] for l in LABELS_ORDER], dtype=float)
    else:
        # shape B
        col_idx = [cols.index(l) for l in LABELS_ORDER]
        vec = np.array([rows[0][i] for i in col_idx], dtype=float)
    return name, vec


per_class = {}
for d in sorted(DL_DIR.iterdir()):
    if d.is_dir():
        name, vec = _load_table(d)
        per_class[name] = vec

# Sanity print
print(f"Loaded {len(per_class)} experiments:")
for k in sorted(per_class):
    print(f"  {k:35s}  macro_f1={per_class[k].mean():.4f}  per_class_min={per_class[k].min():.4f}")

# ---- Build DataFrame ------------------------------------------------------
# Categorize properly: classical (XGB + RF + LogReg multi-label) vs DL.
classical_keys = sorted([k for k in per_class if _family(k) == "classical"])
dl_keys        = sorted([k for k in per_class if _family(k) == "dl"])

# Among classical: XGB-only family (3 seeds + ML2 duplicate). All deterministic
# so we average — values are identical anyway.
xgb_keys = [k for k in classical_keys if "XGB" in k]
rf_keys  = [k for k in classical_keys if "RandomForest" in k or "ML1" in k]
logreg_keys = [k for k in classical_keys if "LogReg" in k]

xgb_mat = np.stack([per_class[k] for k in xgb_keys])
xgb_vec = xgb_mat.mean(axis=0)
xgb_var = xgb_mat.std(axis=0)
rf_vec  = per_class[rf_keys[0]] if rf_keys else None
lr_vec  = per_class[logreg_keys[0]] if logreg_keys else None
print(f"\nclassical: XGB={xgb_vec.mean():.4f}  RF={rf_vec.mean() if rf_vec is not None else '-':.4f}  LogReg={lr_vec.mean() if lr_vec is not None else '-':.4f}")
print(f"dl_keys ({len(dl_keys)}): {dl_keys}")

# ---- Pairwise non-parametric tests: XGB vs each DL ------------------------
def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    wins_a = int(np.sum(diff > 0))
    wins_b = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    sign_2 = binomtest(wins_a, n=8, p=0.5, alternative="two-sided").pvalue
    sign_1 = binomtest(wins_a, n=8, p=0.5, alternative="greater").pvalue
    if np.all(diff == 0):
        wil = {"W": None, "p": 1.0}
    else:
        w = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox",
                     method="exact")
        wil = {"W": float(w.statistic), "p": float(w.pvalue)}
    return {
        "wins_a_of_8": wins_a, "wins_b_of_8": wins_b, "ties": ties,
        "mean_gap_pp": float((a.mean() - b.mean()) * 100),
        "min_gap_pp": float(diff.min() * 100),
        "max_gap_pp": float(diff.max() * 100),
        "sign_test_two_sided_p": float(sign_2),
        "sign_test_one_sided_p_a_better": float(sign_1),
        "wilcoxon": wil,
    }


stats = {}
for k in dl_keys:
    stats[k] = {
        "dl_macro_f1": float(per_class[k].mean()),
        "xgb_macro_f1": float(xgb_vec.mean()),
        **paired_test(xgb_vec, per_class[k]),
    }

# Best DL by macro F1
best_dl = max(dl_keys, key=lambda k: per_class[k].mean())
print(f"\nBEST DL: {best_dl} macro_f1={per_class[best_dl].mean():.4f}")
print(f"XGB     : macro_f1={xgb_vec.mean():.4f}")
print(f"Gap     : {(xgb_vec.mean() - per_class[best_dl].mean())*100:.2f} p.p.")
print(f"XGB beats best DL on {int(np.sum(xgb_vec > per_class[best_dl]))}/8 classes")
print(f"Sign-test p={stats[best_dl]['sign_test_two_sided_p']:.4f}")
print(f"Wilcoxon  p={stats[best_dl]['wilcoxon']['p']:.4f}")

# Summary
out = {
    "labels": LABELS_ORDER,
    "experiments_loaded": list(per_class.keys()),
    "xgb_mean_per_class": xgb_vec.round(4).tolist(),
    "xgb_seed_variance_per_class": xgb_var.round(6).tolist(),
    "best_dl_name": best_dl,
    "best_dl_per_class": per_class[best_dl].round(4).tolist(),
    "xgb_macro_f1": float(xgb_vec.mean()),
    "best_dl_macro_f1": float(per_class[best_dl].mean()),
    "stats_xgb_vs_each_dl": stats,
}
OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
print(f"\nwrote {OUT_JSON}")

# ---- Heatmap (publication quality) ----------------------------------------
# Column order: LogReg ML0, RF ML1, XGB ML2 + seeds, then DL blocks A/B/C
ordered_cols = (
    [k for k in classical_keys if "LogReg" in k or "ML0" in k]
    + [k for k in classical_keys if "RandomForest" in k or "ML1" in k]
    + sorted([k for k in classical_keys if "XGB" in k])
    + [k for k in dl_keys if k.startswith("A1")]
    + sorted([k for k in dl_keys if k.startswith("B")])
    + sorted([k for k in dl_keys if k.startswith("C")])
)

# Pretty column labels
pretty = {
    "ML0_LogReg_balanced":  "ML0 LogReg",
    "ML1_RandomForest":     "ML1 RF",
    "ML2_XGBoost":          "ML2 XGB",
    "F1_XGB_seed42":      "XGB (s42)",
    "F1_XGB_seed123":     "XGB (s123)",
    "F1_XGB_seed2024":    "XGB (s2024)",
    "A1_baseline":        "A1 BCE",
    "B1_pos_weight":      "B1 pos_w",
    "B2_focal_g2":        "B2 focal",
    "B3_focal_pos_weight":"B3 focal+pw",
    "B4_asymmetric":      "B4 asym",
    "B5_threshold_tuning_on_best": "B5 thr-tune",
    "C1_transformer_4layers": "C1 4L-Tr",
    "C2_dmodel_256":      "C2 d=256",
    "C3_pure_cnn":        "C3 CNN",
    "C4_pure_transformer":"C4 pure-Tr",
}
col_labels = [pretty.get(k, k) for k in ordered_cols]
data = np.stack([per_class[k] for k in ordered_cols], axis=1)
df = pd.DataFrame(data, index=LABELS_ORDER, columns=col_labels)

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.4, 4.0))
sns.heatmap(df, annot=True, fmt=".3f", cmap="rocket_r",
            vmin=0.4, vmax=0.95, ax=ax,
            cbar_kws={"label": "$F_1$"}, linewidths=0.35,
            linecolor="white", annot_kws={"fontsize": 7.5})

# Highlight column group separators (classical | block-A | block-B | block-C)
n_classical = sum(1 for k in ordered_cols
                  if _family(k) == "classical")
n_a = sum(1 for k in ordered_cols if k.startswith("A1"))
n_b = sum(1 for k in ordered_cols if k.startswith("B"))
for sep in (n_classical, n_classical + n_a, n_classical + n_a + n_b):
    ax.axvline(sep, color="black", lw=1.4)

ax.set_xlabel("")
ax.set_ylabel("SWC vulnerability class")
ax.set_title("Per-class $F_1$: classical ML (left) vs.\\ "
             "Conv-Transformer 10-config ablation (right)",
             fontsize=10, pad=8)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
fig.savefig(FIG_PDF, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=180, bbox_inches="tight")
print(f"\nwrote {FIG_PDF}")
print(f"wrote {FIG_PNG}")
