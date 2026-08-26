# -*- coding: utf-8 -*-
"""Paper v2 — Phase 4: single-run statistics and the heatmap, honestly framed.

Inputs (all from the SAME v2 protocol — the cross-run splice that poisoned v1
is structurally impossible here):
    results_classical/binary_results_v2.csv + {model}.json  (Phase 2, test set)
    results_classical/paired_delta_v2.json
    runs_v2/{config}.json x10                               (Phase 3, test set)
    multilabel classical: ML*.json if Phase 2b produced them, else the
    multilabel XGB is trained here on v2 train and evaluated on v2 test.

Honesty rules encoded below:
  * n=8 paired tests: report the exact p, and SAY the floor is 2/256=0.0078 —
    a clean sweep is "8/8", nothing stronger. No sign+Wilcoxon double-counting:
    both are reported once, labelled as the same evidence.
  * 20-comparison family: Holm correction applied and reported.
  * Model-level 10/10 claim: stated as descriptive (configs share one dataset;
    independence does not hold, so no binomial p is attached).
Outputs: figures/perlabel_f1_heatmap_v2.{pdf,png}, stats_v2.json,
         a printed paper-numbers block for Phase 5.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper\v2")
RUNS = BASE / "runs_v2"
RESC = BASE / "results_classical"
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)

LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
FEATURES = json.load(open(BASE / "feature_columns.json", encoding="utf-8"))

# ── multilabel classical on v2 (train->test, untuned, as in v1) ─────────────
ML_CACHE = RESC / "multilabel_v2.json"
if ML_CACHE.exists():
    ml = json.load(open(ML_CACHE, encoding="utf-8"))
else:
    print("training multilabel classical (one-time)...")
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    import xgboost as xgb

    def load(name):
        df = pd.read_parquet(BASE / f"{name}_v2.parquet",
                             columns=FEATURES + LABELS)
        return df[FEATURES].to_numpy("float32"), df[LABELS].to_numpy("int8")

    Xtr, Ytr = load("train")
    Xte, Yte = load("test")
    ml = {}
    for name, ctor in {
        "ML0_LogReg_balanced": lambda: MultiOutputClassifier(
            LogisticRegression(max_iter=1500, class_weight="balanced"), n_jobs=-1),
        "ML1_RandomForest": lambda: RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42),
        "ML2_XGBoost": lambda: MultiOutputClassifier(
            xgb.XGBClassifier(n_estimators=400, max_depth=8, n_jobs=-1,
                              random_state=42, eval_metric="logloss"), n_jobs=1),
    }.items():
        m = ctor()
        m.fit(Xtr, Ytr)
        P = m.predict(Xte)
        per = [float(f1_score(Yte[:, i], P[:, i])) for i in range(8)]
        ml[name] = {"macro_f1": float(np.mean(per)), "f1_per_label": per}
        print(f"  {name:22s} macro={ml[name]['macro_f1']:.4f}")
    json.dump(ml, open(ML_CACHE, "w"), indent=2)

# ── DL results ──────────────────────────────────────────────────────────────
EXPECTED = ["A1_baseline", "B1_pos_weight", "B2_focal_g2",
            "B3_focal_pos_weight", "B4_asymmetric",
            "B5_threshold_tuning_on_best", "C1_transformer_4layers",
            "C2_dmodel_256", "C3_pure_cnn", "C4_pure_transformer"]


def _canon(name):
    """Fold a truncated config name onto its registry entry.

    The trainer sometimes emits a shortened name (observed: the ops-drop key
    "B5_threshold_tuning" for registry entry "B5_threshold_tuning_on_best").
    Reading the directory naively then counts one config twice, which inflates
    the comparison family and, through Holm, tightens the threshold for every
    other config. Fold first, then reject any second file for a name already
    taken.
    """
    for e in EXPECTED:
        if e == name or e.startswith(name) or name.startswith(e):
            return e
    return name


dl = {}
for p in sorted(RUNS.glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if "f1_per_label" in d:
        # dl_pipeline writes the held-out score as macro_f1_external; older
        # caches used macro_f1. Fall back to the mean of the per-label vector
        # so a config is never silently treated as zero.
        mf = d.get("macro_f1_external", d.get("macro_f1"))
        if mf is None:
            mf = float(np.mean(d["f1_per_label"]))
        cname = _canon(p.stem)
        if cname in dl:
            print(f"  duplicate ignored: {p.stem} -> {cname}")
            continue
        dl[cname] = {"macro_f1": float(mf),
                      "f1_per_label": d["f1_per_label"],
                      "train_time_sec": d.get("train_time_sec"),
                      "params_m": d.get("model_params_M")}
print(f"\nDL configs available: {len(dl)}/10 -> {sorted(dl)}")
if len(dl) < 10:
    print("!! fewer than 10 DL configs — rerun Phase 3 before final paper numbers")

# ── paired stats: XGB vs each DL, one run, honest framing ───────────────────
from scipy.stats import binomtest, wilcoxon
xgbv = np.array(ml["ML2_XGBoost"]["f1_per_label"])
stats = {"floor_note": "n=8 paired tests: min two-sided p = 2/256 = 0.0078; "
                       "sign test and Wilcoxon on the same 8 numbers are one "
                       "piece of evidence, not two",
         "comparisons": {}}
pvals = []
for k, v in sorted(dl.items()):
    c = np.array(v["f1_per_label"])
    wins = int((xgbv > c).sum())
    p_sign = float(binomtest(wins, 8, 0.5, alternative="two-sided").pvalue)
    try:
        w = wilcoxon(xgbv, c, alternative="two-sided",
                     zero_method="wilcox", method="exact")
        p_w = float(w.pvalue)
    except ValueError:
        p_w = 1.0
    stats["comparisons"][k] = {
        "dl_macro": float(c.mean()), "xgb_macro": float(xgbv.mean()),
        "gap_pp": float(100 * (xgbv.mean() - c.mean())),
        "xgb_wins_of_8": wins, "p_sign": p_sign, "p_wilcoxon": p_w}
    pvals.append((k, p_sign))

# Holm-Bonferroni over the sign tests, with the STEP-DOWN rule.
#
# Holm is sequential: sort ascending, compare p_(i) against alpha/(m-i), and
# STOP at the first failure -- every remaining hypothesis is retained too.
# Testing each p against its own threshold independently (the earlier bug
# here) lets later hypotheses "pass" on the looser thresholds that only
# become available once the earlier, stricter ones have been rejected. With
# ten identical p = 0.0078125 it marked six of ten significant when the
# correct answer is none: the smallest p already exceeds alpha/10 = 0.005.
#
# That is a property of the design, not a weak effect. A sign test on n=8
# classes cannot produce a p below 2/256 = 0.0078, so no comparison in a
# family of ten can clear Holm's first threshold. The ceiling is the class
# count; the effect itself (8/8 in every configuration) is not in doubt.
pv_sorted = sorted(pvals, key=lambda t: t[1])
m = len(pv_sorted)
ALPHA = 0.05
still_rejecting = True
for rank, (k, p) in enumerate(pv_sorted):
    thr = ALPHA / (m - rank)
    if still_rejecting and p > thr:
        still_rejecting = False
    stats["comparisons"][k]["holm_alpha"] = thr
    stats["comparisons"][k]["significant_after_holm"] = bool(still_rejecting)

stats["holm"] = {
    "family_size": m,
    "alpha": ALPHA,
    "first_threshold": ALPHA / m,
    "smallest_p": pv_sorted[0][1] if pv_sorted else None,
    "any_significant": any(
        stats["comparisons"][k]["significant_after_holm"] for k, _ in pv_sorted),
    "note": ("Holm step-down. With n=8 classes the sign test floor is "
             "2/256=0.0078125, above alpha/m for any family of 7 or more, "
             "so family-wise significance is unreachable by construction "
             "here -- a limit of the class count, not evidence of a weak "
             "effect."),
}

n_losing = sum(1 for v in dl.values()
               if v["macro_f1"] is not None and v["macro_f1"] < xgbv.mean())
stats["model_level"] = {
    "dl_configs_below_xgb_macro": n_losing, "of": len(dl),
    "note": "descriptive only — configs share one dataset, independence "
            "does not hold, no binomial p attached"}

json.dump({"classical_multilabel": ml, "dl": dl, "stats": stats},
          open(BASE / "stats_v2.json", "w"), indent=2)

# ── heatmap (same layout as v1 figure, single-run data) ─────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pretty = {"ML0_LogReg_balanced": "ML0 LogReg", "ML1_RandomForest": "ML1 RF",
          "ML2_XGBoost": "ML2 XGB", "A1_baseline": "A1 BCE",
          "B1_pos_weight": "B1 pos_w", "B2_focal_g2": "B2 focal",
          "B3_focal_pos_weight": "B3 focal+pw", "B4_asymmetric": "B4 asym",
          "B5_threshold_tuning_on_best": "B5 thr-tune",
          "C1_transformer_4layers": "C1 4L-Tr", "C2_dmodel_256": "C2 d=256",
          "C3_pure_cnn": "C3 CNN", "C4_pure_transformer": "C4 pure-Tr"}
ordered = (["ML0_LogReg_balanced", "ML1_RandomForest", "ML2_XGBoost"]
           + [k for k in sorted(dl) if k.startswith("A")]
           + [k for k in sorted(dl) if k.startswith("B")]
           + [k for k in sorted(dl) if k.startswith("C")])
allv = {**{k: v["f1_per_label"] for k, v in ml.items()},
        **{k: v["f1_per_label"] for k, v in dl.items()}}
data = np.stack([allv[k] for k in ordered], axis=1)
df = pd.DataFrame(data, index=LABELS,
                  columns=[pretty.get(k, k) for k in ordered])

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.4, 4.0))
sns.heatmap(df, annot=True, fmt=".3f", cmap="rocket_r", vmin=0.4, vmax=0.95,
            ax=ax, cbar_kws={"label": "$F_1$"}, linewidths=0.35,
            linecolor="white", annot_kws={"fontsize": 7.5})
n_a = sum(1 for k in ordered if k.startswith("A"))
n_b = sum(1 for k in ordered if k.startswith("B"))
for sep in (3, 3 + n_a, 3 + n_a + n_b):
    ax.axvline(sep, color="black", lw=1.4)
ax.set_xlabel("")
ax.set_ylabel("SWC vulnerability class")
# No in-figure title: the caption states the content, and the LaTeX
# inter-word escape this title used renders as a literal backslash here.
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
fig.savefig(FIGS / "perlabel_f1_heatmap_v2.pdf", bbox_inches="tight")
fig.savefig(FIGS / "perlabel_f1_heatmap_v2.png", dpi=180, bbox_inches="tight")

# ── paper-numbers block ─────────────────────────────────────────────────────
print("\n===== NUMBERS FOR THE PAPER (v2) =====")
if (RESC / "binary_results_v2.csv").exists():
    b = pd.read_csv(RESC / "binary_results_v2.csv")
    print(b.to_string(index=False))
print(f"\nmultilabel: XGB {xgbv.mean():.4f} | RF "
      f"{np.mean(ml['ML1_RandomForest']['f1_per_label']):.4f} | LogReg "
      f"{np.mean(ml['ML0_LogReg_balanced']['f1_per_label']):.4f}")
if dl:
    best = max(dl, key=lambda k: dl[k]["macro_f1"] or 0)
    c = stats["comparisons"][best]
    print(f"best DL {best}: {dl[best]['macro_f1']:.4f} | gap {c['gap_pp']:.2f}pp"
          f" | XGB wins {c['xgb_wins_of_8']}/8 | p_sign {c['p_sign']:.4f}"
          f" (floor 0.0078) | Holm-significant: {c['significant_after_holm']}")
print("figures/perlabel_f1_heatmap_v2.pdf written")
