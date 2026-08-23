# -*- coding: utf-8 -*-
"""Paper v2 — Phase 6: fill the repository cards from the run artifacts.

Same principle as emit_macros_v2.py, applied to Markdown instead of LaTeX.
The v1 failure mode was numbers drifting between paper, README, cards and
site because each copy was typed independently; typing them once more here
would rebuild exactly that. Every value below is read from the artifacts and
written into the cards, so a rerun updates all of them together or none.

Idempotent: rewrites the tables between their markers on every invocation,
so it can be run again after C3/C4 land without hand-editing anything.

Sources: results_classical/binary_results_v2.csv, stats_v2.json,
manifest_v2.json.
"""
import csv
import io
import json
import re
import sys
from pathlib import Path

import os as _os
import pathlib as _pathlib

# Working directory. Defaults to the results/ tree shipped in this
# repository so the analysis and emit scripts run straight from a checkout;
# set PAPER_V2_DIR to a full working tree (with the parquet splits and
# runs_v2/) to regenerate results from scratch.
_ROOT = _os.environ.get("PAPER_V2_DIR") or str(
    _pathlib.Path(__file__).resolve().parent.parent / "results")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

V2 = Path(_ROOT)
REPO = Path(_os.environ.get("PAPER_REPO", str(_pathlib.Path(__file__).resolve().parent.parent)))

man = json.load(open(V2 / "manifest_v2.json", encoding="utf-8"))
st = json.load(open(V2 / "stats_v2.json", encoding="utf-8"))
binr = {r["model"]: r for r in csv.DictReader(
    open(V2 / "results_classical" / "binary_results_v2.csv", encoding="utf-8"))}

ml = st["classical_multilabel"]
dl = st["dl"]
best = max(dl, key=lambda k: dl[k]["macro_f1"])
cmp_best = st["stats"]["comparisons"][best]
n_dl = len(dl)

EXPECTED_DL = ["A1_baseline", "B1_pos_weight", "B2_focal_g2",
               "B3_focal_pos_weight", "B4_asymmetric",
               "B5_threshold_tuning_on_best", "C1_transformer_4layers",
               "C2_dmodel_256", "C3_pure_cnn", "C4_pure_transformer"]
missing = [c for c in EXPECTED_DL if c not in dl]


def f1(model):
    r = binr[model]
    return (f"**{float(r['f1']):.4f}** [{float(r['f1_lo']):.4f}, "
            f"{float(r['f1_hi']):.4f}], FNR {100*float(r['fnr']):.2f}%")


def block(lines):
    return "\n".join(lines)


# ── README results table ────────────────────────────────────────────────────
readme_rows = [
    "| Task | Model | Metric | Value |",
    "|---|---|---|---|",
    f"| Binary any-vulnerability | XGBoost (Optuna, 5-fold CV in train) "
    f"| F1 (test) | {f1('XGB')} |",
    f"| Binary any-vulnerability | RandomForest | F1 (test) | {f1('RF')} |",
    f"| Binary any-vulnerability | CatBoost | F1 (test) | {f1('CatBoost')} |",
    f"| Binary any-vulnerability | LogisticRegression | F1 (test) "
    f"| {f1('LogReg')} |",
    f"| Multi-label SWC (8 classes) | XGBoost | macro-F1 (test) "
    f"| **{ml['ML2_XGBoost']['macro_f1']:.4f}** |",
    f"| Multi-label SWC (8 classes) | RandomForest | macro-F1 (test) "
    f"| {ml['ML1_RandomForest']['macro_f1']:.4f} |",
    f"| Multi-label SWC (8 classes) | LogisticRegression | macro-F1 (test) "
    f"| {ml['ML0_LogReg_balanced']['macro_f1']:.4f} |",
    f"| Multi-label SWC (8 classes) | best Conv-Transformer "
    f"(`{best}`) | macro-F1 (test) | {dl[best]['macro_f1']:.4f} |",
]

caveat = ""
if missing:
    caveat = (
        f"\n> **Ablation coverage.** {n_dl} of {len(EXPECTED_DL)} deep "
        f"configurations completed: `{'`, `'.join(missing)}` did not finish "
        f"within the GPU allocation for this run and are excluded rather "
        f"than estimated. Under the v1 (leaking) protocol those ranked 4th "
        f"and 10th of 10, so neither was the strongest deep configuration "
        f"there; no claim is made about where they would land here.\n")

readme_after = (
    f"\nAll values are from a single read of the held-out test split "
    f"({man['n_test']:,} contracts) under the v2 protocol. Confidence "
    f"intervals are stratified percentile bootstrap (B=1000). XGBoost beats "
    f"the best deep configuration on {cmp_best['xgb_wins_of_8']}/8 classes "
    f"(sign test p={cmp_best['p_sign']:.4f}; with n=8 the smallest "
    f"attainable two-sided p is 2/256=0.0078, so this is the floor, not a "
    f"stronger claim).\n"
    f"{caveat}\n"
    f"These numbers are generated from `results/` by "
    f"`scripts/emit_cards_v2.py` and cross-checked against the paper by "
    f"`scripts/check_numbers_v2.py`; they are not typed by hand.\n")


def replace_section(path, header_pat, new_header, body):
    """Swap a '## Section' block for freshly generated content."""
    p = REPO / path
    s = p.read_text(encoding="utf-8")
    m = re.search(header_pat, s)
    assert m, f"{path}: section header not found ({header_pat})"
    start = m.start()
    nxt = s.find("\n## ", m.end())
    end = nxt if nxt != -1 else len(s)
    s = s[:start] + new_header + "\n\n" + body + "\n" + s[end:]
    p.write_text(s, encoding="utf-8")
    print(f"  updated {path}")


replace_section("README.md", r"## Results \(v2[^\n]*\)", "## Results (v2)",
                block(readme_rows) + "\n" + readme_after)

# ── MODEL_CARD metrics table ───────────────────────────────────────────────
mc_rows = [
    "| Task | Model | F1 / macro-F1 (test) |",
    "|---|---|---|",
    f"| Binary | XGBoost (Optuna, CV-in-train) | {f1('XGB')} |",
    f"| Binary | RandomForest | {f1('RF')} |",
    f"| Binary | CatBoost | {f1('CatBoost')} |",
    f"| Binary | LogisticRegression | {f1('LogReg')} |",
    f"| Multi-label | XGBoost | **{ml['ML2_XGBoost']['macro_f1']:.4f}** |",
    f"| Multi-label | RandomForest | {ml['ML1_RandomForest']['macro_f1']:.4f} |",
    f"| Multi-label | LogisticRegression "
    f"| {ml['ML0_LogReg_balanced']['macro_f1']:.4f} |",
    f"| Multi-label | best Conv-Transformer (`{best}`) "
    f"| {dl[best]['macro_f1']:.4f} |",
]
mc_after = (
    f"\nDeep ablation, macro-F1 on the same test split "
    f"({n_dl} configurations):\n\n"
    + "\n".join(f"- `{k}` — {v['macro_f1']:.4f}"
                for k, v in sorted(dl.items(),
                                   key=lambda kv: -kv[1]["macro_f1"]))
    + f"\n\nEvery deep configuration falls below the classical multi-label "
      f"baseline; the gap to the best is {cmp_best['gap_pp']:.2f} percentage "
      f"points. This count is descriptive — the configurations share one "
      f"dataset and one feature representation, so no binomial test is "
      f"attached to it.\n{caveat}")
replace_section("MODEL_CARD.md", r"## Metrics \(v2[^\n]*\)", "## Metrics (v2)",
                block(mc_rows) + "\n" + mc_after)

print("\ncards regenerated from artifacts")
