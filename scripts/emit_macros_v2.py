# -*- coding: utf-8 -*-
"""Paper v2 — Phase 5 spine: emit every volatile number as a LaTeX macro.

Run after analyze_v2.py. main_v2.tex does \input{numbers_v2} and then writes
\vBXGBFone instead of "0.948", so a number physically cannot drift between
artifacts and prose — the v1 failure mode (0.751 in README vs 0.775 in the
model card vs a figure showing neither) becomes impossible by construction.

Reads: manifest_v2.json, results_classical/binary_results_v2.csv,
       results_classical/paired_delta_v2.json, stats_v2.json
Writes: numbers_v2.tex
Missing inputs are skipped with a warning, so this is safe to run mid-pipeline.
"""
import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper\v2")
RESC = BASE / "results_classical"
BS = chr(92)  # backslash, kept out of literals to survive shell round-trips


def macro(name: str, value: str) -> str:
    return BS + "newcommand{" + BS + name + "}{" + str(value) + "}"


def thousands(n: int) -> str:
    """LaTeX-safe thousands separator: 112{,}467 keeps math-mode spacing."""
    return f"{n:,}".replace(",", "{,}")


out = []

man_p = BASE / "manifest_v2.json"
if man_p.exists():
    man = json.load(open(man_p, encoding="utf-8"))
    corpus = man["steps"][3]["rows"]
    out += [
        macro("vCorpus", thousands(corpus)),
        macro("vTrain", thousands(man["n_train"])),
        macro("vVal", thousands(man["n_val"])),
        macro("vTest", thousands(man["n_test"])),
        macro("vDupsBc", str(man["dropped_bytecode_dups"])),
        macro("vDupsFeat", thousands(man["dropped_feature_dups"])),
        macro("vPosRate", f"{man['positive_rate']:.3f}"),
        macro("vSeed", str(man["seed"])),
        macro("vNFeat", str(man["n_features"])),
    ]
else:
    print("! manifest_v2.json missing")

bcsv = RESC / "binary_results_v2.csv"
if bcsv.exists():
    for r in csv.DictReader(open(bcsv, encoding="utf-8")):
        n = r["model"].replace("LogReg", "LR").replace("CatBoost", "Cat")
        out += [
            macro(f"vB{n}Fone", f"{float(r['f1']):.3f}"),
            macro(f"vB{n}Lo", f"{float(r['f1_lo']):.3f}"),
            macro(f"vB{n}Hi", f"{float(r['f1_hi']):.3f}"),
            macro(f"vB{n}Prec", f"{float(r['precision']):.3f}"),
            macro(f"vB{n}Rec", f"{float(r['recall']):.3f}"),
            macro(f"vB{n}Mcc", f"{float(r['mcc']):.3f}"),
            macro(f"vB{n}Fnr", f"{100 * float(r['fnr']):.1f}" + BS + "%"),
            macro(f"vB{n}Prauc", f"{float(r['pr_auc']):.3f}"),
            macro(f"vB{n}Train", f"{float(r['train_s']):.0f}"),
        ]
else:
    print("! binary_results_v2.csv missing (Phase 2 still running)")

pd_p = RESC / "paired_delta_v2.json"
if pd_p.exists():
    d = json.load(open(pd_p, encoding="utf-8"))
    out += [
        macro("vDeltaFone", f"{d['delta_f1_xgb_minus_rf']:+.4f}"),
        macro("vDeltaLo", f"{d['ci95'][0]:+.4f}"),
        macro("vDeltaHi", f"{d['ci95'][1]:+.4f}"),
        # NOT a p-value: this is the fraction of bootstrap replicates in
        # which XGB beat RF. Emitting it as "vDeltaP" invited the exact
        # misreading that 1.000 means "no effect", when it means the
        # opposite. B replicates cannot resolve a proportion finer than
        # 1/B, so a clean sweep is reported at that ceiling, never as 1.
        # Math mode is required: a bare ">" in text mode under OT1 renders
        # as an inverted question mark, corrupting the number with no
        # compile error to warn anyone.
        macro("vDeltaPgt", "$>$0.999" if d["p_delta_gt_0"] >= 1.0
              else f"{d['p_delta_gt_0']:.3f}"),
        macro("vDeltaB", f"{d['B']:,}".replace(",", "{,}")),
    ]

# --- derived wording for the deep ablation -------------------------------
# Which architectural axes the ablation covers, and what to admit when a
# config is absent, both follow from the artifacts. Hard-coding either
# invites prose that outlives the facts -- a caveat naming "two" when only
# one is missing is as wrong as promising coverage that is not there.
_DL_DESC = {
    "C3_pure_cnn": ("pure CNN", "a pure convolutional variant",
                    "fourth of ten"),
    "C4_pure_transformer": ("pure Transformer", "a pure Transformer variant",
                            "last of ten"),
}


def _arch_phrase(missing):
    axes = ["depth", "width"]
    for k in ("C3_pure_cnn", "C4_pure_transformer"):
        if k not in missing:
            axes.append(_DL_DESC[k][0])
    if len(axes) == 2:
        return " and ".join(axes)
    return ", ".join(axes[:-1]) + ", and " + axes[-1]


def _caveat(missing):
    absent = [k for k in ("C3_pure_cnn", "C4_pure_transformer")
              if k in missing]
    if not absent:
        return ""
    longs = [_DL_DESC[k][1] for k in absent]
    ranks = [_DL_DESC[k][2] for k in absent]
    if len(absent) == 1:
        head = f"One further architectural ablation ({longs[0]})"
        be, subj, rank = "is", "it ", f"ranked {ranks[0]}"
        tail = ("so it was not the strongest deep configuration there; we "
                "make no claim about where it would land under the present "
                "protocol.")
    else:
        head = (f"Two further architectural ablations ({longs[0]} and "
                f"{longs[1]})")
        be, subj = "are", "those two "
        rank = f"ranked {ranks[0]} and {ranks[1]}"
        tail = ("so neither was the strongest deep configuration there; we "
                "make no claim about where they would land under the present "
                "protocol.")
    return (f"{head} did not complete within the GPU allocation available for "
            f"this study and {be} excluded. We flag this rather than omit it "
            f"silently. Under the earlier, leaking protocol {subj}{rank}, "
            f"{tail}")


_EXPECTED_DL = ["A1_baseline", "B1_pos_weight", "B2_focal_g2",
                "B3_focal_pos_weight", "B4_asymmetric",
                "B5_threshold_tuning_on_best", "C1_transformer_4layers",
                "C2_dmodel_256", "C3_pure_cnn", "C4_pure_transformer"]

st_p = BASE / "stats_v2.json"
if st_p.exists():
    s = json.load(open(st_p, encoding="utf-8"))
    ml = s["classical_multilabel"]
    out += [
        macro("vMlXgb", f"{ml['ML2_XGBoost']['macro_f1']:.4f}"),
        macro("vMlRf", f"{ml['ML1_RandomForest']['macro_f1']:.4f}"),
        macro("vMlLr", f"{ml['ML0_LogReg_balanced']['macro_f1']:.4f}"),
    ]
    dl = s.get("dl") or {}
    if dl:
        _missing = [c for c in _EXPECTED_DL if c not in dl]
        best = max(dl, key=lambda k: dl[k]["macro_f1"] or 0)
        c = s["stats"]["comparisons"][best]
        macros_dl = [
            macro("vDlBestName", best.replace("_", BS + "_")),
            macro("vDlBest", f"{dl[best]['macro_f1']:.4f}"),
            macro("vDlWorst", f"{min(v['macro_f1'] for v in dl.values()):.4f}"),
            macro("vDlMean",
                  f"{sum(v['macro_f1'] for v in dl.values()) / len(dl):.4f}"),
            macro("vDlGap", f"{c['gap_pp']:.2f}"),
            macro("vDlWins", f"{c['xgb_wins_of_8']}/8"),
            macro("vDlPsign", f"{c['p_sign']:.3f}"),
            macro("vDlHolm", "yes" if c["significant_after_holm"] else "no"),
            macro("vDlLosing", str(s["stats"]["model_level"]["dl_configs_below_xgb_macro"])),
            macro("vDlOf", str(s["stats"]["model_level"]["of"])),
            macro("vDlNConfigs", str(len(dl))),
            # The architecture list and the limitation note are DERIVED from
            # which configs actually landed, never typed. Prose promising
            # "pure CNN and pure Transformer" would claim coverage the run
            # may not have, and a caveat naming "two" when only one is absent
            # is equally wrong -- so both are computed from _missing.
            macro("vDlArch", _arch_phrase(_missing)),
            macro("vDlCaveat", _caveat(_missing)),
        ]
        out += macros_dl
else:
    print("! stats_v2.json missing (Phase 4 not run yet)")

hdr = "% AUTO-GENERATED by emit_macros_v2.py - do not edit by hand"
(BASE / "numbers_v2.tex").write_text(hdr + "\n" + "\n".join(out) + "\n",
                                     encoding="utf-8")
print(f"wrote numbers_v2.tex with {len(out)} macros")
for line in out[:6]:
    print("   ", line)
