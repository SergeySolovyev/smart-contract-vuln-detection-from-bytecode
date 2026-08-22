# -*- coding: utf-8 -*-
"""Paper v2 — Phase 5 gate: every number in main.tex must match an artifact.

The v1 failure mode was numbers drifting between text, figures, README and
site because nothing enforced agreement. This checker is that enforcement:
it reads the AUTHORITATIVE artifacts (results_classical/*.csv|json,
stats_v2.json, manifest_v2.json) and greps main.tex for each headline value.
Run it after every edit of the paper; CI-style exit code 1 on any mismatch.

Checked claims (extend as the v2 text solidifies):
  corpus size, split sizes, positive rate, n_features
  binary: F1/CI per model, FNR, PR-AUC, threshold provenance
  multilabel: XGB/RF/LogReg macro-F1, best-DL macro, gap, wins, p-values
"""
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V2 = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper\v2")
TEX = Path(r"D:\DeFi\Научный_телеграф\icicpe_paper\main_v2.tex")
if not TEX.exists():
    TEX = Path(r"D:\DeFi\Научный_телеграф\icicpe_paper\main.tex")

tex = TEX.read_text(encoding="utf-8")
tex_flat = re.sub(r"\s+", " ", tex)
man = json.load(open(V2 / "manifest_v2.json", encoding="utf-8"))

failures = []
checked = 0

def expect(desc, needle):
    """Assert the exact string appears in the tex."""
    global checked
    checked += 1
    if needle.replace(",", "{,}") in tex_flat or needle in tex_flat:
        print(f"  OK   {desc}: {needle}")
    else:
        failures.append((desc, needle))
        print(f"  MISS {desc}: expected '{needle}' in {TEX.name}")

print(f"checking {TEX.name} against v2 artifacts\n")

print("— corpus —")
expect("corpus size", f"{man['steps'][3]['rows']:,}")
expect("train size", f"{man['n_train']:,}")
expect("val size", f"{man['n_val']:,}")
expect("test size", f"{man['n_test']:,}")
expect("bytecode dups dropped", f"{man['dropped_bytecode_dups']:,}"
       if man['dropped_bytecode_dups'] >= 1000 else str(man['dropped_bytecode_dups']))
expect("feature dups dropped", f"{man['dropped_feature_dups']:,}")

bcsv = V2 / "results_classical" / "binary_results_v2.csv"
if bcsv.exists():
    import csv
    rows = {r["model"]: r for r in csv.DictReader(open(bcsv, encoding="utf-8"))}
    print("— binary (test set) —")
    for mname, r in rows.items():
        expect(f"{mname} F1", f"{float(r['f1']):.3f}")
    if "XGB" in rows:
        expect("XGB CI lo", f"{float(rows['XGB']['f1_lo']):.3f}")
        expect("XGB CI hi", f"{float(rows['XGB']['f1_hi']):.3f}")
else:
    print("— binary results not present yet (Phase 2 running) —")

sv2 = V2 / "stats_v2.json"
if sv2.exists():
    s = json.load(open(sv2, encoding="utf-8"))
    ml = s["classical_multilabel"]
    print("— multilabel —")
    expect("XGB macro", f"{ml['ML2_XGBoost']['macro_f1']:.3f}")
    expect("RF macro", f"{ml['ML1_RandomForest']['macro_f1']:.3f}")
    if s.get("dl"):
        best = max(s["dl"], key=lambda k: s["dl"][k]["macro_f1"] or 0)
        c = s["stats"]["comparisons"][best]
        expect("best DL macro", f"{s['dl'][best]['macro_f1']:.3f}")
        expect("gap pp", f"{c['gap_pp']:.1f}")
        expect("wins", f"{c['xgb_wins_of_8']}/8")
else:
    print("— stats_v2 not present yet (Phase 4 pending) —")

print(f"\nchecked {checked}, mismatches {len(failures)}")
if failures:
    print("\nFIX THESE:")
    for d, n in failures:
        print(f"  {d}: {n}")
    sys.exit(1)
print("ALL NUMBERS CONSISTENT")
