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

import os as _os
import pathlib as _pathlib

# Working directory. Defaults to the results/ tree shipped in this
# repository so the analysis and emit scripts run straight from a checkout;
# set PAPER_V2_DIR to a full working tree (with the parquet splits and
# runs_v2/) to regenerate results from scratch.
_ROOT = _os.environ.get("PAPER_V2_DIR") or str(
    _pathlib.Path(__file__).resolve().parent.parent / "results")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V2 = Path(_ROOT)
# Check the RENDERED document, not the source. Every number reaches the
# paper through a macro, so the .tex holds no digits to verify; a stale
# numbers_v2.tex or an unexpanded macro would sail past a source-level grep
# and land in the PDF a reviewer reads.
TEX = Path(_os.environ.get("PAPER_PDF", str(_pathlib.Path(_ROOT).parent / "main_v2.pdf")))
if not TEX.exists():
    # The built paper is deliberately not shipped in this repository:
    # whether a preprint may be posted is a venue decision for the author,
    # not something a code release should presume.
    raise SystemExit(
        f"paper PDF not found at {TEX}\n"
        f"Build it, then point the gate at it:\n"
        f"  PAPER_PDF=/path/to/main_v2.pdf PAPER_V2_DIR=./results \\\n"
        f"    python scripts/check_numbers_v2.py")

import fitz  # PyMuPDF

with fitz.open(TEX) as _doc:
    tex = "\n".join(pg.get_text() for pg in _doc)

# Hyphenation and column breaks split values across lines; flatten first.
tex_flat = re.sub(r"\s+", " ", tex)
# The PDF renders thin-space thousands separators as U+2009/U+00A0 or as a
# plain space; normalise them to a comma so "112,467" matches either form.
tex_flat = re.sub(r"(?<=\d)[\u2009\u202f\u00a0 ](?=\d\d\d\b)", ",", tex_flat)
man = json.load(open(V2 / "manifest_v2.json", encoding="utf-8"))

failures = []
checked = 0

def expect(desc, needle):
    """Assert the exact string appears in the tex."""
    global checked
    checked += 1
    if needle in tex_flat or needle.replace(",", "") in tex_flat:
        print(f"  OK   {desc}: {needle}")
    else:
        failures.append((desc, needle))
        print(f"  MISS {desc}: expected '{needle}' in {TEX.name}")

def expect_val(desc, value, lo=3, hi=4):
    """Assert `value` appears at some precision between lo and hi decimals.

    Anchored on both sides: without the trailing guard, "0.679" would be
    satisfied by "0.6793", so a paper reporting a different number at higher
    precision would pass. Without the leading guard, "0.945" would match
    inside "10.945".
    """
    global checked
    checked += 1
    for dp in range(lo, hi + 1):
        needle = f"{value:.{dp}f}"
        for m in re.finditer(re.escape(needle), tex_flat):
            after = tex_flat[m.end():m.end() + 1]
            before = tex_flat[max(0, m.start() - 1):m.start()]
            if not after.isdigit() and not before.isdigit():
                print(f"  OK   {desc}: {needle}")
                return
    failures.append((desc, f"{value:.{lo}f}"))
    print(f"  MISS {desc}: expected {value:.{lo}f}..{value:.{hi}f} "
          f"in {TEX.name}")


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
    expect_val("XGB macro", ml['ML2_XGBoost']['macro_f1'])
    expect_val("RF macro", ml['ML1_RandomForest']['macro_f1'])
    if s.get("dl"):
        best = max(s["dl"], key=lambda k: s["dl"][k]["macro_f1"] or 0)
        c = s["stats"]["comparisons"][best]
        expect_val("best DL macro", s['dl'][best]['macro_f1'])
        expect_val("gap pp", c['gap_pp'], lo=1, hi=2)
        expect("wins", f"{c['xgb_wins_of_8']}/8")
else:
    print("— stats_v2 not present yet (Phase 4 pending) —")

# Every value the macro file emits must be visible in the rendered PDF.
# This is the check that cannot be satisfied by a stale numbers_v2.tex: if
# the macros were regenerated after the last pdflatex run, the PDF still
# carries the old values and they will not match here.
print("\n- macro sweep (numbers_v2.tex -> PDF) -")
nums = Path(_ROOT) / "numbers_v2.tex"
skip = {"vDlCaveat", "vDlArch", "vDlBestName", "vDlHolm", "vSeed"}
for line in nums.read_text(encoding="utf-8").split("\n"):
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(.+)\}$", line.strip())
    if not m:
        continue
    name, val = m.group(1), m.group(2)
    if name in skip:
        continue
    # Strip LaTeX presentation so the comparison sees the rendered value:
    # thousands braces, escaped percent/underscore, and math delimiters
    # (">" is emitted as "$>$" because text mode would corrupt it).
    plain = (val.replace("{,}", ",").replace("\\%", "%")
                .replace("\\_", "_").replace("$", "").replace(">", ""))
    if not re.search(r"\d", plain):
        continue
    checked += 1
    if plain in tex_flat or plain.replace(",", "") in tex_flat:
        pass
    else:
        failures.append((f"macro {name}", plain))
        print(f"  MISS macro {name}: {plain}")
print(f"  macro sweep done")

print(f"\nchecked {checked}, mismatches {len(failures)}")
if failures:
    print("\nFIX THESE:")
    for d, n in failures:
        print(f"  {d}: {n}")
    sys.exit(1)
print("ALL NUMBERS CONSISTENT")
