# -*- coding: utf-8 -*-
"""Paper v2 — Phase 7: build the public reproduction notebook.

The notebook re-derives every headline number in the paper from the released
splits and prints a PASS/FAIL table against the published values. Those
published values are injected here FROM THE ARTIFACTS, never typed, so the
notebook cannot drift from the paper the way v1's numbers drifted from each
other.

What it reproduces honestly, and what it does not:
  * classical models are retrained from the recorded hyperparameters and
    re-evaluated at the recorded validation-selected thresholds -- a real
    reproduction, minus the Optuna search itself (50 trials x 5 folds is not
    a notebook-sized job; the search's chosen parameters are published and
    the notebook says so plainly).
  * the deep ablation is NOT retrained -- that needs a GPU and hours. The
    notebook carries the per-class scores from the run and recomputes the
    paired statistics from them, which is the part a reader is most likely
    to want to check.
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

V2 = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper\v2")
RESC = V2 / "results_classical"
OUT = V2 / "kernel_repro"
OUT.mkdir(exist_ok=True)

man = json.load(open(V2 / "manifest_v2.json", encoding="utf-8"))
st = json.load(open(V2 / "stats_v2.json", encoding="utf-8"))
thr = json.load(open(RESC / "thresholds_v2.json", encoding="utf-8"))
opt = json.load(open(RESC / "optuna_summary.json", encoding="utf-8"))
delta = json.load(open(RESC / "paired_delta_v2.json", encoding="utf-8"))
feats = json.load(open(V2 / "feature_columns.json", encoding="utf-8"))

import csv
binr = {r["model"]: {k: (float(v) if k != "model" else v)
                     for k, v in r.items()}
        for r in csv.DictReader(open(RESC / "binary_results_v2.csv",
                                     encoding="utf-8"))}

PUBLISHED = {
    "manifest": {k: man[k] for k in
                 ["seed", "n_features", "n_train", "n_val", "n_test",
                  "sha256_train", "sha256_val", "sha256_test",
                  "positive_rate", "positive_rate_train", "positive_rate_val",
                  "positive_rate_test", "dropped_bytecode_dups",
                  "dropped_feature_dups"]},
    "binary": {m: {"f1": v["f1"], "f1_lo": v["f1_lo"], "f1_hi": v["f1_hi"],
                   "precision": v["precision"], "recall": v["recall"],
                   "fnr": v["fnr"], "mcc": v["mcc"], "pr_auc": v["pr_auc"]}
               for m, v in binr.items()},
    "thresholds": thr,
    "xgb_params": opt["best_params"],
    "multilabel": {k: v["macro_f1"] for k, v in
                   st["classical_multilabel"].items()},
    "multilabel_per_label": {k: v["f1_per_label"] for k, v in
                             st["classical_multilabel"].items()},
    "dl": {k: v["f1_per_label"] for k, v in st["dl"].items()},
    "dl_macro": {k: v["macro_f1"] for k, v in st["dl"].items()},
    "delta": delta,
    "comparisons": st["stats"]["comparisons"],
    "labels": man["labels"],
    "features": feats,
}

_env = {}
for _mod, _key in [("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
                   ("numpy", "numpy"), ("pandas", "pandas"),
                   ("scipy", "scipy"), ("catboost", "catboost")]:
    try:
        _env[_key] = __import__(_mod).__version__
    except Exception:
        pass
_env["python"] = sys.version.split()[0]
PUBLISHED["env"] = _env

PUBLISHED["n_checks"] = (
    3                                            # split digests
    + 6                                          # row counts + positive rates
    + 3                                          # pairwise overlap
    + len(PUBLISHED["binary"]) * 8               # 6 point metrics + 2 CI ends
    + 3                                          # paired delta + its CI
    + len(PUBLISHED["multilabel"]) * (1 + len(PUBLISHED["labels"]))
    + len(PUBLISHED["dl"]) * 4                   # wins, p_sign, gap, Holm
)


_CELL_N = [0]


def _lines(text):
    """Split into nbformat source lines, KEEPING line endings.

    Bare fragments make Jupyter concatenate consecutive lines; keepends is
    what the format actually specifies.
    """
    return text.splitlines(keepends=True)


def _cid():
    _CELL_N[0] += 1
    return f"cell-{_CELL_N[0]:02d}"


def md(src):
    return {"cell_type": "markdown", "id": _cid(), "metadata": {},
            "source": _lines(src.strip())}


def code(src):
    return {"cell_type": "code", "id": _cid(), "metadata": {},
            "execution_count": None, "outputs": [],
            "source": _lines(src.strip("\n"))}


def _coverage_sentence(dl):
    """Say what actually landed. The count used to be derived while the
    excuse beside it was hard-coded, so the notebook could claim ten of ten
    complete AND two excluded in the same sentence -- which it did."""
    have = set(dl)
    missing = [c for c in ("C3_pure_cnn", "C4_pure_transformer")
               if c not in have]
    if not missing:
        return (f"all {len(have)} of 10 planned deep configurations "
                "completed and are checked below.")
    names = " and ".join(f"`{m}`" for m in missing)
    return (f"{len(have)} of 10 planned deep configurations completed; "
            f"{names} exhausted the GPU allocation and "
            f"{'is' if len(missing) == 1 else 'are'} excluded rather than "
            "estimated.")

cells = []

_coverage = _coverage_sentence(PUBLISHED['dl'])
cells.append(md(f"""
# Reproducing the v2 results

This notebook re-derives every headline number in the paper from the
released splits and reports **PASS/FAIL** against the published values.

The published values below are injected from the run artifacts by
`build_repro_notebook.py`; none of them is typed by hand. That matters: the
defect this paper corrects is partly a bookkeeping one — v1's numbers drifted
between the paper, the README and the product page because each copy was
maintained separately.

**What is reproduced.** The classical models are retrained from the recorded
hyperparameters and re-evaluated on the test split at the thresholds selected
on validation. The Optuna search itself (50 trials x 5 folds) is not re-run
here — its chosen parameters are published and used directly.

**What is not.** The {len(PUBLISHED['dl'])}-configuration Conv-Transformer
ablation needs a GPU and hours. This notebook carries the per-class scores
from that run and recomputes the paired statistics from them, which is the
part most worth checking independently.

**Ablation coverage.** {_coverage}
"""))

cells.append(code(f"""
import hashlib, json, os, time
from pathlib import Path
import numpy as np, pandas as pd

PUBLISHED = json.loads(r'''{json.dumps(PUBLISHED)}''')
LABELS   = PUBLISHED["labels"]
FEATURES = PUBLISHED["features"]
RESULTS  = {{}}          # check name -> (ok, published, reproduced)

def check(name, published, got, tol=5e-4):
    ok = (published == got) if isinstance(published, str) \\
         else abs(float(published) - float(got)) <= tol
    RESULTS[name] = (ok, published, got)
    print(f"  {{'PASS' if ok else 'FAIL'}}  {{name}}: published={{published}} got={{got}}")
    return ok

def mount():
    \"\"\"Kaggle has moved dataset mount points; accept either layout.\"\"\"
    for c in [Path("/kaggle/input/defi-bytecode-features-v2"),
              Path("/kaggle/input/datasets/sergeisolovyev/defi-bytecode-features-v2"),
              Path(".")]:
        if (c / "train_v2.parquet").exists():
            return c
    raise SystemExit("dataset not mounted")

D = mount()
print("dataset at", D)
"""))

cells.append(code("""
import importlib, platform
print("library versions -- 'authored' is what produced the published "
      "numbers; a mismatch is the first thing to suspect if a check misses "
      "by a hair")
for pkg, ver in sorted(PUBLISHED["env"].items()):
    if pkg == "python":
        here = platform.python_version()
    else:
        try:
            here = importlib.import_module(
                "sklearn" if pkg == "scikit-learn" else pkg).__version__
        except ImportError:
            here = "MISSING"
    print(f"  {pkg:14s} authored={ver:10s} here={here}"
          f"{'' if here == ver else '   <-- differs'}")
"""))

cells.append(md("""
## 1. Split integrity

Before any metric is worth reading, the splits must be the ones the paper
used. The digests below are the build-time SHA-256 of each parquet file, and
the no-overlap assertion is the invariant the build script enforced: no
feature vector may appear in more than one split. That invariant *is* the
correction — v1 deduplicated on the raw bytecode string, which let contracts
that are bit-identical in feature space sit on both sides of the split.
"""))

cells.append(code("""
def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

print("digests")
for split in ["train", "val", "test"]:
    check(f"sha256_{split}", PUBLISHED["manifest"][f"sha256_{split}"],
          sha256(D / f"{split}_v2.parquet"))

# Read only the numeric columns. The bytecode column dominates the file
# (train_v2.parquet is ~493 MB, almost all of it strings) and the classical
# pipeline never touches it -- loading it costs minutes and gigabytes for
# nothing.
COLS = FEATURES + LABELS + ["is_vulnerable"]
parts = {s: pd.read_parquet(D / f"{s}_v2.parquet", columns=COLS) for s in
         ["train", "val", "test"]}

print("\\nrow counts and positive rates")
for s, df in parts.items():
    check(f"n_{s}", PUBLISHED["manifest"][f"n_{s}"], len(df))
    check(f"positive_rate_{s}", PUBLISHED["manifest"][f"positive_rate_{s}"],
          float(df["is_vulnerable"].mean()), tol=1e-6)

print("\\nno feature vector appears in more than one split")
# Hash rows rather than building sets of 67-tuples: the tuple form needs
# ~90k x 67 Python floats per split and takes minutes. A 64-bit row hash
# over 112k rows has a collision probability around 3e-10, which is far
# below anything that would change the verdict.
keys = {s: set(pd.util.hash_pandas_object(df[FEATURES], index=False))
        for s, df in parts.items()}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    n = len(keys[a] & keys[b])
    RESULTS[f"overlap_{a}_{b}"] = (n == 0, 0, n)
    print(f"  {'PASS' if n == 0 else 'FAIL'}  overlap {a}/{b}: {n}")
"""))

cells.append(md("""
## 2. Binary detection

Retrained from the published hyperparameters, scored on the test split at
the thresholds chosen on validation. The test split is read once, here.
"""))

cells.append(code("""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             matthews_corrcoef, average_precision_score)
import xgboost as xgb

Xtr = parts["train"][FEATURES].to_numpy("float32")
ytr = parts["train"]["is_vulnerable"].to_numpy()
Xte = parts["test"][FEATURES].to_numpy("float32")
yte = parts["test"]["is_vulnerable"].to_numpy()

sc = StandardScaler().fit(Xtr)

models = {
    "XGB": xgb.XGBClassifier(**PUBLISHED["xgb_params"], n_jobs=-1,
                             random_state=42, eval_metric="logloss"),
    "RF":  RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  n_jobs=-1, random_state=42),
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=42),
}
SKIPPED = 0
try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=800, auto_class_weights="Balanced",
        random_seed=42, verbose=False)
except ImportError:
    SKIPPED += 8
    print("catboost unavailable: the 8 published CatBoost numbers are NOT "
          "reproduced in this run")

def boot_ci(y, s, thr, B=1000, seed=42):
    \"\"\"Stratified percentile bootstrap for F1 -- ported verbatim from
    run_classical_v2.py:114-122, so the published interval is recomputed
    rather than restated.\"\"\"
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)),
                            rng.choice(neg, len(neg))])
        vals.append(f1_score(y[i], (s[i] >= thr).astype(int)))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

probs = {}

for name, m in models.items():
    t0 = time.time()
    A, B = (sc.transform(Xtr), sc.transform(Xte)) if name == "LogReg" \\
           else (Xtr, Xte)
    m.fit(A, ytr)
    p = m.predict_proba(B)[:, 1]
    probs[name] = p
    pred = (p >= PUBLISHED["thresholds"][name]).astype(int)
    got = {"f1": f1_score(yte, pred), "precision": precision_score(yte, pred),
           "recall": recall_score(yte, pred), "mcc": matthews_corrcoef(yte, pred),
           "pr_auc": average_precision_score(yte, p)}
    got["fnr"] = 1 - got["recall"]
    got["f1_lo"], got["f1_hi"] = boot_ci(yte, p,
                                         PUBLISHED["thresholds"][name])
    print(f"\\n{name}  ({time.time()-t0:.0f}s)")
    # Tolerances calibrated on two MEASURED signatures from real runs:
    #  * correct spec across library builds (authored: sklearn 1.8.0 /
    #    xgboost 3.1.2; Kaggle: newer): F1 agrees to 4e-6, while
    #    precision/recall/FNR/MCC drift up to 1.6e-3 -- a different build
    #    lands a handful of borderline contracts across the threshold;
    #  * a WRONG spec (RF as 400 unweighted trees instead of 500 balanced)
    #    moved those components by 3.0e-3..4.3e-3 and was caught.
    # Headline scores stay tight; components get 2e-3, which separates
    # version noise from a genuine specification error.
    for k in ["f1", "pr_auc"]:
        check(f"binary_{name}_{k}", PUBLISHED["binary"][name][k], got[k],
              tol=5e-4)
    for k in ["precision", "recall", "fnr", "mcc"]:
        check(f"binary_{name}_{k}", PUBLISHED["binary"][name][k], got[k],
              tol=2e-3)
    # looser: these two also carry the resampler's stream. Measured spread
    # across rng seeds is <= 2.4e-4.
    for k in ["f1_lo", "f1_hi"]:
        check(f"binary_{name}_{k}", PUBLISHED["binary"][name][k], got[k],
              tol=1e-3)
"""))

cells.append(md("""
## 2b. Is XGBoost really ahead of the random forest?

Not an argument from the two intervals above: overlapping marginal CIs do
not license a difference claim, and leaning on them was one of v1's defects.
The paper's claim rests on a *paired* bootstrap over the same resampled test
sets, recomputed here from the two score vectors section 2 just produced.
"""))

cells.append(code("""
rng = np.random.default_rng(42)
pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
deltas = []
for _ in range(PUBLISHED["delta"]["B"]):
    i = np.concatenate([rng.choice(pos, len(pos)),
                        rng.choice(neg, len(neg))])
    deltas.append(
        f1_score(yte[i], (probs["XGB"][i]
                          >= PUBLISHED["thresholds"]["XGB"]).astype(int))
        - f1_score(yte[i], (probs["RF"][i]
                            >= PUBLISHED["thresholds"]["RF"]).astype(int)))
d = np.array(deltas)
lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
print(f"paired dF1 (XGB-RF) = {d.mean():+.5f}  "
      f"CI95 [{lo:+.5f}, {hi:+.5f}]  P(d>0) = {(d > 0).mean():.3f}")
check("delta_f1_xgb_minus_rf", PUBLISHED["delta"]["delta_f1_xgb_minus_rf"],
      float(d.mean()), tol=5e-4)
check("delta_ci95_lo", PUBLISHED["delta"]["ci95"][0], lo, tol=1e-3)
check("delta_ci95_hi", PUBLISHED["delta"]["ci95"][1], hi, tol=1e-3)
"""))

cells.append(md("""
## 3. Multi-label

Eight one-vs-rest heads at the default threshold, exactly as in the paper --
all three estimators, including the balanced logistic regression whose
0.4538 is the leftmost column of the published per-label heatmap.
"""))

cells.append(code("""
from sklearn.multioutput import MultiOutputClassifier

Ytr = parts["train"][LABELS].to_numpy("int8")
Yte = parts["test"][LABELS].to_numpy("int8")

ml_models = {
    "ML2_XGBoost": MultiOutputClassifier(
        xgb.XGBClassifier(n_estimators=400, max_depth=8, n_jobs=-1,
                          random_state=42, eval_metric="logloss"), n_jobs=1),
    "ML1_RandomForest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42),
    # analyze_v2.py:63-64. NOT wrapped in a scaler: the multi-label LogReg
    # is fitted on the raw features, unlike the binary one. Standardising
    # here would silently produce something other than 0.4538.
    "ML0_LogReg_balanced": MultiOutputClassifier(
        LogisticRegression(max_iter=1500, class_weight="balanced"), n_jobs=-1),
}

ml_got = {}
for name, m in ml_models.items():
    t0 = time.time()
    m.fit(Xtr, Ytr)
    P = m.predict(Xte)
    per = [float(f1_score(Yte[:, i], P[:, i])) for i in range(len(LABELS))]
    ml_got[name] = per
    print(f"\\n{name}  ({time.time()-t0:.0f}s)")
    check(f"multilabel_{name}", PUBLISHED["multilabel"][name],
          float(np.mean(per)), tol=3e-3)
    # the published figure is the per-label matrix, not the macro: a macro
    # check alone lets per-class errors cancel, and the rare classes
    # (ML1 double-spending 0.267, bad-randomness 0.484) are exactly where a
    # reproduction would drift.
    # 1e-2, not 5e-3: rare-class cells are decided by a few dozen
    # positives, and a different sklearn build breaks ties differently --
    # measured 7.9e-3 on ML1/bad-randomness for a spec-identical forest,
    # while a wrong forest spec shifts rare-class F1 by several times that.
    for i, lab in enumerate(LABELS):
        check(f"multilabel_{name}_{lab}",
              PUBLISHED["multilabel_per_label"][name][i], per[i], tol=1e-2)
"""))

cells.append(md("""
## 4. The deep comparator, and what eight paired observations support

The ablation is not retrained here. What *is* recomputed is the statistical
claim built on it — and that claim is deliberately modest. With eight
classes the smallest two-sided p a sign test can return is 2/256 = 0.0078,
so a clean sweep means "eight out of eight" and nothing stronger. The sign
test and Wilcoxon applied to the same eight numbers are one piece of
evidence reported twice, not two confirmations; only the sign test is used.
"""))

cells.append(code("""
from scipy.stats import binomtest

# The comparator must be the vector section 3 just REPRODUCED. Reading the
# published one here makes the table self-referential: it would print the
# same 8/8, the same p and the same 8.53 pp even if the retrain above had
# produced nonsense.
xgbv = np.array(ml_got["ML2_XGBoost"])
rows, pvals = [], []
for cfg, per in sorted(PUBLISHED["dl"].items()):
    c = np.array(per)
    wins = int((xgbv > c).sum())
    p = binomtest(wins, len(LABELS), 0.5, alternative="two-sided").pvalue
    rows.append((cfg, float(c.mean()), wins, float(p)))
    pvals.append((cfg, float(p)))

# Holm STEP-DOWN. Sort ascending, compare p_(i) to alpha/(m-i), and stop at
# the first failure -- everything after it is retained too. Testing each p
# against its own threshold independently (what stood here) lets later
# hypotheses pass on the looser thresholds that only become available once
# the stricter ones have been rejected. With ten identical p = 0.0078125
# that marked six of ten significant; the correct answer is none, because
# the smallest p already exceeds alpha/10 = 0.005.
m = len(pvals)
holm, _rejecting = {}, True
for _r, (_k, _p) in enumerate(sorted(pvals, key=lambda t: t[1])):
    if _rejecting and _p > 0.05 / (m - _r):
        _rejecting = False
    holm[_k] = _rejecting

print(f"{'config':32s} {'macro-F1':>9s} {'XGB wins':>9s} {'p_sign':>8s}  Holm")
for cfg, macro, wins, p in sorted(rows, key=lambda t: -t[1]):
    print(f"{cfg:32s} {macro:9.4f} {wins:6d}/8 {p:8.4f}  "
          f"{'yes' if holm[cfg] else 'no'}")
    pub = PUBLISHED["comparisons"][cfg]
    check(f"dl_wins_{cfg}", pub["xgb_wins_of_8"], wins, tol=0)
    check(f"dl_psign_{cfg}", pub["p_sign"], p, tol=1e-9)
    check(f"dl_gap_pp_{cfg}", pub["gap_pp"],
          100 * (xgbv.mean() - macro), tol=0.3)
    check(f"dl_holm_{cfg}", float(pub["significant_after_holm"]),
          float(holm[cfg]), tol=0)

best = max(rows, key=lambda t: t[1])
print(f"\\nbest deep configuration: {best[0]} at {best[1]:.4f}")
print(f"classical XGBoost: {xgbv.mean():.4f}")
print(f"gap: {100*(xgbv.mean()-best[1]):.2f} percentage points")
print(f"sign-test floor at n=8: {2/2**8:.4f} -- p={best[3]:.4f} IS that floor"
      if abs(best[3] - 2/2**8) < 1e-9 else "")
"""))

cells.append(md("""
## 5. Verdict
"""))

cells.append(code("""
bad = [k for k, (ok, *_) in RESULTS.items() if not ok]
print(f"checks run: {len(RESULTS)}  (+{SKIPPED} skipped)")
print(f"expected:   {PUBLISHED['n_checks']}")
print(f"failed:     {len(bad)}")
if len(RESULTS) + SKIPPED != PUBLISHED["n_checks"]:
    raise SystemExit(
        f"COVERAGE mismatch: ran {len(RESULTS)} checks +{SKIPPED} skipped, "
        f"expected {PUBLISHED['n_checks']} -- a published number is going "
        f"unreproduced")
if bad:
    print("\\nFAILED:")
    for k in bad:
        ok, pub, got = RESULTS[k]
        print(f"  {k}: published={pub} reproduced={got}")
    raise SystemExit("reproduction MISMATCH -- see above")
print("\\nAll published numbers reproduced.")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

(OUT / "repro_v2.ipynb").write_text(json.dumps(nb, indent=1),
                                    encoding="utf-8")

meta = {"id": "sergeisolovyev/sc-vuln-v2-reproduction",
        "title": "SC Vuln v2 Reproduction",
        "code_file": "repro_v2.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": ["sergeisolovyev/defi-bytecode-features-v2"],
        "competition_sources": [], "kernel_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")

n_checks = PUBLISHED["n_checks"]
print(f"wrote {OUT / 'repro_v2.ipynb'}")
print(f"  {len(cells)} cells, ~{n_checks} assertions")
print(f"  DL configs carried: {len(PUBLISHED['dl'])}")
