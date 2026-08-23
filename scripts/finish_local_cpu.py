# -*- coding: utf-8 -*-
"""Train the two configs Kaggle could not deliver (C3_pure_cnn,
C4_pure_transformer) on the local CPU.

Why this exists: the weekly Kaggle GPU quota (30h) was exhausted by the two
sessions the 12h wall destroyed plus the one that succeeded, so the reverse
kernel never delivered C3/C4. C3 ranked 4th of 10 in the v1 ablation, so it
is NOT safe to omit it and assume the conclusion holds.

Same pipeline, same v2 splits, same protocol — only the device differs, and
the device does not affect the metric. Results are written in the identical
JSON shape to runs_v2/, so collect_dl_v2.py and analyze_v2.py consume them
without knowing the difference. Also POSTs to ops-drop for parity of
provenance.
"""
import io
import json
import os
import sys
import time
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
KP = Path(_ROOT).parent
RUNS = V2 / "runs_v2"
RUNS.mkdir(exist_ok=True)
sys.path.insert(0, str(KP))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

from dl_pipeline import (BytecodeTokenizer, build_token_ids, DL_EXPERIMENTS,
                         run_dl_experiment)

LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
FEATURES = json.load(open(V2 / "feature_columns.json", encoding="utf-8"))
WANT = {"C3_pure_cnn", "C4_pure_transformer"}

print("loading v2 splits...")
cols = ["bytecode"] + FEATURES + LABELS
tr = pd.read_parquet(V2 / "train_v2.parquet", columns=cols)
te = pd.read_parquet(V2 / "test_v2.parquet", columns=cols)
print(f"train {len(tr):,}  test {len(te):,}  threads={torch.get_num_threads()}")

tokenizer = BytecodeTokenizer().fit(tr["bytecode"].head(20000))
tr_ids = build_token_ids(tr["bytecode"], tokenizer)
te_ids = build_token_ids(te["bytecode"], tokenizer)
y_tr = tr[LABELS].to_numpy("float32")
y_te = te[LABELS].to_numpy("float32")
X_tr = tr[FEATURES].to_numpy("float32")
X_te = te[FEATURES].to_numpy("float32")
del tr, te
print("vocab:", tokenizer.vocab_size)

def upload(name: str, payload: dict) -> None:
    try:
        import urllib.request
        sec = [l.split("=", 1)[1].strip()
               for l in io.open(os.environ.get("DL_DROP_ENV", ""),
                                encoding="utf-8")
               if l.startswith("REVERT_OPS_SECRET=")][0]
        req = urllib.request.Request(
            f'{os.environ.get("DL_DROP_URL", "")}/{name}',
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-ops-secret": sec})
        urllib.request.urlopen(req, timeout=45).read()
        print(f"  uploaded {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  upload failed ({e}) — local copy is authoritative")

for cfg in DL_EXPERIMENTS:
    if cfg["name"] not in WANT:
        continue
    out = RUNS / f"{cfg['name']}.json"
    if out.exists():
        print(f"skip {cfg['name']} (already have it)")
        continue
    print(f"\n=== {cfg['name']} on CPU ===")
    t0 = time.time()
    res = run_dl_experiment(
        cfg, tokenizer, tr_ids, y_tr, X_tr, te_ids, y_te, X_te,
        runs_dir=RUNS, wandb_run=None)
    mf = res.get("macro_f1_external", res.get("macro_f1"))
    print(f"  macro_f1={mf:.4f}  ({(time.time()-t0)/60:.0f} min)")
    upload(cfg["name"], res)

print("\nlocal finisher done")
