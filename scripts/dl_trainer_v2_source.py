# -*- coding: utf-8 -*-
"""Paper v2 — Phase 3 trainer body (appended after the inlined dl_pipeline.py).

Runs the 10-config Conv-Transformer ablation on the v2 corpus:
  * train on train_v2 (internal 80/20 carve-out inside run_dl_experiment
    handles early stopping / B5 thresholds — that logic was correct in v1
    and is reused untouched);
  * report on TEST_v2 — the same held-out split the classical models report
    on, with zero feature-vector overlap against train (manifest_v2.json).

Resumable: finished configs live in runs_v2/{name}.json; attach a previous
run's output as dataset sc-vuln-dl-v2-cache to skip them. Set REVERSE_ORDER=1
in a second kernel to walk C4->A1 and halve wall-clock across two sessions.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REVERSE = bool(int(os.environ.get("REVERSE_ORDER", "0")))
TIME_BUDGET_SEC = int(os.environ.get("DL_TIME_BUDGET_SEC", str(11 * 3600)))

def _mount(slug, owner="sergeisolovyev"):
    """Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug> on the
    current image (observed 2026-08-21) but at /kaggle/input/<slug> on older
    ones. Resolve whichever exists so the kernel survives both layouts."""
    for c in (Path(f"/kaggle/input/datasets/{owner}/{slug}"),
              Path(f"/kaggle/input/{slug}")):
        if c.exists():
            return c
    return Path(f"/kaggle/input/{slug}")

DATA = _mount("defi-bytecode-features-v2")
CACHE_IN = _mount("sc-vuln-dl-v2-cache")
RUNS = Path("/kaggle/working/runs_v2")
RUNS.mkdir(parents=True, exist_ok=True)

if CACHE_IN.exists():
    for p in CACHE_IN.glob("*.json"):
        if not (RUNS / p.name).exists():
            shutil.copy(p, RUNS / p.name)
    print("resumed cache:", sorted(x.name for x in RUNS.glob("*.json")))

LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
FEATURES = json.load(open(DATA / "feature_columns.json", encoding="utf-8"))
assert len(FEATURES) == 67

print("loading v2 parquets...")
cols = ["bytecode"] + FEATURES + LABELS
tr = pd.read_parquet(DATA / "train_v2.parquet", columns=cols)
te = pd.read_parquet(DATA / "test_v2.parquet", columns=cols)
print(f"train {len(tr):,}  test {len(te):,}")

print("fitting tokenizer (first 20K train rows, as in v1)...")
tokenizer = BytecodeTokenizer().fit(tr["bytecode"].head(20000))
print("  vocab:", tokenizer.vocab_size)

print("encoding sequences...")
tr_ids = build_token_ids(tr["bytecode"], tokenizer)
te_ids = build_token_ids(te["bytecode"], tokenizer)

y_tr = tr[LABELS].to_numpy("float32")
y_te = te[LABELS].to_numpy("float32")
X_tr = tr[FEATURES].to_numpy("float32")
X_te = te[FEATURES].to_numpy("float32")
del tr, te

print("CUDA:", torch.cuda.is_available())
print(f"configs: {len(DL_EXPERIMENTS)} | reverse={REVERSE} | budget={TIME_BUDGET_SEC}s")

# ── external result uploader ────────────────────────────────────────────────
# Kaggle batch kernels persist /kaggle/working ONLY on successful completion:
# a 12h-wall cancellation destroys everything (happened twice, 2026-08-22).
# A daemon thread watches runs_v2/ and POSTs every new config JSON to the
# REVERT ops-drop inbox, so finished work survives any session death.
import threading
import time as _time
import urllib.request as _rq

OPS = "https://revert-scan-api-5a5fd54430cb.herokuapp.com/v1/ops-drop"
OPS_SECRET = "<<OPS_SECRET>>"

def _upload_loop():
    sent = set()
    while True:
        for p in RUNS.glob("*.json"):
            if p.name in sent:
                continue
            try:
                body = p.read_bytes()
                req = _rq.Request(f"{OPS}/{p.stem}", data=body, headers={
                    "Content-Type": "application/json",
                    "x-ops-secret": OPS_SECRET})
                with _rq.urlopen(req, timeout=30) as r:
                    r.read()
                sent.add(p.name)
                print(f"[uploader] sent {p.name}")
            except Exception as e:  # noqa: BLE001
                print(f"[uploader] {p.name}: {e}")
        _time.sleep(120)

threading.Thread(target=_upload_loop, daemon=True).start()

dl_results = run_all_dl(
    tokenizer=tokenizer,
    train_token_ids=tr_ids,
    train_labels=y_tr,
    train_nums=X_tr,
    val_token_ids=te_ids,
    val_labels=y_te,
    val_nums=X_te,
    runs_dir=RUNS,
    wandb_run=None,
    time_budget_sec=TIME_BUDGET_SEC,
    reverse_order=REVERSE,
)

summary = {}
for p in sorted(RUNS.glob("*.json")):
    d = json.loads(p.read_text())
    summary[p.stem] = {"macro_f1": d.get("macro_f1"),
                       "f1_per_label": d.get("f1_per_label")}
json.dump(summary, open("/kaggle/working/dl_summary_v2.json", "w"), indent=2)
print(f"\ncached this far: {len(summary)}/10")
for k, v in sorted(summary.items()):
    print(f"  {k:32s} {v['macro_f1']}")
