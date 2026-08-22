# -*- coding: utf-8 -*-
"""Paper v2 — Phase 1: clean corpus with feature-level dedup and a 3-way split.

Inputs (the v1 parquets carry full raw material — bytecode, opcodes, the 67
canonical features and the 8 labels — so v2 is built without re-downloading HF):
    kaggle_dataset/train_dl_features.parquet  (105,027 rows)
    kaggle_dataset/val_dl_features.parquet    ( 11,670 rows)

What v1 got wrong and v2 fixes, in order:
  1. v1 deduplicated on the raw bytecode string only. Contracts differing only
     in the CBOR metadata trailer (solc fingerprint) or constructor args slip
     through and land on both sides of the split: measured 555 val rows
     (4.76%) bit-identical in feature space to train rows. v2 dedups on
     (a) metadata-stripped runtime bytecode AND (b) the 67-d feature vector.
  2. v1 had train/val only, and Optuna tuned ON val while the paper reported
     ON val. v2 makes a stratified 80/10/10 train/val/test split: tuning and
     thresholds on val, one final report on test.
Every dropped row is counted and written to the manifest — no silent drops.

Outputs (v2/ directory):
    train_v2.parquet, val_v2.parquet, test_v2.parquet
    manifest_v2.json          — exact counts of every step, sha256 of splits
    label_mappings.json       — copied verbatim (39 Slither detectors -> classes)
"""
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"D:\DeFi\Научный_телеграф\kaggle_paper")
OUT = BASE / "v2"
OUT.mkdir(exist_ok=True)

SEED = 376  # same seed family as v1 for continuity
LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
FEATURES = json.load(open(
    r"D:\DeFi\tmp\kaggle_solovev_output_v11\results\feature_columns.json",
    encoding="utf-8"))
assert len(FEATURES) == 67

manifest = {"seed": SEED, "n_features": 67, "labels": LABELS, "steps": []}

def step(name, n):
    manifest["steps"].append({"step": name, "rows": int(n)})
    print(f"{name:58s} {n:>8,}")

# ── load ────────────────────────────────────────────────────────────────────
cols = ["bytecode", "opcodes_clean"] + FEATURES + LABELS
tr = pd.read_parquet(BASE / "kaggle_dataset/train_dl_features.parquet", columns=cols)
va = pd.read_parquet(BASE / "kaggle_dataset/val_dl_features.parquet", columns=cols)
df = pd.concat([tr, va], ignore_index=True)
step("loaded (v1 train + v1 val)", len(df))

# ── drop empty / degenerate bytecode ────────────────────────────────────────
df = df[df["bytecode"].str.len() >= 4]
step("after dropping empty/degenerate bytecode", len(df))

# ── dedup 1: metadata-stripped runtime bytecode ─────────────────────────────
# Solidity appends a CBOR-encoded metadata blob whose length is stored in the
# final 2 bytes (big-endian, in bytes). Two builds of identical code differ
# only there. Strip it when the declared length is plausible, else keep as-is.
def strip_cbor(bc: str) -> str:
    if len(bc) < 8:
        return bc
    try:
        trailer_len = int(bc[-4:], 16) * 2 + 4  # hex chars incl. the length field
    except ValueError:
        return bc
    if 8 <= trailer_len <= 200 and trailer_len < len(bc):
        return bc[:-trailer_len]
    return bc

df["_norm"] = df["bytecode"].map(strip_cbor)
n0 = len(df)
df = df.drop_duplicates(subset="_norm", keep="first")
manifest["dropped_bytecode_dups"] = n0 - len(df)
step("after dedup on metadata-stripped bytecode", len(df))

# ── dedup 2: 67-d feature vector ────────────────────────────────────────────
# Different bytecode can still compile to an identical mnemonic profile; the
# model sees only the 67 features, so this is the dedup that actually
# prevents memorisation across the split.
n0 = len(df)
df = df.drop_duplicates(subset=FEATURES, keep="first")
manifest["dropped_feature_dups"] = n0 - len(df)
step("after dedup on 67-d feature vector", len(df))

df = df.drop(columns="_norm").reset_index(drop=True)
df["is_vulnerable"] = (df[LABELS].sum(axis=1) > 0).astype(int)
manifest["positives"] = int(df["is_vulnerable"].sum())
manifest["positive_rate"] = float(df["is_vulnerable"].mean())
print(f"{'positives (any-vulnerability)':58s} {manifest['positives']:>8,}"
      f"  ({manifest['positive_rate']:.3f})")

# ── stratified 80/10/10 ─────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
idx = np.arange(len(df))
strata = df["is_vulnerable"].values
tr_i, va_i, te_i = [], [], []
for s in (0, 1):
    part = idx[strata == s]
    rng.shuffle(part)
    n = len(part)
    a, b = int(n * 0.8), int(n * 0.9)
    tr_i.append(part[:a]); va_i.append(part[a:b]); te_i.append(part[b:])
tr_i = np.sort(np.concatenate(tr_i))
va_i = np.sort(np.concatenate(va_i))
te_i = np.sort(np.concatenate(te_i))

splits = {"train": df.iloc[tr_i], "val": df.iloc[va_i], "test": df.iloc[te_i]}

# ── invariant checks: the whole point of v2 ─────────────────────────────────
def fhash(d):
    a = d[FEATURES].to_numpy(dtype="float64")
    return set(hashlib.md5(r.tobytes()).hexdigest() for r in a)

h = {k: fhash(v) for k, v in splits.items()}
assert not (h["train"] & h["val"]), "feature leak train->val"
assert not (h["train"] & h["test"]), "feature leak train->test"
assert not (h["val"] & h["test"]), "feature leak val->test"
print("invariant: zero feature-vector overlap across all three splits  OK")

for name, part in splits.items():
    p = OUT / f"{name}_v2.parquet"
    part.reset_index(drop=True).to_parquet(p, engine="pyarrow", index=False)
    manifest[f"n_{name}"] = len(part)
    h256 = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h256.update(chunk)
    manifest[f"sha256_{name}"] = h256.hexdigest()
    manifest[f"positive_rate_{name}"] = float(part["is_vulnerable"].mean())
    step(f"wrote {name}_v2.parquet", len(part))

shutil.copy(r"D:\DeFi\DeFi_AI_dl_study\datasets\label_mappings.json",
            OUT / "label_mappings.json")
manifest["label_mapping"] = "label_mappings.json (39 Slither detectors -> 8 classes + ignore/safe)"
(OUT / "manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("\nmanifest_v2.json written")
