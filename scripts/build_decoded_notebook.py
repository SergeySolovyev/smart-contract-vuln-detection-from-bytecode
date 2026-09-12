# -*- coding: utf-8 -*-
"""Generate 05_notebooks/decoded_opcodes_c2.ipynb (the decoded-opcode comparator run).

Builds ONE self-contained notebook that runs the two experiments the reviewers
asked for, on the released v2 data, without ever opening the test split:

  1. time the multi-label XGBoost fit the paper reports but never timed;
  2. train the paper's best deep configuration (C2) on correctly decoded EVM
     opcodes, and, as the control, on the legacy tokens, with identical code,
     seed and internal split; compare on the validation split.

The released rows do not carry the bytecode itself: the stored `bytecode` column
is the legacy converter's projection of the hex string (one named token per hex
character, PUSH operands removed), which is not invertible. The upstream corpus
(Rossini's slither-audited-smart-contracts) still carries the deployed bytecode,
and the projection is deterministic, so the notebook recomputes it on every
upstream contract and re-attaches the bytecode to the released rows by exact
match. It then decodes that bytecode properly.

The model code is the archive's dl_pipeline.py, embedded verbatim as a cell so
a reader sees exactly what ran. Every released input file is hashed against the
paper's manifest; the upstream corpus is pinned to a commit. Results are written
as JSON next to the notebook. The executed copy released in 05_notebooks/ was run
on Kaggle; run this script to regenerate the unexecuted source and diff it.

Locally: set V2_DATA_DIR to 01_data_v2 and UPSTREAM_DIR to a folder holding the
upstream parquet files (or let the notebook download them); SMOKE=1 for a tiny run.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import nbformat as nbf

A = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("DECODED_NOTEBOOK_OUT", str(A / "notebooks")))

pipeline_src = (A / "src" / "dl_pipeline.py").read_text(encoding="utf-8")
pipeline_sha = hashlib.sha256(pipeline_src.encode("utf-8")).hexdigest()
extractor_src = (A / "src" / "evm_extractor.py").read_text(encoding="utf-8")
extractor_sha = hashlib.sha256(extractor_src.encode("utf-8")).hexdigest()
features = json.loads((A / "results" / "feature_columns.json").read_text(encoding="utf-8"))
manifest = json.loads((A / "results" / "manifest_v2.json").read_text(encoding="utf-8"))
assert len(features) == 67
assert "'''" not in pipeline_src
assert "'''" not in extractor_src

UPSTREAM_REV = "13594107c7afa216cb0c126f38b8ff6548112dcf"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(f"""# Decoded-opcode comparator and classical timing

This notebook answers two questions left open by the paper
*Lightweight Machine Learning for Smart-Contract Vulnerability Detection from EVM Bytecode* (v2):

1. **How long does the winning classical model take to fit?** The paper reports the multi-label
   XGBoost result but never recorded its training time. Cell 6 fits it with the paper's exact
   hyperparameters on the full training split and times it.
2. **Does the deep comparator's loss survive a correct input?** The paper found that its
   Conv-Transformer was fed the *hexadecimal text* of each contract, disassembled character by
   character, rather than the decoded bytes. The released rows hold only that legacy text, and it is
   not invertible (the PUSH operands were removed). Cell 3 therefore goes back to the upstream corpus,
   which still carries the deployed bytecode, re-attaches it to every released row by recomputing the
   legacy projection and matching it exactly, and decodes the bytecode properly. Cells 7 and 8 then
   train the paper's best deep configuration (**C2**, `d_model=256`) on the decoded opcodes and, as
   the control, on the legacy tokens, with identical code, seed, rows and internal split.

A third question comes free once the bytecode is back. The paper's models see 67 engineered features,
and nothing had ever tied those columns to the contracts they describe. Cell 4b recomputes them from
the re-attached bytecode and checks each released column against its own extractor output.

**Protocol.** Same released `train_v2` / `val_v2` splits as the paper, verified by SHA-256 against
its manifest. Models are selected and early-stopped on an internal 20% carve-out of `train_v2`;
the reported score is on `val_v2`. **The test split is never opened** (Cell 2 asserts it).
This is therefore a validation-level comparison, not a new test-set result.

**Model code.** Cell 5 is the archive's `02_code/dl_pipeline.py`, embedded verbatim
(SHA-256 `{pipeline_sha}`), so the architecture, losses, optimiser, early stopping and metrics are
the ones behind the paper's Table and Figure 2. Cell 4 is the feature extractor, likewise verbatim.

**Where the paper's numbers come from.** The last cell prints every LaTeX macro this run feeds into
the paper next to the value and the cell that produced it.

**Upstream corpus.** `mwritescode/slither-audited-smart-contracts` on the Hugging Face Hub,
raw files `data/raw/contracts0..8.parquet`, pinned to commit `{UPSTREAM_REV}`.
Only the `contracts` (address) and `bytecode` columns are read.

**Reference points from the paper's released artifacts** (not recomputed here): legacy C2 on
validation 0.6578 and XGBoost on validation 0.7518, from the released end-to-end run
(`results/full_run_v12/full-ranking.json`); legacy C2 on test 0.6793 (paper, Sec. 6).
"""))

cells.append(nbf.v4.new_code_cell(f"""# 1. Configuration and environment
import os, sys, json, time, hashlib, gc, platform, re
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

SMOKE = os.environ.get("SMOKE", "0") == "1"          # tiny CPU run to check the notebook end to end
RUN_LEGACY_CONTROL = os.environ.get("RUN_LEGACY_CONTROL", "1") == "1"
SEED = 42

KAGGLE_IN = Path("/kaggle/input/defi-bytecode-features-v2")
LOCAL_IN = Path(os.environ.get("V2_DATA_DIR", "../data"))
DATA = KAGGLE_IN if KAGGLE_IN.exists() else LOCAL_IN
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = OUT_DIR / "runs"; RUNS_DIR.mkdir(exist_ok=True)

UPSTREAM_REPO = "mwritescode/slither-audited-smart-contracts"
UPSTREAM_REV = "{UPSTREAM_REV}"
# The upstream corpus is 1.75 GB; anything under /kaggle/working becomes kernel output, so it
# goes to the session's scratch space instead.
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else OUT_DIR
UPSTREAM_DIR = Path(os.environ.get("UPSTREAM_DIR", str(SCRATCH / "upstream")))

FEATURES = {json.dumps(features)}
LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
EXPECTED_SHA = {{"train_v2.parquet": "{manifest['sha256_train']}",
                "val_v2.parquet": "{manifest['sha256_val']}"}}
EXPECTED_ROWS = {{"train_v2.parquet": {manifest['n_train']}, "val_v2.parquet": {manifest['n_val']}}}

try:
    import pyevmasm  # noqa: F401
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyevmasm==0.2.3"], check=True)
import torch, sklearn, xgboost
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("xgboost", xgboost.__version__, "| sklearn", sklearn.__version__, "| cpus", os.cpu_count())
print("data dir:", DATA, "| SMOKE:", SMOKE, "| legacy control:", RUN_LEGACY_CONTROL)
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Inputs, bound to the paper's manifest. The test split is never touched.
def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def split_path(name):
    assert "test" not in name, "the test split is not part of this protocol"
    return DATA / name

def check_split(name):
    p = split_path(name)
    digest, rows = sha256_file(p), pq.ParquetFile(p).metadata.num_rows
    ok = digest == EXPECTED_SHA[name] and rows == EXPECTED_ROWS[name]
    print(f"{name}: rows {rows:,} | sha256 {digest[:16]}... | matches manifest: {ok}")
    assert ok, "released split does not match the paper's manifest"

def iter_text(name, n_rows=None):
    \"\"\"Stream the legacy text column row by row; never holds the whole column.\"\"\"
    seen = 0
    for b in pq.ParquetFile(split_path(name)).iter_batches(batch_size=1024, columns=["bytecode"]):
        for s in b.column("bytecode").to_pylist():
            if n_rows is not None and seen >= n_rows:
                return
            yield s; seen += 1

for name in ("train_v2.parquet", "val_v2.parquet"):
    check_split(name)

# Smoke mode only shrinks the row counts: 2,000/500 rows keep every label positive for the
# classical fit, and the deep runs take the first 240/80 of them.
N_XGB = (2000, 500) if SMOKE else (None, None)
N_DL = (240, 80) if SMOKE else (None, None)
train = pd.read_parquet(split_path("train_v2.parquet"), columns=FEATURES + LABELS)[: N_XGB[0]]
val = pd.read_parquet(split_path("val_v2.parquet"), columns=FEATURES + LABELS)[: N_XGB[1]]
X_tr, y_tr = train[FEATURES].to_numpy("float32"), train[LABELS].to_numpy("float32")
X_va, y_va = val[FEATURES].to_numpy("float32"), val[LABELS].to_numpy("float32")
n_tr_dl, n_va_dl = (N_DL[0] or len(train)), (N_DL[1] or len(val))
print("train", X_tr.shape, "| val", X_va.shape, "| positive rate", round(float((y_tr.max(1) > 0).mean()), 4),
      "| rows for the deep runs:", n_tr_dl, "/", n_va_dl)
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Recover the deployed bytecode from the upstream corpus, then decode it properly.
#
# What the released text is. The legacy converter disassembled the hexadecimal STRING: each hex
# character became one opcode byte (ASCII '0'..'9' -> 0x30..0x39 = ADDRESS..CODECOPY, 'a'..'f' ->
# 0x61..0x66 = PUSH2..PUSH7) and each PUSHn swallowed the following n characters as its operand; the
# operands were then removed, and a trailing PUSH whose operand ran past the end of the string was
# dropped. That projection loses the swallowed characters, so the text cannot be inverted -- but it
# can be recomputed from the original hex string, which gives an exact join key.
NAME = {"0": "ADDRESS", "1": "BALANCE", "2": "ORIGIN", "3": "CALLER", "4": "CALLVALUE",
        "5": "CALLDATALOAD", "6": "CALLDATASIZE", "7": "CALLDATACOPY", "8": "CODESIZE",
        "9": "CODECOPY", "a": "PUSH2", "b": "PUSH3", "c": "PUSH4", "d": "PUSH5", "e": "PUSH6", "f": "PUSH7"}
SKIP = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}

PREFIX_TOKENS = 120
# Hex characters behind one projected token: a digit stands for itself, PUSHn also swallowed n.
TOKEN_CHARS = {NAME[c]: 1 + SKIP.get(c, 0) for c in NAME}
MAX_DROPPED = 7               # a trailing PUSH the converter could not complete

def legacy_projection(h, limit=None):
    out, i, n = [], 0, len(h)
    while i < n and (limit is None or len(out) < limit):
        c = h[i]; w = SKIP.get(c, 0)
        if i + w >= n: break        # a trailing PUSH without its full operand was dropped by the converter
        out.append(NAME[c]); i += 1 + w
    return " ".join(out)

def clean_tokens(text):       # released text: drop the rare surviving operand token (only ever the last one)
    return text.split() if "0x" not in text else [t for t in text.split() if not t.startswith("0x")]

def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()

t0 = time.time()
keys, prefixes = {}, {}
for split, name, n_rows in (("train", "train_v2.parquet", n_tr_dl), ("val", "val_v2.parquet", n_va_dl)):
    keys[split] = []
    for s in iter_text(name, n_rows):
        toks = clean_tokens(s)
        key = sha1(" ".join(toks))
        keys[split].append(key)
        implied = sum(TOKEN_CHARS[t] for t in toks)     # hex characters this row accounts for
        prefixes.setdefault(sha1(" ".join(toks[:PREFIX_TOKENS])), {})[key] = implied
wanted = set(keys["train"]) | set(keys["val"])
print(f"released rows keyed: train {len(keys['train']):,} | val {len(keys['val']):,} | "
      f"distinct keys {len(wanted):,} | distinct {PREFIX_TOKENS}-token prefixes {len(prefixes):,} | {time.time()-t0:.0f}s")

def upstream_file(i):
    for p in (UPSTREAM_DIR / f"contracts{i}.parquet", UPSTREAM_DIR / "data" / "raw" / f"contracts{i}.parquet"):
        if p.exists(): return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(UPSTREAM_REPO, f"data/raw/contracts{i}.parquet", repo_type="dataset",
                                revision=UPSTREAM_REV, local_dir=str(UPSTREAM_DIR)))

# Each upstream contract is projected to PREFIX_TOKENS tokens, which is cheap; the full
# projection is paid for only when both that prefix and the implied hex length match a released
# row. The prefix alone rejects little -- every solc contract opens the same way -- so the length
# is what makes the scan cheap.
t0 = time.time()
code_by_key, n_up, n_unusable, n_collide, n_candidates = {}, 0, 0, 0, 0
for i in range(9):
    pf = pq.ParquetFile(upstream_file(i))
    for b in pf.iter_batches(batch_size=2048, columns=["contracts", "bytecode"]):
        for addr, h in zip(b.column("contracts").to_pylist(), b.column("bytecode").to_pylist()):
            n_up += 1
            h = (h[2:] if h and h.startswith("0x") else h or "").lower()
            if not h or len(h) % 2 or not re.fullmatch(r"[0-9a-f]+", h):
                n_unusable += 1; continue
            hits = prefixes.get(sha1(legacy_projection(h, PREFIX_TOKENS)))
            if not hits: continue
            # length is exact and free: reject before paying for the full projection
            hits = {k for k, implied in hits.items() if 0 <= len(h) - implied <= MAX_DROPPED}
            if not hits: continue
            n_candidates += 1
            k = sha1(legacy_projection(h))
            if k in hits:
                code = bytes.fromhex(h)
                if k in code_by_key: n_collide += code_by_key[k][1] != code
                else: code_by_key[k] = (addr, code)
    print(f"  contracts{i}: upstream rows {n_up:,} | re-attached keys {len(code_by_key):,}/{len(wanted):,}")
recovery = {"upstream_rows": n_up, "upstream_unusable": n_unusable, "released_keys": len(wanted),
            "keys_reattached": len(code_by_key), "same_projection_different_bytecode": n_collide,
            "prefix_candidates": n_candidates, "upstream_revision": UPSTREAM_REV}
for split in ("train", "val"):
    hit = sum(k in code_by_key for k in keys[split])
    recovery[f"{split}_rows"] = len(keys[split]); recovery[f"{split}_rows_reattached"] = hit
    print(f"{split}: {hit:,}/{len(keys[split]):,} rows re-attached ({100*hit/len(keys[split]):.2f}%)")
print(f"upstream contracts {n_up:,} | unusable {n_unusable:,} | projection collisions with different bytecode: {n_collide} | {time.time()-t0:.0f}s")
# `code_by_key` holds the re-attached bytecode of every matched row; cell 7 frees it as soon as
# the decoded arm is encoded.

# Proper decoding: bytes -> one mnemonic per instruction (pyevmasm's Istanbul table), PUSH operands
# skipped (constants and addresses would make the vocabulary open-ended), unknown bytes -> INVALID.
# The Solidity metadata trailer (CBOR, length in the last two bytes) is not code and is stripped when
# its marker is where the declared length says it is.
from pyevmasm import instruction_tables
_T = instruction_tables["istanbul"]; OPNAME = {}
for op in range(256):
    try: OPNAME[op] = _T[op].name
    except KeyError: pass

def strip_metadata(code):
    if len(code) >= 4:
        n = int.from_bytes(code[-2:], "big")
        if 8 <= n <= 120 and n + 2 < len(code):
            trailer = code[-(n + 2):]
            if b"ipfs" in trailer or b"bzzr" in trailer:
                return code[: -(n + 2)], True
    return code, False

def decode_opcodes(code):
    out, i, n = [], 0, len(code)
    while i < n:
        op = code[i]; out.append(OPNAME.get(op, "INVALID")); i += 1 + (op - 0x5f if 0x60 <= op <= 0x7f else 0)
    return " ".join(out)

def decoded_text(k):
    code, stripped = strip_metadata(code_by_key[k][1])
    return decode_opcodes(code), stripped

# sanity on the re-attached bytecode: the solc preamble PUSH1 0x80 PUSH1 0x40 MSTORE, and no
# INVALID/STOP runs that a mis-aligned decode would produce
sample = [k for k in keys["train"][:2000] if k in code_by_key]
dec = [decoded_text(k) for k in sample]
pre = sum(code_by_key[k][1][:5] in (bytes.fromhex("6080604052"), bytes.fromhex("6060604052")) for k in sample)
lens = np.array([d.count(" ") + 1 for d, _ in dec]); inv = np.array([d.split().count("INVALID") for d, _ in dec])
recovery.update({"sample_rows": len(sample), "sample_solc_preamble": int(pre), "sample_metadata_stripped": int(sum(s for _, s in dec)),
                 "sample_decoded_len_median": int(np.median(lens)), "sample_invalid_per_contract_median": float(np.median(inv))})
print(f"sample of {len(sample):,} train rows: solc preamble {pre:,} | metadata stripped {sum(s for _, s in dec):,} | "
      f"decoded tokens median {int(np.median(lens)):,} (max {lens.max():,}) | INVALID per contract median {np.median(inv):.0f}")
print("example:", dec[0][0][:160])
del sample, dec; gc.collect()
"""))

cells.append(nbf.v4.new_markdown_cell(f"""## 4. Do the released features describe these contracts?

The paper's models see all 67 engineered features, not the sequences, and until now nothing
tied those columns to the actual deployed bytecode: the paper says so in its limitations. The
re-attached bytecode makes the check possible. The released columns are standardised, so they cannot
be compared value by value; standardisation is affine and increasing, so if a released column is the
extractor's output for the same contract then a least-squares fit of the released column on the
recomputed one has $R^2 = 1$ and a positive slope. Anything less means the column is not that
function of this bytecode.

The next cell is the feature extractor, embedded verbatim
(SHA-256 `{extractor_sha}`), followed by the check on a sample of training contracts."""))
cells.append(nbf.v4.new_code_cell(extractor_src))

cells.append(nbf.v4.new_code_cell("""# 4b. The check itself.
N_PROV = 40 if SMOKE else int(os.environ.get("N_PROVENANCE", "1200"))
prov_keys = [k for k in keys["train"] if k in code_by_key][:N_PROV]
prov_rows = [i for i, k in enumerate(keys["train"]) if k in code_by_key][:N_PROV]
prov_hex = [code_by_key[k][1].hex() for k in prov_keys]

t0 = time.time()
extractor = EVMBytecodeFeatureExtractor(bytecode_column="bytecode", n_workers=os.cpu_count())
mine = extractor.transform(pd.DataFrame({"bytecode": prov_hex}))
if not isinstance(mine, pd.DataFrame):
    mine = pd.DataFrame(mine, columns=extractor.feature_names_)
rel = train[FEATURES].iloc[prov_rows].reset_index(drop=True)
print(f"recomputed {mine.shape[0]:,} contracts x {mine.shape[1]} features in {time.time()-t0:.0f}s")

exact, imperfect, flat = [], [], []
for f in FEATURES:
    x, y = mine[f].to_numpy("float64"), rel[f].to_numpy("float64")
    if np.std(x) == 0 or np.std(y) == 0:
        flat.append(f); continue
    a, b = np.polyfit(x, y, 1)
    r2 = 1 - float(np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2))
    (exact if r2 > 1 - 1e-9 and a > 0 else imperfect).append((f, r2, float(a)))
print(f"released = affine(recomputed) exactly: {len(exact)}/{len(FEATURES)} features")
print(f"constant across the sample (no test possible): {len(flat)} {flat}")
for f, r2, a in sorted(imperfect, key=lambda t: t[1])[:10]:
    print(f"   NOT an affine image: {f:34s} R2 {r2:.6f} slope {a:.4g}")
provenance = {"rows": len(prov_hex), "features_exact": len(exact), "features_total": len(FEATURES),
              "features_constant_in_sample": flat,
              "features_not_affine": [{"feature": f, "r2": r2, "slope": a} for f, r2, a in imperfect]}
(OUT_DIR / "feature_provenance.json").write_text(json.dumps(provenance, indent=2))
del mine, rel, prov_hex; gc.collect()
"""))

cells.append(nbf.v4.new_markdown_cell(f"""## 5. Model code, verbatim from the archive

The next cell is `02_code/dl_pipeline.py` from the paper's archive, unchanged
(SHA-256 `{pipeline_sha}`). It defines the tokeniser, the Conv-Transformer,
the losses, the training loop with early stopping, the metrics, and the configuration registry
`DL_EXPERIMENTS` from which C2 is taken."""))
cells.append(nbf.v4.new_code_cell(pipeline_src))

cells.append(nbf.v4.new_code_cell("""# 6. Time the paper's multi-label XGBoost. Same constructor as 02_code/05_analyze_v2.py, full training split.
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

xgb_ctor = lambda: MultiOutputClassifier(
    xgb.XGBClassifier(n_estimators=20 if SMOKE else 400, max_depth=8, n_jobs=-1,
                      random_state=SEED, eval_metric="logloss"), n_jobs=1)
m = xgb_ctor()
t0 = time.perf_counter(); m.fit(X_tr, y_tr.astype("int8")); xgb_fit_s = time.perf_counter() - t0
t0 = time.perf_counter(); P = m.predict(X_va); xgb_pred_s = time.perf_counter() - t0
xgb_val_macro = float(np.mean([f1_score(y_va[:, i], P[:, i]) for i in range(8)]))
print(f"multi-label XGBoost: fit {xgb_fit_s:.1f} s on {os.cpu_count()} CPUs | predict {len(X_va):,} rows in {xgb_pred_s:.2f} s | validation macro-F1 {xgb_val_macro:.4f}")
timing = {"model": "ML2_XGBoost", "n_estimators": 20 if SMOKE else 400, "max_depth": 8,
          "fit_seconds": xgb_fit_s, "predict_seconds_val": xgb_pred_s, "cpus": os.cpu_count(),
          "platform": platform.platform(), "val_macro_f1": xgb_val_macro, "train_rows": int(len(X_tr)), "val_rows": int(len(X_va))}
(OUT_DIR / "xgb_timing.json").write_text(json.dumps(timing, indent=2))
del m, P; gc.collect()
"""))

cells.append(nbf.v4.new_code_cell("""# 7. C2 on the decoded opcodes. The two arms are encoded and trained one after the other, not
#    together: the ragged token ids of both representations plus the re-attached bytecode would be
#    several gigabytes at once on a session that has about thirteen. The row set is fixed here,
#    before either arm, so both still see exactly the same contracts.
#    The conv stack has fixed strides (4*5*4*2 = 160) and a hard-coded 125-position mask, so
#    max_seq_len stays 20000 for both representations; smoke mode shrinks rows and epochs only.
C2 = next(c for c in DL_EXPERIMENTS if c["name"] == "C2_dmodel_256")
OVERRIDE = {"epochs": 1, "batch_size": 8} if SMOKE else {}
MAX_LEN = 20000

keep_tr = np.array([k in code_by_key for k in keys["train"]])
keep_va = np.array([k in code_by_key for k in keys["val"]])
y_tr_dl, X_tr_dl = y_tr[:n_tr_dl][keep_tr], X_tr[:n_tr_dl][keep_tr]
y_va_dl, X_va_dl = y_va[:n_va_dl][keep_va], X_va[:n_va_dl][keep_va]
recovery.update({"dl_train_rows": int(keep_tr.sum()), "dl_val_rows": int(keep_va.sum())})
print(f"rows for both arms: train {keep_tr.sum():,} | val {keep_va.sum():,}")

def encode(texts, tok):
    ids = [tok.encode_unpadded(s, MAX_LEN) for s in texts]
    lens = np.array([len(x) for x in ids])
    print(f"  encoded {len(ids):,} | length median {int(np.median(lens)):,} | truncated at {MAX_LEN}: {(lens >= MAX_LEN).sum():,}")
    return ids

def decoded_texts(split_keys):
    for k in split_keys:
        if k in code_by_key:
            yield decoded_text(k)[0]

def legacy_texts(name, keep, n_rows):
    \"\"\"Selects on the row mask, not on code_by_key: that map is freed with the decoded arm.\"\"\"
    for s, wanted in zip(iter_text(name, n_rows), keep):
        if wanted:
            yield s

def run_c2(tr_ids, va_ids, tok, tag):
    cfg = {**C2, **OVERRIDE, "name": f"C2_dmodel_256_{tag}"}
    t0 = time.time()
    res = run_dl_experiment(cfg, tok, tr_ids, y_tr_dl, X_tr_dl, va_ids, y_va_dl, X_va_dl, RUNS_DIR)
    res["wall_seconds_incl_eval"] = time.time() - t0
    res["representation"] = tag; res["external_split"] = "val_v2"; res["vocab_size"] = tok.vocab_size
    res["train_rows"] = int(len(tr_ids)); res["val_rows"] = int(len(va_ids))
    (OUT_DIR / f"c2_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{tag}] C2 validation macro-F1 {res['macro_f1_external']:.4f} | best epoch {res['best_epoch']} of {res['total_epochs']} | {res['train_time_sec']/60:.1f} min")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return res

# The tokeniser is fitted on the first 20,000 training rows, as in the paper. `fit` only iterates,
# so it is fed a generator: materialising 20,000 legacy strings costs more than a gigabyte.
t0 = time.time()
tok_dec = BytecodeTokenizer().fit(decoded_texts(keys["train"][:20000]))
print(f"[decoded] vocabulary {tok_dec.vocab_size} | fitted in {time.time()-t0:.0f}s")
tr_dec = encode(decoded_texts(keys["train"]), tok_dec)
va_dec = encode(decoded_texts(keys["val"]), tok_dec)
metadata_stripped = sum(decoded_text(k)[1] for k in keys["train"] if k in code_by_key)
recovery["metadata_stripped_train"] = int(metadata_stripped)
del code_by_key; gc.collect()      # the bytecode is not needed once the decoded arm is encoded
res_decoded = run_c2(tr_dec, va_dec, tok_dec, "decoded")
del tr_dec, va_dec; gc.collect()
"""))

cells.append(nbf.v4.new_code_cell("""# 8. Control: the same C2 on the legacy tokens, same rows, seed and internal split. Only the
#    representation differs. Encoded here rather than in cell 7 so the two never share memory.
if RUN_LEGACY_CONTROL:
    t0 = time.time()
    tok_leg = BytecodeTokenizer().fit(legacy_texts("train_v2.parquet", keep_tr[:20000], 20000))
    print(f"[legacy] vocabulary {tok_leg.vocab_size} | fitted in {time.time()-t0:.0f}s")
    tr_leg = encode(legacy_texts("train_v2.parquet", keep_tr, n_tr_dl), tok_leg)
    va_leg = encode(legacy_texts("val_v2.parquet", keep_va, n_va_dl), tok_leg)
    res_legacy = run_c2(tr_leg, va_leg, tok_leg, "legacy")
    del tr_leg, va_leg; gc.collect()
else:
    res_legacy = None
"""))

cells.append(nbf.v4.new_code_cell("""# 9. Results
ref = {"legacy_C2_val_full_run_v12": 0.6578, "xgb_val_full_run_v12": 0.7518, "legacy_C2_test_paper": 0.6793}
rows = [("XGBoost, this run (val)", xgb_val_macro, f"fit {xgb_fit_s:.0f} s on {os.cpu_count()} CPUs"),
        ("XGBoost, released run (val)", ref["xgb_val_full_run_v12"], "reference"),
        ("C2 decoded opcodes, this run (val)", res_decoded["macro_f1_external"], f"{res_decoded['train_time_sec']/60:.0f} min, {res_decoded['total_epochs']} epochs")]
if res_legacy:
    rows.append(("C2 legacy tokens, this run (val)", res_legacy["macro_f1_external"], f"{res_legacy['train_time_sec']/60:.0f} min, {res_legacy['total_epochs']} epochs"))
rows.append(("C2 legacy tokens, released run (val)", ref["legacy_C2_val_full_run_v12"], "reference"))
print(f"{'model':42s} {'macro-F1':>9s}   note"); print("-" * 72)
for name, v, note in rows: print(f"{name:42s} {v:9.4f}   {note}")
pick = lambda r: {k: r[k] for k in ("macro_f1_external", "best_epoch", "total_epochs", "train_time_sec", "vocab_size", "f1_per_label", "train_rows", "val_rows")}
summary = {"smoke": SMOKE, "external_split": "val_v2", "test_split_opened": False,
           "bytecode_recovery": recovery, "feature_provenance": provenance, "xgb_timing": timing,
           "c2_decoded": pick(res_decoded), "c2_legacy": pick(res_legacy) if res_legacy else None,
           "reference_points": ref, "dl_pipeline_sha256": "__PIPELINE_SHA__",
           "environment": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "torch": torch.__version__, "xgboost": xgboost.__version__, "python": platform.python_version()}}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print("\\nwritten:", sorted(p.name for p in OUT_DIR.glob("*.json")))

# Where each number this notebook contributes appears in the paper. The paper cites numbers only
# through LaTeX macros generated from these files, so this is the whole mapping for this run.
print("\\n" + "=" * 72)
print("every number this run contributes to the paper")
print("=" * 72)
gap = abs(100 * (res_decoded["macro_f1_external"] - res_legacy["macro_f1_external"])) if res_legacy else float("nan")
xgap = abs(100 * (xgb_val_macro - res_decoded["macro_f1_external"]))
hit = recovery["train_rows_reattached"] + recovery["val_rows_reattached"]
tot = recovery["train_rows"] + recovery["val_rows"]
for name, value, source in [
    ("vDecodedVal", f"{res_decoded['macro_f1_external']:.4f}", "cell 7, C2 on decoded opcodes, val_v2 macro-F1"),
    ("vLegacyVal", f"{res_legacy['macro_f1_external']:.4f}" if res_legacy else "-", "cell 8, same C2 on legacy tokens"),
    ("vDecXgbVal", f"{xgb_val_macro:.4f}", "cell 6, multi-label XGBoost, val_v2 macro-F1"),
    ("vXgbFitS", f"{xgb_fit_s:.0f}", "cell 6, wall clock of the XGBoost fit, seconds"),
    ("vXgbCpus", f"{os.cpu_count()}", "cell 6, CPU cores of this session"),
    ("vDecodedMin", f"{res_decoded['train_time_sec']/60:.0f}", "cell 7, GPU training minutes"),
    ("vLegacyMin", f"{res_legacy['train_time_sec']/60:.0f}" if res_legacy else "-", "cell 8, GPU training minutes"),
    ("vDecGapAbs", f"{gap:.2f}", "decoded minus legacy, percentage points"),
    ("vDecXgbGapAbs", f"{xgap:.2f}", "XGBoost minus decoded, percentage points"),
    ("vDecTrainRows", f"{res_decoded['train_rows']:,}", "cell 7, contracts both arms trained on"),
    ("vDecValRows", f"{res_decoded['val_rows']:,}", "cell 7, contracts both arms scored on"),
    ("vDecRowsPct", f"{100*hit/tot:.1f}%", "cell 3, released rows whose bytecode was re-attached"),
    ("vDecUpstream", f"{recovery['upstream_rows']:,}", "cell 3, upstream contracts scanned"),
    ("vDecVocab", f"{res_decoded['vocab_size']:,}", "cell 7, decoded-opcode vocabulary"),
    ("vDecLenMed", f"{recovery['sample_decoded_len_median']:,}", "cell 3, median decoded tokens per contract"),
]:
    print(f"  {name:16s} {value:>10s}   {source}")
print("\\nfeature provenance (cell 4b): "
      f"{provenance['features_exact']}/{provenance['features_total']} released features are an exact "
      f"affine image of the extractor's output on the re-attached bytecode")
print("reference points NOT recomputed here:", ref)
""".replace("__PIPELINE_SHA__", pipeline_sha)))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "decoded_opcodes_c2.ipynb"
nbf.write(nb, path)
print("wrote", path, "| cells", len(cells), "| dl_pipeline sha", pipeline_sha[:16],
      "| evm_extractor sha", extractor_sha[:16])
