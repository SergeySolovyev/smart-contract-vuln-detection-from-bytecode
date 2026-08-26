# -*- coding: utf-8 -*-
"""Train the two deep configs the Kaggle GPU quota could not deliver.

Runs headless on a Colab CLI runtime (`colab exec`), so it must be safe to
launch detached and to resume: every finished config is written to
runs_out/ AND printed in full to stdout between ONE_CONFIG markers. The
printed copy is what matters -- an earlier attempt lost C3's per-label
vector because it existed only on a VM disk that was recycled overnight,
while the log text survived in the page.

Comparability with the eight configs trained on Kaggle is enforced, not
assumed: same 67 features in the same order, same split sizes, and a
tokenizer fit on exactly the first 20000 train rows, which must yield a
498-token vocabulary. Any mismatch stops the run rather than producing
numbers that cannot share a table with the rest of the ablation.

Tokenisation streams batch by batch: the bytecode strings total ~8.8 GB and
materialising them at once peaked at 12.4 GB, which is what killed the
first free-tier attempt.
"""
import gc
import json
import os
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from dl_pipeline import (BytecodeTokenizer, DL_EXPERIMENTS,  # noqa: E402
                         run_dl_experiment)

LABELS = ["access-control", "arithmetic", "bad-randomness", "double-spending",
          "locked-ether", "other", "reentrancy", "unchecked-calls"]
WANT = {"C3_pure_cnn", "C4_pure_transformer"}
MAX_LEN = 20000
BASE = ("https://www.kaggle.com/api/v1/datasets/download/"
        "sergeisolovyev/defi-bytecode-features-v2")
FILES = {"train_v2.parquet": 516601748, "test_v2.parquet": 65239576}

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = pathlib.Path("runs_out")
RUNS.mkdir(exist_ok=True)


def log(msg):
    print(msg, flush=True)


def push(name, payload):
    """Third copy of a finished config, to an HTTP inbox.

    Sends the WHOLE result, never a trimmed view. An earlier emergency
    channel forwarded only the metrics to save bandwidth, and it did save
    the numbers when the session died -- but six configurations reached the
    archive without config, history or timing, and those fields cannot be
    recovered because the sessions that held them are gone. A rescue path
    that drops provenance rescues the report and loses the run.

    stdout survives the VM but only if someone is reading it; this survives
    even an unattended run that dies before the next poll. Enabled only when
    the caller supplies both env vars, so the public repo carries no secret
    and the script stays runnable by anyone without one.
    """
    url = os.environ.get("OPS_DROP_URL")
    secret = os.environ.get("OPS_DROP_SECRET")
    if not (url and secret):
        return
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/{name}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-ops-secret": secret})
        urllib.request.urlopen(req, timeout=60).read()
        log(f"  pushed {name} to inbox")
    except Exception as e:  # noqa: BLE001
        log(f"  inbox push failed ({e}); stdout copy still stands")


def fetch(name, size):
    dst = pathlib.Path("data_v2") / name
    dst.parent.mkdir(exist_ok=True)
    if dst.exists() and dst.stat().st_size == size:
        log(f"{name}: already present")
        return dst
    req = urllib.request.Request(f"{BASE}/{name}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (100 << 20) < (1 << 20):
                log(f"  {name}: {done/1e6:.0f}/{size/1e6:.0f} MB")
    got = dst.stat().st_size
    assert got == size, f"{name}: got {got} bytes, expected {size}"
    with open(dst, "rb") as f:
        assert f.read(4) == b"PAR1", f"{name}: not a parquet (server error?)"
    log(f"{name}: ready")
    return dst


def fit_tokenizer(path, n=20000):
    rows, pf = [], pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=4096, columns=["bytecode"]):
        rows.extend(b.column("bytecode").to_pylist())
        if len(rows) >= n:
            break
    tok = BytecodeTokenizer().fit(rows[:n])
    del rows, pf
    gc.collect()
    return tok


def encode_stream(path, tok, total):
    out, done, pf = [], 0, pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=2048, columns=["bytecode"]):
        chunk = b.column("bytecode").to_pylist()
        out.extend(tok.encode_unpadded(s, MAX_LEN) for s in chunk)
        done += len(chunk)
        del chunk, b
        if done % 20480 == 0:
            gc.collect()
            log(f"  encoded {done}/{total}")
    del pf
    gc.collect()
    return out


def main():
    log(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "no GPU visible"
    log(f"gpu: {torch.cuda.get_device_name(0)}")

    features = json.load(open(ROOT / "data" / "feature_columns.json"))
    assert len(features) == 67, f"expected 67 features, got {len(features)}"

    tr_path = fetch("train_v2.parquet", FILES["train_v2.parquet"])
    te_path = fetch("test_v2.parquet", FILES["test_v2.parquet"])
    n_tr = pq.ParquetFile(tr_path).metadata.num_rows
    n_te = pq.ParquetFile(te_path).metadata.num_rows
    log(f"train {n_tr:,}  test {n_te:,}")
    assert n_tr == 89973 and n_te == 11247, "unexpected split sizes"

    def small(p):
        df = pd.read_parquet(p, columns=features + LABELS)
        return (df[features].to_numpy("float32"),
                df[LABELS].to_numpy("float32"))

    X_tr, y_tr = small(tr_path)
    X_te, y_te = small(te_path)

    tok = fit_tokenizer(tr_path)
    log(f"vocab: {tok.vocab_size}")
    assert tok.vocab_size == 498, (
        f"vocab {tok.vocab_size} != 498 -- inputs differ from the Kaggle run, "
        "results would not be comparable. Stopping.")

    log("encoding train ...")
    tr_ids = encode_stream(tr_path, tok, n_tr)
    log("encoding test ...")
    te_ids = encode_stream(te_path, tok, n_te)
    gc.collect()

    for cfg in DL_EXPERIMENTS:
        if cfg["name"] not in WANT:
            continue
        name = cfg["name"]
        done_file = RUNS / f"{name}.json"
        if done_file.exists():
            log(f"{name}: already done, skipping")
            continue
        log(f"\n=== {name} ===")
        t0 = time.time()
        res = run_dl_experiment(cfg, tok, tr_ids, y_tr, X_tr,
                                te_ids, y_te, X_te,
                                runs_dir=RUNS, wandb_run=None)
        done_file.write_text(json.dumps(res))
        mf = res.get("macro_f1_external", res.get("macro_f1"))
        if mf is None:
            mf = float(np.mean(res["f1_per_label"]))
        log(f"{name} macro_f1={mf:.4f} ({(time.time()-t0)/60:.0f} min)")
        # Durable copy: stdout outlives the VM disk.
        log(f"ONE_CONFIG_START {name}")
        log(json.dumps(res))
        log(f"ONE_CONFIG_END {name}")
        push(name, res)
        torch.cuda.empty_cache()
        gc.collect()

    out = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json")}
    log("\nRESULTS_JSON_START")
    log(json.dumps(out))
    log("RESULTS_JSON_END")
    log(f"configs delivered: {len(out)} {sorted(out)}")


if __name__ == "__main__":
    main()
