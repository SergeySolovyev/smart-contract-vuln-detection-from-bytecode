# Data Provenance (v2 corpus)

## Upstream source

The raw corpus is the public HuggingFace dataset
[`mwritescode/slither-audited-smart-contracts`](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts)
by Martina Rossini: Ethereum smart contracts with deployed bytecode and
Slither static-analysis reports. The 39 Slither detector tags are mapped to
8 SWC-aligned classes (plus `ignore`/`safe`) via
[`label_mappings.json`](label_mappings.json).

## v2 corpus construction

Built by `scripts/build_data_v2.py`. The v2 corpus fixes the v1 leakage:
deduplication is performed on **(a)** metadata-stripped runtime bytecode
(removing CBOR metadata-trailer / constructor-argument twins) **and**
**(b)** the 67-dimensional feature vector emitted by
`src/evm_extractor.py` (canonical column order in
[`feature_columns.json`](feature_columns.json)).

| Step | Rows |
|---|---:|
| Loaded (v1 train + v1 val) | 116,697 |
| After dropping empty/degenerate bytecode | 116,697 |
| After dedup on metadata-stripped bytecode | 115,839 |
| After dedup on 67-d feature vector | **112,467** |

## Split

Stratified 80/10/10 train/val/test split, seed **376**:

| Split | Rows |
|---|---:|
| train | 89,973 |
| val | 11,247 |
| test | 11,247 |

Exact per-step counts, per-split positive rates, and the sha256 of each
parquet are recorded in [`manifest_v2.json`](manifest_v2.json). The parquet
files themselves (~600 MB total) are not stored in git; they are
regenerated deterministically by `scripts/build_data_v2.py` and must match
the manifest hashes.

## Roles of the splits (v2 protocol)

- **train** — model fitting; hyperparameter tuning via 5-fold stratified CV
  inside train only.
- **val** — operating-threshold selection.
- **test** — touched exactly once per model for the final report.

## Files

| File | Description |
|---|---|
| `label_mappings.json` | 39 Slither detectors -> 8 classes + `ignore`/`safe` |
| `feature_columns.json` | canonical ordered list of the 67 feature names |
| `manifest_v2.json` | v2 build manifest: step counts, seeds, split sha256 |
