# Data Provenance (v2 corpus)

## Upstream source

The raw corpus is the public HuggingFace dataset
[`mwritescode/slither-audited-smart-contracts`](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts)
by Martina Rossini: Ethereum smart contracts with deployed bytecode and
Slither static-analysis reports. A contract is positive for a class when the
corpus records at least one Slither finding whose detector maps to it; the 39
detector tags map to the eight classes (plus `ignore`/`safe`) via
[`label_mappings.json`](label_mappings.json). The mapping is by detector name
only and the released derivative carries just the eight binary labels, so no
severity or confidence filter is applied or recoverable: a positive means
Slither emitted a finding, not that it is reachable or exploitable. The eight
classes are this corpus's own label set, only partly aligned with DASP.

## v2 corpus construction

Built by `scripts/build_data_v2.py`. The v2 corpus fixes the v1 leakage by
deduplicating on **(a)** the stored instruction text **and** **(b)** the
67-dimensional feature vector emitted by `src/evm_extractor.py` (canonical
column order in [`feature_columns.json`](feature_columns.json)).

Pass (a) is named "metadata-stripped bytecode" in the script, in its step
label and in the manifests, and that name overstates it. The `bytecode`
column of the upstream feature files holds space-separated opcode mnemonics
produced by the legacy converter, not hexadecimal runtime bytecode, so the
CBOR-trailer heuristic in that step never fires: re-applying it to the
released inputs alters no row in 31,670 sampled contracts. Pass (a) is
therefore exact-duplicate removal on instruction text. The step label and the
recorded manifests are left as the pipeline wrote them; this note is the
correction. Pass (b) is the one that bears the weight, since it operates on
exactly what the models see.

| Step | Rows |
|---|---:|
| Loaded (v1 train + v1 val) | 116,697 |
| After dropping empty/degenerate bytecode | 116,697 |
| After dedup on stored instruction text (labelled "metadata-stripped" in the manifest) | 115,839 |
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
