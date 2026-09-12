# Reproducibility Guide

Two reproduction paths exist. The **v2 script pipeline in this repository
is the authoritative one**; the v1 Kaggle notebook is kept for provenance
only, because the v1 evaluation protocol was found defective (leaky split,
tuning on the reporting split — see the root `README.md`).

## v2 Pipeline (authoritative, rerun in progress)

Fixed constants of the v2 protocol:

- split seed: `376`
- feature schema size: `67` (see `FEATURE_SCHEMA.md`,
  `data/feature_columns.json`)
- corpus after dedup: `112,467` contracts
- split sizes: train `89,973` / val `11,247` / test `11,247`
- bootstrap resamples: `1,000`
- hyperparameter tuning: Optuna, 5-fold stratified CV **inside train only**
- thresholds: selected on val; final metrics reported on test **once**

Phases, in order:

| Phase | Script | What it does |
|---|---|---|
| 1 | `scripts/build_data_v2.py` | Dedup on the stored instruction text + 67-d feature vector; stratified 80/10/10 split; writes parquets + `data/manifest_v2.json` |
| 2 | `scripts/run_classical_v2.py` | Classical binary models under the honest protocol; per-model JSON + CSV + paired delta-F1 bootstrap |
| 3 | `scripts/build_kernel_v2.py` + `scripts/dl_trainer_v2_source.py` | Assembles and runs the 10-config Conv-Transformer ablation on Kaggle GPU (train on train_v2, report on test_v2) |
| 4 | `scripts/analyze_v2.py` | Paired statistics (exact p-values, Holm correction), per-label F1 heatmap |
| 5 | `scripts/check_numbers_v2.py` | Gate: every number in the paper must match a result artifact; exit 1 on mismatch |

Notes:

- Phase 1 consumes the v1 parquet feature matrices (public Kaggle dataset
  `sergeisolovyev/defi-bytecode-features-public`) as raw material; it does
  not re-download the HuggingFace corpus. Regenerated splits must match
  the sha256 values in `data/manifest_v2.json`.
- The scripts were extracted from the working tree and still contain
  absolute local paths (`D:\...`) and Kaggle-specific mount logic; they
  will be parameterised for the v2 release.
- `scripts/build_kernel_v2.py` injects an operational secret from a local
  env file **outside this repository** at build time;
  `scripts/dl_trainer_v2_source.py` accordingly contains the placeholder
  `<<OPS_SECRET>>`, never a real value. The telemetry ping it guards is
  optional and does not affect results.
- All metric values are generated from the artifacts in `results/`
  by `scripts/emit_cards_v2.py` (Markdown) and
  `scripts/emit_macros_v2.py` (LaTeX), and verified against the
  built paper by `scripts/check_numbers_v2.py`, which exits non-zero
  on any disagreement. No metric is typed by hand in this repo.
- The deep ablation covers 8 of 10 planned configurations;
  `C3_pure_cnn` and `C4_pure_transformer` did not complete within
  the GPU allocation for this run. `scripts/finish_local_cpu.py`
  trains exactly those two on CPU against the same splits, and the
  emit scripts pick them up automatically once present.

## The Executed Run (version 12)

The phases above are the authoritative pipeline, and their fitted models were
not retained. One complete execution of the author's two-mode notebook was
run on a Kaggle Tesla T4 and is released here in full:

| | |
|---|---|
| Notebook | [`notebooks/ML_Experiments_v2_full_run.ipynb`](notebooks/ML_Experiments_v2_full_run.ipynb) and an [HTML copy](notebooks/ML_Experiments_v2_full_run.html) |
| Artifacts | [`results/full_run_v12/`](results/full_run_v12/README.md) |
| Verifier | `python scripts/verify_full_run_v12.py` |
| Kernel | `sergeisolovyev/smart-contract-two-mode-full-run`, id 133095850, version 12 |
| Evidence of completion | 105 cells, 54 code cells, execution counts 1 to 54, zero error outputs |
| Protocol | 16 stages trained, selection frozen before any test access, test read once |

The verifier is the entry point: it rehashes every artifact, reads the
notebook's execution counts and recomputes each number the paper's Section 7
prints, exiting non-zero on any disagreement. It needs only the standard
library, and it trains nothing, loads no model and reads no dataset row.

Two limits are worth stating before anyone reuses this run. It reads the same
test split the rest of the paper reads, so it is a second execution of one
protocol rather than an independent confirmation. And the sequence models
consumed a hexadecimal-character encoding of the bytecode rather than decoded
opcodes; `results/full_run_v12/README.md` documents that finding and what it
does and does not affect.

## The Decoded-Opcode Run

| | |
|---|---|
| Notebook | [`notebooks/decoded_opcodes_c2.ipynb`](notebooks/decoded_opcodes_c2.ipynb) and an [HTML copy](notebooks/decoded_opcodes_c2.html) |
| Artifacts | [`results/decoded_c2/`](results/decoded_c2/README.md) |
| Generator | `python scripts/build_decoded_notebook.py` rebuilds the unexecuted source |
| Inputs | the released `train_v2`/`val_v2` parquet files, hashed against `manifest_v2.json`, plus the upstream corpus at a pinned commit |
| Protocol | selection and early stopping on an internal carve-out of the training split; scores on validation; the test split is never opened, and the notebook asserts it |

The notebook is self-contained: it embeds `src/dl_pipeline.py` and `src/evm_extractor.py`
verbatim, so what ran is visible without leaving the document, and its last cell prints
every number it contributes to the paper next to the cell that produced it.

## v1 Notebook (provenance only — numbers retracted)

The v1 end-to-end run lives in the Kaggle notebook and its attached public
datasets. Its recorded constants (global seed `42`, split seed `376`,
corpus `117,091`, validation size `11,670`) describe the **superseded v1
corpus**, and its metric outputs are retracted.

| Artefact | Purpose | URL |
|---|---|---|
| Raw Slither-labelled corpus | Upstream source contracts, bytecode, reports | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| Kaggle notebook (v1) | End-to-end v1 pipeline | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| Public feature matrices | Train/validation parquet matrices (raw material for Phase 1) | https://www.kaggle.com/datasets/sergeisolovyev/defi-bytecode-features-public |
| Public run cache (v1) | Cached v1 JSON outputs | https://www.kaggle.com/datasets/sergeisolovyev/smart-contract-vuln-run-cache |
| W&B project | Deep-learning ablation tracking | https://wandb.ai/sesesolovev-hse-university/defi-binary-vuln |

## Local Inspection

The repository is directly useful for inspecting and testing the bytecode
feature extractor:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows; POSIX: . .venv/bin/activate
pip install -r requirements.txt
pytest
```

The tests are intentionally small. They check extractor behaviour on empty,
invalid, and simple bytecode inputs and pin the 67-feature schema. They do
not retrain the models.
