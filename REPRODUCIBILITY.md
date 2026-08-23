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
| 1 | `scripts/build_data_v2.py` | Dedup on metadata-stripped bytecode + 67-d feature vector; stratified 80/10/10 split; writes parquets + `data/manifest_v2.json` |
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
