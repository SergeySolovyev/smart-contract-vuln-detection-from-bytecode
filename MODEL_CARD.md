---
license: mit
tags:
  - smart-contracts
  - ethereum
  - evm
  - vulnerability-detection
  - security
  - tabular
  - xgboost
  - random-forest
datasets:
  - mwritescode/slither-audited-smart-contracts
language: en
library_name: scikit-learn
pipeline_tag: tabular-classification
---

# EVM Bytecode Vulnerability Screening Model Card

This model card documents the machine-learning pipeline accompanying:

> S. S. Solovev. *Lightweight Machine Learning for Smart-Contract
> Vulnerability Detection from EVM Bytecode: Binary and Multi-Label
> Classification with a Deep-Learning Comparator.* Preprint, 2026.

The model is intended as a first-stage screening system for EVM bytecode. It
estimates agreement with Slither-derived vulnerability labels and ranks
contracts for deeper analysis. It is not a final security verdict.

## Status: v1 metrics retracted, v2 rerun in progress

All previously published metric values (v1) are **retracted**: the v1
evaluation used a leaky split (4.76% of validation rows feature-identical
to train after metadata-trailer variation) and tuned hyperparameters and
thresholds on the reporting split. No v1 number, including any previously
listed here or in the README, should be cited.

## Evaluation Protocol (v2)

- Corpus: 112,467 contracts after dedup on metadata-stripped bytecode and
  on the 67-d feature vector (see `data/README.md`).
- Split: stratified 80/10/10 train/val/test (89,973 / 11,247 / 11,247),
  seed 376.
- Hyperparameter tuning: Optuna with **5-fold stratified CV inside train
  only**.
- Operating thresholds: selected on **val**.
- Reporting: **test, touched exactly once per model**; B=1000 stratified
  percentile bootstrap CIs; paired delta-F1 bootstrap for model
  comparisons.

## Model Family

- Binary task: RandomForest, XGBoost, CatBoost, and LogisticRegression
  classifiers over engineered bytecode features.
- Multi-label task: XGBoost and RandomForest over the same feature
  representation.
- Comparator: Conv-Transformer ablation (10 configurations) over opcode
  sequences.

The released feature extractor is documented in
[`FEATURE_SCHEMA.md`](FEATURE_SCHEMA.md).

## Metrics (v2 — pending)

| Task | Model | F1 / macro-F1 (test) |
|---|---|---|
| Binary | RandomForest | TBD (v2 rerun in progress, 2026-08-22) |
| Binary | XGBoost (Optuna, CV-in-train) | TBD (v2 rerun in progress, 2026-08-22) |
| Binary | CatBoost | TBD (v2 rerun in progress, 2026-08-22) |
| Binary | LogisticRegression | TBD (v2 rerun in progress, 2026-08-22) |
| Multi-label | XGBoost | TBD (v2 rerun in progress, 2026-08-22) |
| Multi-label | RandomForest | TBD (v2 rerun in progress, 2026-08-22) |
| Multi-label | best Conv-Transformer | TBD (v2 rerun in progress, 2026-08-22) |

Values will be filled from the v2 result artifacts in `results/` and
cross-checked against the paper by `scripts/check_numbers_v2.py`.

## Intended Use

- Prioritising large contract corpora for manual review or deeper static
  analysis.
- Bytecode-only screening when verified Solidity source is unavailable.
- A research baseline for EVM bytecode vulnerability detection.

## Out of Scope

- Final audit verdicts.
- Exploit-path generation.
- Formal verification.
- Detection of vulnerability classes absent from, or weakly represented in,
  the Slither-derived label source.

## Limitations

- Labels come from Slither static analysis and inherit its false positives and
  false negatives.
- The v2 split is stratified and duplicate-controlled but not temporal,
  compiler-version stratified, or address-family grouped.
- Proxy, clone, and compiler-family correlations may remain after
  deduplication.
- The model should be reported as estimating Slither-consistency, not
  human-audit ground truth.

## Artefact Availability

This repository contains the paper sources, the standalone feature
extractor, the full v2 pipeline scripts, and the split manifest. The v1
end-to-end Kaggle notebook remains available for provenance:

https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode

If trained model weights are published separately, their artefact page should
state the exact feature schema and model version used for inference.
