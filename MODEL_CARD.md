---
license: apache-2.0
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
model-index:
  - name: evm-bytecode-vulnerability-screening
    results:
      - task:
          type: binary-classification
          name: Smart-contract any-vulnerability screening
        dataset:
          name: slither-audited-smart-contracts derived feature matrix
          type: tabular
          split: validation
        metrics:
          - type: f1
            value: 0.947
            name: F1
          - type: precision
            value: 0.933
            name: Precision
          - type: recall
            value: 0.962
            name: Recall
          - type: matthews_correlation_coefficient
            value: 0.826
            name: MCC
          - type: false_negative_rate
            value: 0.038
            name: FNR
---

# EVM Bytecode Vulnerability Screening Model Card

This model card documents the machine-learning pipeline accompanying:

> S. S. Solovev. *Lightweight Machine Learning for Smart-Contract
> Vulnerability Detection from EVM Bytecode: Binary and Multi-Label
> Classification with a Deep-Learning Comparator.* Preprint, 2026.

The model is intended as a first-stage screening system for EVM bytecode. It
estimates agreement with Slither-derived vulnerability labels and ranks
contracts for deeper analysis. It is not a final security verdict.

## Model Family

- Binary task: RandomForest and XGBoost classifiers over engineered bytecode
  features.
- Multi-label task: XGBoost over the same feature representation.
- Comparator: Conv-Transformer ablation over opcode sequences.

The released feature extractor is documented in
[`FEATURE_SCHEMA.md`](FEATURE_SCHEMA.md).

## Reported Validation Metrics

Metrics are centralised in [`results/metrics.json`](results/metrics.json).

| Task | Model | F1 / macro-F1 | Precision | Recall | FNR | MCC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary | RandomForest | 0.947 | 0.933 | 0.962 | 0.038 | 0.826 | 0.991 |
| Binary | XGBoost + Optuna | 0.948 | 0.945 | 0.950 | 0.050 | 0.832 | 0.990 |
| Binary | CatBoost | 0.918 | 0.951 | 0.888 | 0.112 | 0.761 | 0.983 |
| Multi-label | XGBoost | 0.7746 | - | - | - | - | - |
| Multi-label | best Conv-Transformer | 0.7302 | - | - | - | - | - |

The comparison supports a conservative conclusion: for this engineered
tabular bytecode representation, tree-based models are a strong baseline and
outperform the tested deep-learning comparator on the multi-label task.

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
- The split is stratified but not temporal, compiler-version stratified, or
  address-family grouped.
- Proxy, clone, and compiler-family correlations may remain after exact
  bytecode-hash deduplication.
- The model should be reported as estimating Slither-consistency, not
  human-audit ground truth.

## Artefact Availability

This GitHub repository contains documentation and the standalone feature
extractor. Full end-to-end execution is provided through the Kaggle notebook:

https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode

If trained model weights are published separately, their artefact page should
state the exact feature schema and model version used for inference.
