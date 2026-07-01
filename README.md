# Smart-Contract Vulnerability Screening from EVM Bytecode

Research artefacts for:

> S. S. Solovev. *Lightweight Machine Learning for Smart-Contract
> Vulnerability Detection from EVM Bytecode: Binary and Multi-Label
> Classification with a Deep-Learning Comparator.* Preprint, 2026.

This repository accompanies a study of lightweight machine learning for
smart-contract vulnerability screening in the bytecode-only setting. The
main research question is whether an interpretable feature representation
extracted from EVM bytecode can reproduce Slither-derived vulnerability
labels well enough to serve as a first-stage risk-ranking filter before
more expensive static or symbolic analysis.

The repository is intentionally conservative: it documents the public data
source, the feature extractor, the reported metrics, and the limitations of
the label source. The executable end-to-end notebook remains the primary
reproducibility entry point.

## Research Contribution

The work makes three claims.

1. **Bytecode-only screening.** The pipeline operates on EVM bytecode rather
   than verified Solidity source, targeting the regime where source-required
   tools cannot be applied directly.
2. **Interpretable engineered features.** The released extractor emits an
   ordered feature vector based on opcode counts, control-flow statistics,
   gas-cost aggregates, external-call patterns, and SWC-related risk
   indicators.
3. **Classical models as a strong baseline.** On this tabular feature
   representation, tree-based ensembles match or outperform a
   Conv-Transformer comparator while requiring substantially less training
   compute.

The intended role of the model is not to replace audit tools or formal
verification. It is a Tier-1 pre-filter: a fast risk-ranking layer that can
prioritise contracts for deeper analysis.

## Reported Results

All headline metrics below are recorded in
[`results/metrics.json`](results/metrics.json).

| Task | Model | Metric | Value |
|---|---:|---:|---:|
| Binary any-vulnerability | RandomForest | F1 | 0.947 |
| Binary any-vulnerability | RandomForest | Recall | 0.962 |
| Binary any-vulnerability | RandomForest | FNR | 0.038 |
| Binary any-vulnerability | RandomForest | PR-AUC | 0.991 |
| Binary any-vulnerability | XGBoost + Optuna | F1 | 0.948 |
| Multi-label SWC | XGBoost | macro-F1 | 0.7746 |
| Multi-label SWC | best Conv-Transformer | macro-F1 | 0.7302 |

The binary results are evaluated on an 11,670-contract held-out split from
117,091 Slither-labelled Ethereum contracts after exact bytecode-hash
deduplication. The multi-label task covers eight SWC-aligned classes.

## Repository Contents

```text
.
├── DATASET_CARD.md
├── FEATURE_SCHEMA.md
├── MODEL_CARD.md
├── REPRODUCIBILITY.md
├── features/
│   └── evm_extractor.py
├── results/
│   └── metrics.json
├── tests/
│   └── test_evm_extractor.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility Entry Points

| Artefact | URL |
|---|---|
| Raw dataset | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| End-to-end Kaggle notebook | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| W&B ablation project | https://wandb.ai/sesesolovev-hse-university/defi-binary-vuln |
| GitHub repository | https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode |

For a full rerun, use the Kaggle notebook and follow
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). This repository keeps the
standalone feature extractor and documentation needed to inspect the method
without depending on the notebook UI.

## Limitations

- Labels are produced by Slither and are not human-audit ground truth.
- Reported performance is bounded by Slither's own false-positive and
  false-negative behaviour.
- The original split is stratified but not temporal, compiler-version
  stratified, or address-family grouped.
- Proxy and clone families may induce residual correlations across folds.
- The released extractor schema must match the feature matrix used for
  training; see [`FEATURE_SCHEMA.md`](FEATURE_SCHEMA.md).

## Citation

```bibtex
@misc{solovev2026smartcontract,
  author = {Solovev, S. S.},
  title  = {Lightweight Machine Learning for Smart-Contract Vulnerability
            Detection from {EVM} Bytecode: Binary and Multi-Label
            Classification with a Deep-Learning Comparator},
  year   = {2026},
  note   = {Preprint}
}
```

## License

- Code in this repository: MIT License.
- Paper text, figures, and documentation: CC-BY 4.0 unless noted otherwise.
- External datasets and model artefacts retain their own licences.
