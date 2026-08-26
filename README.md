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

## Status: v2 rerun complete — v1 numbers retracted

**All previously reported metric values (v1) are retracted and must not be
cited.** An internal audit (2026-08) found two protocol defects in the v1
evaluation:

1. **Leaky split.** v1 deduplicated on the raw bytecode string only.
   Contracts differing only in the CBOR metadata trailer (solc fingerprint)
   or constructor arguments landed on both sides of the split: 4.76% of the
   v1 validation rows were bit-identical in feature space to training rows.
2. **Tuned on the reporting split.** v1 hyperparameter search (Optuna) and
   threshold selection used the same split on which results were reported.

This also explains the historical inconsistency between the README
(macro-F1 0.751) and the model card (0.775): both values came from
different v1 runs under the broken protocol, and both are withdrawn rather
than reconciled.

The **v2 protocol** fixes both defects: deduplication on metadata-stripped
runtime bytecode *and* on the 67-dimensional feature vector; a stratified
80/10/10 train/val/test split (seed 376); hyperparameter tuning via 5-fold
cross-validation inside train only; threshold selection on val; a single
final report on test. The v2 rerun is **complete**: every metric below comes
from that protocol, all ten deep configurations included, and each one is
generated from `results/` rather than typed.

## Research Contribution

1. **Bytecode-only screening.** The pipeline operates on EVM bytecode rather
   than verified Solidity source, targeting the regime where source-required
   tools cannot be applied directly.
2. **Interpretable engineered features.** The released extractor emits an
   ordered 67-feature vector based on opcode counts, control-flow
   statistics, gas-cost aggregates, external-call patterns, and SWC-related
   risk indicators.
3. **Classical models as a strong baseline.** The study compares tree-based
   ensembles against a Conv-Transformer comparator on the same corpus under
   the v2 protocol.

The intended role of the model is not to replace audit tools or formal
verification. It is a Tier-1 pre-filter: a fast risk-ranking layer that can
prioritise contracts for deeper analysis.

## Results (v2)

| Task | Model | Metric | Value |
|---|---|---|---|
| Binary any-vulnerability | XGBoost (Optuna, 5-fold CV in train) | F1 (test) | **0.9492** [0.9460, 0.9525], FNR 3.72% |
| Binary any-vulnerability | RandomForest | F1 (test) | **0.9452** [0.9416, 0.9487], FNR 4.19% |
| Binary any-vulnerability | CatBoost | F1 (test) | **0.9350** [0.9309, 0.9385], FNR 6.22% |
| Binary any-vulnerability | LogisticRegression | F1 (test) | **0.8900** [0.8862, 0.8939], FNR 6.08% |
| Multi-label SWC (8 classes) | XGBoost | macro-F1 (test) | **0.7646** |
| Multi-label SWC (8 classes) | RandomForest | macro-F1 (test) | 0.6908 |
| Multi-label SWC (8 classes) | LogisticRegression | macro-F1 (test) | 0.4538 |
| Multi-label SWC (8 classes) | best Conv-Transformer (`C2_dmodel_256`) | macro-F1 (test) | 0.6793 |

All values are from a single read of the held-out test split (11,247 contracts) under the v2 protocol. Confidence intervals are stratified percentile bootstrap (B=1000). XGBoost beats the best deep configuration on 8/8 classes (sign test p=0.0078; with n=8 the smallest attainable two-sided p is 2/256=0.0078, so this is the floor, not a stronger claim).

These numbers are generated from `results/` by `scripts/emit_cards_v2.py` and cross-checked against the paper by `scripts/check_numbers_v2.py`; they are not typed by hand.


## Reproduce

A public Kaggle notebook re-derives every headline number from the released
splits and prints PASS/FAIL against the published values — split digests,
no-overlap invariants, all binary metrics with bootstrap CIs, the paired
XGB-vs-RF delta, all 24 per-class multi-label cells, and the paired
statistics of the deep comparison:

- Notebook: https://www.kaggle.com/code/sergeisolovyev/sc-vuln-v2-reproduction
- Local copy: [`notebooks/repro_v2.ipynb`](notebooks/repro_v2.ipynb)
- Generator: [`scripts/build_repro_notebook.py`](scripts/build_repro_notebook.py)
  (published values are injected from `results/` at build time, never typed)

The notebook exits non-zero on any mismatch, so a "Failed" run on Kaggle
means the guard fired — read its PASS/FAIL table, not just the status badge.
A coverage guard also fails the run if the number of executed checks ever
drifts from the expected count, so no published number can silently drop out
of verification. Tolerances are calibrated against measured cross-version
noise (F1 reproduces to ~1e-5; threshold-sensitive components to <2e-3, while
a wrong model specification moves them by >3e-3 and is caught).

## Repository Structure

```text
.
├── CITATION.cff
├── DATASET_CARD.md
├── FEATURE_SCHEMA.md
├── MODEL_CARD.md
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md            # corpus provenance and v2 split manifest
│   ├── feature_columns.json # canonical 67-feature order
│   ├── label_mappings.json  # 39 Slither detectors -> 8 classes (+ignore/safe)
│   └── manifest_v2.json     # exact per-step row counts + split sha256
├── paper/                   # v2 sources; every number arrives via numbers_v2.tex
│   ├── README.md
│   ├── main_v2.tex
│   ├── main_v2_body.tex
│   ├── numbers_v2.tex       # GENERATED by scripts/emit_macros_v2.py
│   ├── references.bib
│   ├── icicpe.sty
│   ├── ICICPEtran.bst
│   └── figures/
│       └── perlabel_f1_heatmap_v2.pdf
├── notebooks/
│   ├── repro_v2.ipynb           # 114 checks; re-derives every published number
│   └── colab_c3c4_finisher.ipynb  # GPU finisher, notebook form
├── results/
│   ├── README.md
│   ├── dl_runs_v2/              # all 10 deep configurations, one JSON each
│   ├── results_classical/       # binary results, thresholds, paired delta
│   ├── stats_v2.json            # paired statistics behind the comparison
│   ├── numbers_v2.tex           # the macros the paper reads
│   ├── manifest_v2.json         # per-step row counts + split sha256
│   ├── feature_columns.json     # canonical 67-feature order
│   └── metrics_v1_retracted.json  # v1 numbers, kept for provenance ONLY
├── scripts/
│   ├── build_data_v2.py         # Phase 1: dedup + 80/10/10 split
│   ├── run_classical_v2.py      # Phase 2: classical models, honest protocol
│   ├── train_c3c4_gpu.py        # Phase 3: deep configs on GPU, resumable
│   ├── dl_trainer_v2_source.py  # Phase 3: Conv-Transformer trainer body
│   ├── build_kernel_v2.py       # Phase 3 helper: assemble Kaggle DL kernel
│   ├── collect_dl_v2.py         # Phase 3b: harvest DL results with provenance
│   ├── analyze_v2.py            # Phase 4: statistics + per-label heatmap
│   ├── emit_macros_v2.py        # Phase 5: artifacts -> LaTeX macros
│   ├── emit_cards_v2.py         # Phase 5b: README/MODEL_CARD from artifacts
│   ├── check_numbers_v2.py      # Phase 5 gate: built PDF vs artifacts
│   ├── build_repro_notebook.py  # generates the reproduction notebook
│   ├── full_ablation_analysis.py  # v1 ablation analysis (provenance)
│   └── paired_class_test.py       # v1 significance tests (provenance)
├── src/
│   ├── evm_extractor.py     # 67-feature EVM bytecode extractor (sklearn API)
│   ├── dl_pipeline.py       # Conv-Transformer pipeline (tokeniser, model, loop)
│   └── metrics.py           # bootstrap CIs, confusion summaries, top-k recall
└── tests/
    └── test_evm_extractor.py
```

## How to Reproduce (v2 pipeline)

The v2 pipeline is script-based and runs in phase order. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for details and data locations.

```bash
pip install -r requirements.txt

# Phase 1 — data: dedup (metadata-stripped bytecode + 67-d feature vector),
# stratified 80/10/10 split with seed 376, manifest with sha256 per split
python scripts/build_data_v2.py

# Phase 2 — classical models: Optuna via 5-fold CV inside train only,
# thresholds on val, single final report on test, B=1000 bootstrap CIs
python scripts/run_classical_v2.py

# Phase 3 — deep-learning comparator: 10-config Conv-Transformer ablation
# (runs on Kaggle GPU; build_kernel_v2.py assembles the kernel from
# src/dl_pipeline.py + scripts/dl_trainer_v2_source.py)
python scripts/build_kernel_v2.py

# Phase 4 — analysis: paired statistics, Holm correction, per-label heatmap
python scripts/analyze_v2.py

# Phase 5 — gate: every number in the paper must match a result artifact
python scripts/check_numbers_v2.py
```

Local tests for the feature extractor:

```bash
pytest
```

## Reproducibility Entry Points

| Artefact | URL |
|---|---|
| Raw dataset | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| End-to-end Kaggle notebook (v1) | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| W&B ablation project | https://wandb.ai/sesesolovev-hse-university/defi-binary-vuln |
| GitHub repository | https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode |

## Limitations

- Labels are produced by Slither and are not human-audit ground truth.
- Reported performance is bounded by Slither's own false-positive and
  false-negative behaviour.
- The v2 split controls exact and metadata-trailer duplicates and
  feature-vector duplicates, but is not temporal, compiler-version
  stratified, or address-family grouped.
- Proxy and clone families may induce residual correlations across splits.
- The released extractor schema must match the feature matrix used for
  training; see [`FEATURE_SCHEMA.md`](FEATURE_SCHEMA.md).

## Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@misc{solovev2026smartcontract,
  author = {Solovev, Sergei},
  title  = {Lightweight Machine Learning for Smart-Contract Vulnerability
            Detection from {EVM} Bytecode: Binary and Multi-Label
            Classification with a Deep-Learning Comparator},
  year   = {2026},
  note   = {Preprint. v2 evaluation complete; v1 metrics retracted}
}
```

## License

- Code in this repository: MIT License.
- Paper text, figures, and documentation: CC-BY 4.0 unless noted otherwise.
- External datasets and model artefacts retain their own licences.
