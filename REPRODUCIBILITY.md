# Reproducibility Guide

This repository is a compact research companion. The executable rerun is kept
in the Kaggle notebook because it depends on prepared parquet feature matrices,
model-result caches, and figure-generation cells.

## Public Artefacts

| Artefact | Purpose | URL |
|---|---|---|
| Raw Slither-labelled corpus | Upstream source contracts, bytecode, reports | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| Kaggle notebook | End-to-end feature-matrix loading, training, evaluation, figures | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| Public feature matrices | Train/validation parquet matrices used by the notebook | https://www.kaggle.com/datasets/sergeisolovyev/defi-bytecode-features-public |
| Public run cache | Cached JSON outputs for reviewer reruns and completed neural records | https://www.kaggle.com/datasets/sergeisolovyev/smart-contract-vuln-run-cache |
| W&B project | Deep-learning ablation tracking and run metadata | https://wandb.ai/sesesolovev-hse-university/defi-binary-vuln |
| This repository | Feature extractor, cards, metrics, schema notes | https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode |

## Recommended Reviewer Rerun

1. Open the Kaggle notebook.
2. Confirm that the notebook has the following public Kaggle datasets attached:
   `sergeisolovyev/defi-bytecode-features-public` and
   `sergeisolovyev/smart-contract-vuln-run-cache`.
3. CPU execution is sufficient for the default public reviewer rerun. The
   notebook starts with `CACHE_BACKED_REVIEWER_RUN=True`, restores cached model
   records, and skips fresh SHAP, feature-importance, severity-waterfall, and
   uncached neural refits.
4. For a heavier rerun, set `CACHE_BACKED_REVIEWER_RUN=False` in the notebook.
   Enable a GPU/T4 runtime before launching fresh uncached Conv-Transformer
   configurations.
5. Enable internet access if W&B logging or package installation is required.
   W&B credentials are optional; the notebook falls back to local outputs.
6. Run all cells.
7. Check the generated `results/reproducibility_summary.json`,
   `results/feature_columns.json`, and CSV result tables in the Kaggle output.
8. Compare the regenerated tables against `results/metrics.json`.

The notebook records:

- global random seed: `42`
- split seed: `376`
- feature schema size: `67`
- validation size: `11,670`
- bootstrap resamples: `1,000`

The public run cache contains JSON outputs from the binary models, classical
multi-label baselines, and completed Conv-Transformer configurations. The
notebook exports `cache_backed_reviewer_run`, `dl_cached_configs`,
`dl_expected_configs`, and `dl_cache_complete` in
`results/reproducibility_summary.json`, so reviewers can distinguish the default
CPU cache-backed rerun from a fresh full neural ablation. Missing
Conv-Transformer configurations should be launched only with GPU/T4 enabled and
then added to a refreshed run-cache dataset.

## Local Inspection

The local repository is useful for inspecting and testing the bytecode feature
extractor.

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell users can activate manually
pip install -r requirements.txt
pytest
```

The tests are intentionally small. They check extractor behaviour on empty,
invalid, and simple bytecode inputs. They do not retrain the models.

## What Is Not Yet Fully Local

For a journal-grade artefact package, the following should eventually be moved
from notebook-only execution into versioned scripts:

- deterministic raw-to-feature extraction script;
- train/validation split manifest with contract IDs or bytecode hashes;
- classical binary training script;
- multi-label training script;
- bootstrap and statistical-test script;
- figure-generation script.

Until then, the Kaggle notebook is the executable source of truth for full
reproduction, while this repository documents the method and ships the
standalone extractor.
