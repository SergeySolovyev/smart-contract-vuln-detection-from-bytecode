# Reproducibility Guide

This repository is a compact research companion. The full end-to-end run is
kept in the Kaggle notebook because it depends on the prepared parquet feature
matrices, cached deep-learning ablation runs, and figure-generation cells.

## Public Artefacts

| Artefact | Purpose | URL |
|---|---|---|
| Raw Slither-labelled corpus | Upstream source contracts, bytecode, reports | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| Kaggle notebook | End-to-end feature-matrix loading, training, evaluation, figures | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| W&B project | Deep-learning ablation tracking and run metadata | https://wandb.ai/sesesolovev-hse-university/defi-binary-vuln |
| This repository | Feature extractor, cards, metrics, schema notes | https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode |

## Recommended Full Rerun

1. Open the Kaggle notebook.
2. Confirm that the notebook has the following attached Kaggle datasets:
   `sergeisolovyev/defi-bytecode-features`,
   `sergeisolovyev/dl-ablation-cache`, and
   `sergeisolovyev/dl-pipeline-utils`.
3. Enable the GPU accelerator for the Conv-Transformer comparator.
4. Enable internet access if W&B logging or package installation is required.
   W&B credentials are optional; the notebook falls back to local outputs.
5. Run all cells.
6. Check the generated `results/reproducibility_summary.json`,
   `results/feature_columns.json`, and CSV result tables in the Kaggle output.
7. Compare the regenerated tables against `results/metrics.json`.

The notebook records:

- global random seed: `42`
- split seed: `376`
- feature schema size: `67`
- validation size: `11,670`
- bootstrap resamples: `1,000`

The deep-learning ablation uses the attached `dl-ablation-cache` dataset. This
is still reproducible: the cache contains the run JSONs produced by the
notebook's Conv-Transformer cells. Without that cache, Kaggle's session limit
requires resuming the ablation across multiple notebook versions.

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
