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
  - name: defi-vuln-rf-65feat
    results:
      - task:
          type: binary-classification
          name: Smart-contract any-vulnerability detection
        dataset:
          name: slither-audited-smart-contracts (derived 65-feature matrix)
          type: tabular
          config: bytecode-65feat
          split: validation
        metrics:
          - type: f1
            value: 0.947
            name: F1
          - type: mcc
            value: 0.82
            name: MCC
          - type: false_negative_rate
            value: 0.052
            name: FNR
---

# defi-vuln-rf-65feat

Lightweight machine-learning detector for smart-contract vulnerabilities
from raw EVM bytecode — **no source code required**. Pipelines a
65-feature bytecode-disassembly extractor into a Random Forest (binary)
or XGBoost (multi-label) classifier.

This is the model release accompanying the paper:

> **S. S. Solovev (2026)**. *Lightweight Machine Learning for
> Smart-Contract Vulnerability Detection from EVM Bytecode: Binary and
> Multi-Label Classification with a Deep-Learning Comparator.*
> (Chiang Mai, Thailand).
> [Paper repo](https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode)

## Headline numbers

| Task        | Model            | F1    | MCC  | FNR   |
|-------------|------------------|-------|------|-------|
| Binary      | RandomForest     | **0.947** | 0.82 | 5.2%  |
| Binary      | XGBoost (Optuna) | 0.943 | 0.81 | 5.6%  |
| Multi-label | XGBoost (macro)  | **0.751** | —    | —     |

All numbers on a stratified 90/10 split (seed 376) of 117,091
Slither-labelled contracts. Bootstrap CIs reported in the paper.

The classical pipeline **out-performs** a 14-run Conv-Transformer
ablation on multi-label (best DL macro-F1 = 0.730) at roughly 30× less
training compute — a tabular-data-regime result consistent with
Grinsztajn et al. (2022) and Shwartz-Ziv & Armon (2022).

## Files

| File | Purpose |
|---|---|
| `evm_extractor.py` | source of the `EVMBytecodeFeatureExtractor` (scikit-learn `TransformerMixin`) |
| `evm_extractor.pkl` | fitted instance (used at inference) |
| `model_rf_binary.pkl` | trained RandomForest binary classifier |
| `model_xgb_binary.pkl` | trained XGBoost binary classifier |
| `model_xgb_multilabel.pkl` | trained XGBoost multi-label classifier |
| `feature_names.json` | list of the 67 (= 65 base + 2 aggregated) feature names |
| `label_classes.json` | list of the 8 SWC multi-label classes |

## Inference example

```python
import joblib
from evm_extractor import EVMBytecodeFeatureExtractor  # noqa: F401 (for unpickling)
import pandas as pd

extractor = joblib.load("evm_extractor.pkl")
model     = joblib.load("model_rf_binary.pkl")

# Bytecode hex string (0x-prefixed) for one mainnet contract:
bytecode = "0x6080604052..."

X = extractor.transform(pd.DataFrame({"bytecode": [bytecode]}))
score = model.predict_proba(X)[0, 1]
print(f"vulnerability risk score: {score:.3f}")
```

For multi-label per-class probabilities:

```python
model_ml = joblib.load("model_xgb_multilabel.pkl")
import json
classes = json.load(open("label_classes.json"))
proba = model_ml.predict_proba(X)
for c, p in zip(classes, proba[0]):
    print(f"{c:>18s}: {p:.3f}")
```

## Intended use

- Pre-filter for security auditors and DEX risk teams: triage a large
  contract corpus down to a short list for manual review
- Compose with source-required tools (Slither, Mythril) — this model
  fills the gap where source is unavailable (~95% of mainnet contracts)
- Research baseline for new bytecode-only vulnerability methods

## Out of scope

- **Not a final verdict.** Output is "agreement with Slither at scale",
  not human-verified ground truth
- Not a substitute for a comprehensive audit
- Not optimized for confidential / encrypted bytecode (e.g.
  zkApp-wrapped contracts)

## Limitations (verbatim from the paper)

- **Label provenance**: labels come from Slither static analysis;
  inherits Slither's false-positive / false-negative rate
- **Split methodology**: 90/10 stratified by any-vulnerability label,
  seed 376. No address-grouped deduplication for proxy / clone families,
  no compiler-version stratification, no temporal split
- **Dataset coverage**: 2017-2022 deployments only; recent contract
  patterns (account abstraction, EIP-7702 delegations) under-represented

See the paper for full discussion (§Threats to Validity).

## Reproducibility

| Artefact | URL |
|---|---|
| Paper repo (LaTeX source) | https://github.com/SergeySolovyev/smart-contract-vuln-detection-from-bytecode |
| Kaggle notebook (end-to-end) | https://www.kaggle.com/code/sergeisolovyev/smart-contract-vuln-detection-from-bytecode |
| Raw dataset | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| W&B ablation run | project `defi-binary-vuln`, run `hk57ndy1` |

## Citation

```bibtex
@misc{solovev2026smartcontract,
  author = {S. S. Solovev},
  title  = {Lightweight Machine Learning for Smart-Contract Vulnerability
            Detection from {EVM} Bytecode: Binary and Multi-Label
            Classification with a Deep-Learning Comparator},
  year   = {2026},
  note   = {Preprint}
}
```

## License

- **Model weights and inference code**: Apache 2.0
- **Paper text and figures**: CC-BY 4.0 (see GitHub repo)

## Contact

S. S. Solovev — sssolovjov@gmail.com
