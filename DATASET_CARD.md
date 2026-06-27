---
license: cc-by-4.0
task_categories:
  - tabular-classification
language:
  - en
tags:
  - smart-contracts
  - ethereum
  - evm
  - vulnerability-detection
  - feature-engineering
  - slither
pretty_name: EVM Bytecode 65-Feature Matrix (117K, Slither-Labelled)
size_categories:
  - 100K<n<1M
source_datasets:
  - mwritescode/slither-audited-smart-contracts
---

# EVM Bytecode 65-Feature Matrix (117 K, Slither-Labelled)

Derived dataset for the paper *"Lightweight Machine Learning
for Smart-Contract Vulnerability Detection from EVM Bytecode"* by
S. S. Solovev. Each row is a smart contract represented by **67
numerical features** (65 base + 2 aggregated risk indicators) extracted
from the raw EVM bytecode via a deterministic pyevmasm-based pipeline.

This dataset is published so that users can **reproduce the paper's
results without re-running feature extraction** — the upstream raw
bytecode corpus
([`mwritescode/slither-audited-smart-contracts`](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts))
is large and feature extraction takes several CPU-hours.

## Splits

| Split | Rows |
|---|---|
| train | 105,027 |
| validation | 11,670 |
| **total** | **117,091** |

Stratified 90/10 split by any-vulnerability label, seed 376. Bytecode-
hash deduplication removed 3,517 duplicates from the raw corpus before
splitting.

## Feature columns (67)

15 SWC-mapped categories + 2 aggregated indicators:

1. **Basic stats** (2): total / unique instructions
2. **Block dependence** (8): TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT,
   COINBASE, BLOCKHASH presence + counts
3. **Environmental opcodes** (4): count, ratio, unique count, complexity
4. **External deps** (6): BALANCE/ADDRESS/CALLER/ORIGIN/CALLVALUE counts
   + dependency index
5. **Calldata ops** (5): SIZE/LOAD/COPY counts, total, density
6. **External calls** (5): count, presence, value/gas ops, reentrancy pattern
7. **Memory ops** (3): reads, writes, ratio
8. **Stack ops** (5): pushes, pops, imbalance, ops ratio, underflow risk
9. **Gas analysis** (5): total cost, average, max, high-gas count, DoS risk index
10. **Arithmetic ops** (3): count, density, unsafe pattern
11. **Control flow** (4): ops, JUMPI count, branching ratio, complexity
12. **Access control** (4): caller checks, origin usage, ratio, origin-instead-caller flag
13. **Advanced patterns** (3): balance-before-call, randomness, bad-randomness flag
14. **Complexity measures** (3): dangerous ops count + density, opcode entropy
15. **Risk scores (base)** (5): reentrancy, frontrunning, DoS, arithmetic, overall
16. **Aggregated risk** (2): vuln_risk_score, complexity_score
17. **Binary flags** (5): reentrancy, unchecked-calls, arithmetic, access-control, DoS indicators

## Labels

8 SWC-aligned multi-label classes (each binary 0/1):

- access-control
- arithmetic
- bad-randomness
- double-spending
- locked-ether
- other
- reentrancy
- unchecked-calls

The `binary` target is `any-of-the-eight`, used for binary classification
in the paper's main result (F1 = 0.947 with RandomForest).

## Load

```python
import pandas as pd

train = pd.read_parquet("train_dl_features.parquet")
val   = pd.read_parquet("val_dl_features.parquet")
print(train.shape, val.shape)  # (105027, 67+labels)  (11670, ...)
```

## Limitations

- **Label provenance**: Slither static-analysis output, NOT
  human-verified ground truth. Inherits Slither's false-positive
  and false-negative rate.
- **No address-grouped deduplication**: proxy/clone families may have
  correlated samples across train/val folds.
- **No compiler-version stratification**: older solc versions
  over-represented; modern patterns (AA wallets, EIP-7702) under-represented.
- **No temporal split**: deployments span 2017-2022; performance on
  2024-2026 contracts is not validated here.

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

CC-BY 4.0 — free for commercial and non-commercial use with attribution.
Note: the upstream raw dataset (`mwritescode/slither-audited-smart-contracts`)
is itself MIT-licensed; this derived feature matrix complies with that.
