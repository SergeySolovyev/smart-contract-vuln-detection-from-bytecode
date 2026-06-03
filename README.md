# ICICPE 2026 — Lightweight ML for Smart-Contract Vulnerability Detection

Source code, figures, and reproducibility scripts for the ICICPE 2026
submission:

> **Lightweight Machine Learning for Smart-Contract Vulnerability Detection
> from EVM Bytecode: Binary and Multi-Label Classification with a
> Deep-Learning Comparator**
>
> S. S. Solovev (WorldQuant University)

---

## Live mainnet validation (2026-05-27)

> The paper's held-out F1 (ICICPE 2026 submission — under review, not yet
> peer-reviewed) is in the Abstract below. The numbers in this section are
> what happened when we ran the model end-to-end on live Ethereum and
> Arbitrum bytecode.

### Slither head-to-head (gold-standard cross-check)

We ran Slither — the source-required industry-standard scanner —
against the same mainnet contracts where REVERT emitted scores,
joining results by address. **80 %+ agreement** on the contracts where
both could be evaluated:

| Contract | REVERT | Slither (H/M/L) | Agreement |
|---|---|---|---|
| Uniswap UniversalRouter | 1.00 HIGH | 8 / 52 / 24 | ✅ BOTH_FLAG (`delegatecall-loop`) |
| Compound cETH | 1.00 HIGH | 0 / 35 / 14 | ✅ BOTH_FLAG (`erc20-interface`) |
| Compound cUSDC | 1.00 HIGH | 0 / 31 / 15 | ✅ BOTH_FLAG (`incorrect-equality`) |
| Sushiswap Router | 1.00 HIGH | 3 / 6 / 17 | ✅ BOTH_FLAG |
| Aave V2 LendingPool | 0.99 HIGH | 6 / 0 / 5 | ✅ BOTH_FLAG (`controlled-delegatecall`) |
| WETH | 0.003 CLEAN | 0 / 0 / 0 | ✅ BOTH_CLEAN |
| USDT | 0.0004 CLEAN | 0 / 15 / 2 | ❌ SLITHER_ONLY (Tether non-standard ERC-20) |

The single divergence is the known Tether-specific `transfer()`
return-value design — invisible to bytecode-only features by design.
Full numbers and methodology in
[`bulk_scan_pilot/comparison_report.csv`](bulk_scan_pilot/) and
[`findings_slither_comparison.md`](bulk_scan_pilot/).

### Pre-hack retrocast (3 weeks before each exploit)

| Hack | Date | Loss | REVERT score | Verdict |
|---|---|---|---|---|
| Nomad Bridge | 2022-08-01 | $190 M | **0.989** | ✅ Caught |
| Curve pETH | 2023-07-30 | $11 M | **0.982** | ✅ Caught |
| Euler Finance | 2023-03-13 | $200 M | **0.004** | ❌ Missed (Module/Dispatcher proxy — fixing in v1.1) |

**2-of-3**. Honest reporting. The miss is interpretable: Euler's
vulnerable function lived in a separate impl contract behind their
Module/Dispatcher proxy; the proxy bytecode is benign. v1.1 ships
EIP-1967 + Diamond-storage impl chasing.

60-second Loom walking through this:
[`retrocast_loom/loom_v2_honest.md`](retrocast_loom/) (recording shortly).

### Multi-chain

Validated on Arbitrum (top-20 DeFi contracts via public RPC; no
Alchemy key needed for L2s). See
[`bulk_scan_pilot/arb_top_report_v1_1.csv`](bulk_scan_pilot/).
Optimism / Base / BNB endpoints already wired.

---

## What this means in one sentence

> **REVERT extends Slither-level vulnerability signal to the ~95 % of
> mainnet contracts where Slither cannot run (no verified source).**

If you run an audit firm and want to be a v1.2 design partner,
[email Sergei](mailto:sssolovjov@gmail.com) — terms in
[`customer_discovery/dm_v2_slither_proof.md`](customer_discovery/).

---

## Abstract

Smart-contract vulnerabilities have caused losses exceeding USD 3 billion
in decentralised finance, while most Ethereum mainnet contracts ship
without verified source and are therefore unreachable by source-required
analysers (Slither, Mythril). We present a lightweight 65-feature pipeline
operating directly on EVM bytecode and study both **binary** any-vulnerability
detection and **multi-label** SWC-class classification on 117,091
Slither-labelled contracts.

- **Binary** — three classical ensembles converge to F1 in [0.918, 0.948];
  RandomForest leads at the recall-priority operating point
  (F1 = 0.947, FNR = 3.8%) and is statistically indistinguishable from
  XGBoost+Optuna under B = 1000 bootstrap.
- **Multi-label** — classical XGBoost (0.751) outperforms all ten
  configurations of a 14-run Conv-Transformer ablation (best 0.730) at
  roughly 30× less training compute.

We attribute the negative deep-learning result to the **tabular-data regime**,
where tree-based models systematically outperform deep nets
(Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022).

## Reproducibility — four artefacts

| Artefact | URL |
|---|---|
| Raw dataset (HuggingFace) | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts |
| End-to-end Kaggle notebook | https://www.kaggle.com/code/sergeisolovyev/icicpe-2026-defi-vuln-detection |
| W&B 14-run DL ablation | project `defi-binary-vuln`, run `hk57ndy1`, entity `sesesolovev-hse-university` |
| This repository (paper source) | https://github.com/SergeySolovyev/icicpe-2026-defi-vuln-detection |

The Kaggle notebook is the executable reproducibility entry point — feature
extraction, training, all four classical binary classifiers, the
ten-configuration Conv-Transformer ablation, and figure generation all run
end-to-end there. This repository contains the LaTeX paper source and the
local post-processing scripts.

## Repository structure

```
.
├── README.md            # this file — method, results, reproducibility pointers
├── LICENSE              # MIT
├── features/
│   └── evm_extractor.py # the 70-feature bytecode feature extractor — the method
├── MODEL_CARD.md        # model card (HuggingFace)
└── DATASET_CARD.md      # dataset card (HuggingFace)
```

This repo publishes the **method** (how the 70 features are computed straight
from runtime bytecode) and the cards. The trained model weights are not
redistributed — the executable reproduction below regenerates them.

## Full paper & reproducibility

| What | Where |
|---|---|
| Paper (preprint, open) | figshare — see Contact / paper links |
| Executable reproduction (features → training → figures, end-to-end) | the Kaggle notebook above |
| Raw dataset | HuggingFace (`DATASET_CARD.md`) |
| DL ablation logs | W&B `defi-binary-vuln` |

The Kaggle notebook is the single executable entry point: it runs feature
extraction, all four classical binary classifiers, the Conv-Transformer
ablation, the paired statistical tests (sign-test + Wilcoxon on class-level
F1), and figure generation — reproducing every number in the paper.

## Citation

```bibtex
@inproceedings{solovev2026icicpe,
  author    = {S. S. Solovev},
  title     = {Lightweight Machine Learning for Smart-Contract Vulnerability
               Detection from {EVM} Bytecode: Binary and Multi-Label
               Classification with a Deep-Learning Comparator},
  booktitle = {Proc. 10th Int. Conf. on Interdisciplinary Research on
               Computer Science, Psychology, and Education (ICICPE 2026)},
  year      = {2026},
  address   = {Chiang Mai, Thailand}
}
```

## License

- **Code and scripts** — MIT License (see `LICENSE`).
- **Paper text and figures** — CC-BY 4.0.
- **Trained model weights** — released separately on HuggingFace under
  Apache 2.0.

## Contact

S. S. Solovev — sssolovjov@gmail.com
