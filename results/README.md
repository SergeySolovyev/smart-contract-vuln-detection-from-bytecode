# Results

## Current status

**No valid results are published yet.** The v2 rerun is in progress
(as of 2026-08-22). When it completes, this directory will receive the v2
result artifacts produced by `scripts/run_classical_v2.py` and
`scripts/analyze_v2.py`, and the tables in `README.md`, `MODEL_CARD.md`,
and the paper will be filled from them (gated by
`scripts/check_numbers_v2.py`).

## `metrics_v1_retracted.json`

The v1 metrics file is kept **for provenance only** under an explicit
retracted name. Its numbers were produced under a defective protocol
(leaky split: 4.76% of validation rows feature-identical to train;
hyperparameters and thresholds tuned on the reporting split) and must not
be cited or compared against.
