# The decoded-opcode comparator run

One notebook, executed end to end on a Kaggle Tesla T4 on 2026-09-12, answering the two
questions the paper could not: how long the winning classical model takes to fit, and
whether the deep comparator's deficit survives a correctly decoded input. A third answer
came free once the bytecode was back: whether the released feature columns really describe
these contracts. Scores are on the **validation** split; the test split is never opened,
and the notebook asserts it.

| | |
|---|---|
| Executed run, with every output | https://www.kaggle.com/code/sergeisolovyev/decoded-opcodes-c2-and-xgb-timing |
| Notebook | [`05_notebooks/decoded_opcodes_c2.ipynb`](../../05_notebooks/decoded_opcodes_c2.ipynb) |
| Generator | [`02_code/30_build_decoded_notebook.py`](../../02_code/30_build_decoded_notebook.py) |
| Upstream corpus | `mwritescode/slither-audited-smart-contracts`, commit `13594107c7afa216cb0c126f38b8ff6548112dcf` |
| Rows re-attached | 101,220 of 101,220 (100.00%) |
| Environment | Tesla T4, 4 CPU cores, torch 2.10.0+cu128, xgboost 3.2.0, python 3.12.13 |
| Wall clock | 8 h 2 min |

Kaggle renders the executed notebook with all of its outputs at the link above, but serves
only the source through its API, so the copy kept here is the source — byte-identical to
what the generator produces and to what Kaggle ran — and the execution record is
`run_stdout.txt`, the run's own output, exactly as it was printed.

## The numbers

| | validation macro-F1 |
|---|---|
| C2 on decoded opcodes | **0.6841** |
| C2 on legacy tokens — the control: same rows, seed and carve-out | **0.6567** |
| Multi-label XGBoost | **0.7653** |

Decoding the input is worth 2.74 percentage points. Gradient boosting is still 8.12 ahead
of it. The classical fit took **56 s on 4 CPU cores** and predicted 11,247 contracts in
0.48 s; the decoded deep run took **261 min of GPU time** over 20 epochs, the legacy
control 199 min over 15.

Two things qualify the comparison, and both are measured here rather than argued.

*The arms differ in length as well as in meaning.* At the 20,000-token cap the model
reads, the decoded stream fits whole — nothing truncated — while the legacy text is cut
for 13.25% of training contracts (median 8,803 tokens against 3,659). Part of what
decoding buys is simply that the contract now fits.

*One point of macro-F1 is not much.* The decoded arm was run three times in all, on three
machines, with the same code, data and seed: 0.6841 here, 0.6827 on a Colab A100, 0.6691
on a Colab T4 — a spread of 1.51 points. The 2.74 the decoding buys sits just above that;
the 8.12 separating it from gradient boosting sits far outside it.

## What each file holds

- `summary.json` — everything the paper cites, in one place: the recovery statistics, the
  feature check, the classical timing, both deep runs, and the reference points that were
  *not* recomputed here.
- `c2_decoded.json`, `c2_legacy.json` — the full records of the two deep runs as
  `dl_pipeline.run_dl_experiment` writes them: configuration, per-epoch history, per-label
  scores, wall clock.
- `xgb_timing.json` — the multi-label XGBoost fit: hyperparameters, seconds, cores, score.
- `feature_provenance.json` — 66 of the 67 released feature columns are an exact affine
  image of the extractor's output recomputed from the re-attached bytecode, on 1,200
  sampled contracts; the remaining column is constant across that sample, and none is
  contradicted.
- `sequence_lengths.json` — the median encoded length and truncation count of each arm,
  extracted from the run's log.
- `repetition.json` — every execution of the decoded arm, with the spread between them.
- `run_stdout.txt` — what the run printed, start to finish.
- `kaggle_run.log` — the same, in Kaggle's own timestamped form.

## What this is not

A validation-level comparison of one deep configuration under one seed. The test split was
read once, long before, and is not reopened here: nothing in this directory is a test-split
result, and nothing here re-ranks the paper's table.
