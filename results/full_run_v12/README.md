# The executed version-12 run

This directory holds the reference artifacts of one complete end-to-end
execution of the author's experiment notebook, and the notebook itself is at
[`notebooks/ML_Experiments_v2_full_run.ipynb`](../../notebooks/ML_Experiments_v2_full_run.ipynb),
which also opens at [nbviewer](https://nbviewer.org/github/SergeySolovyev/smart-contract-vuln-detection-from-bytecode/blob/main/notebooks/ML_Experiments_v2_full_run.ipynb) if GitHub declines to render it.
Together they are the evidence behind Section 7 of the paper.

To check that claim rather than take it, run from the repository root:

```bash
python scripts/verify_full_run_v12.py
```

It rehashes every file here, reads the notebook's execution counts, and
recomputes each number Section 7 prints. It needs only the standard library,
and it trains nothing, loads no model and opens no dataset row.

## What was run, and where the numbers come from

The script pipeline in `scripts/` produced the results in the rest of
`results/`, but its fitted models and fit-time environment were not saved.
The run recorded here closes that gap from the other side. The author's
original notebook was restructured into a single document with two modes, a
local teaching mode and a full mode, and the full mode was run to completion
on a Kaggle Tesla T4.

| | |
|---|---|
| Kaggle kernel | `sergeisolovyev/smart-contract-two-mode-full-run`, id 133095850, version 12 |
| Completed | 2026-09-06, provider status COMPLETE with no failure message |
| Notebook | 105 cells, 54 code cells, execution counts exactly 1 to 54, zero error outputs |
| Trained | 16 model stages: 7 classical and 9 deep |
| Read the test split | once |

The chain from the run to the paper has no hand-carried step. The controller
wrote these JSON records during the run; `scripts/emit_macros_v2.py` decodes
them into LaTeX macros; the paper's prose contains only macros, never digits;
and `scripts/check_numbers_v2.py` reads the *rendered* PDF and fails if any
value disagrees with its artifact. Floating-point values are stored as
little-endian IEEE-754 hex, so nothing is lost to rounding on the way in.

The Kaggle kernel itself is private, and its page shows only the small
launcher that starts the reference notebook inside the session. Everything
that matters from that run is here, so nothing in this repository depends on
that page being reachable.

## The files

`README.json` records the exact byte size and SHA-256 of every file in this
directory, and the identity of the executed notebook. The verifier checks all
of them.

**What the run decided.** `full-ranking.json` scores every candidate on the
validation split. `selection-frozen.json` freezes the final set before any
test access, and it is fixed by the method rather than by the ranking: the
multi-label logistic reference, the validation-selected deep configuration,
and the binary logistic reference. `thresholds.json` and
`validation-metrics.json` hold the selection detail.

**What the run measured.** `final/ML0_LogReg.json`, `final/C2.json` and
`final/G3_LogReg.json` carry the test results of those three models, with
per-label supports and an explicit status on every metric.
`final-metrics.json` indexes them and pins each record's hash.

**That the protocol held.** `run-state.json` records state
`HELDOUT_EVALUATED`, execution status `COMPLETE`, and one official test
access. `heldout-access.json` carries the same selection fingerprint as the
state, which is what ties the evaluation to the frozen selection rather than
to some later one. `b5-threshold-sensitivity.json` is marked
`DIAGNOSTIC_ONLY` and `POST_SELECTION_FROZEN_PRE_HELDOUT`: it ran after the
freeze and before the test, and no part of it entered selection.

**Provenance.** `checkpoint.json` lists the completed stages with their epoch
counts. `controller-attempt-ledger.json` and `receipt.json` bind the attempt.
`resource-preflight.json` records the GPU. `artifact-manifest.json` and
`authority-contract.json` record the code and data identities the run
authenticated before it started.

**The representation audit.** `input-representation-probe.json` is the
finding that bounds the deep-learning result. The historical converter passed
the hexadecimal *string* to the disassembler without decoding it to bytes, so
each hexadecimal character was read as an opcode. The fitted vocabulary has
1,405 entries of which exactly 16 are named mnemonics, and those 16 are the
opcodes whose values equal the ASCII codes of the sixteen hexadecimal digits.
On the synthetic input `60016000` the historical path yields eight
instructions where decoding first yields two `PUSH1`s. The 67 numeric
features are unaffected, because their extractor decodes first. Every
deep-learning number in the paper is therefore a result for that legacy
representation, and the paper says so.

## Two things this run does not establish

It reads the same test split the earlier sections read. Two reads of one
partition by two pipelines are still one partition, so a change motivated by
either result cannot be confirmed on it.

The models retain last-epoch weights rather than a restored best epoch, the
training chain was resumed across quota-bounded sessions, and both runs used
a single seed. The paper reports the gap between the two executions without
apportioning it among those causes.

## One inherited field, deliberately left alone

The notebook's own metadata contains an `authorial_revision` block with
`execution_status: UNEXECUTED_CANDIDATE`. That block describes the base
template the notebook was generated from, not this run. It is left untouched
because the file's SHA-256 is pinned in the paper's archive and its audit
journal, and rewriting metadata to look better is exactly the kind of edit
that record exists to prevent. The execution evidence is the counts 1 to 54
in the cells themselves and the controller state above; the verifier checks
both and reports this field explicitly.
