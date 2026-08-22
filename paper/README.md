# Paper Sources

This directory contains the LaTeX sources of:

> S. S. Solovev. *Lightweight Machine Learning for Smart-Contract
> Vulnerability Detection from EVM Bytecode: Binary and Multi-Label
> Classification with a Deep-Learning Comparator.*

## Version note — these are v1 sources

**`main.tex`, `references.bib`, and the figures in `figures/` are the v1
versions and will be replaced by v2.** The v1 evaluation was retracted due
to a leaky train/validation split and tuning on the reporting split (see
the root `README.md`). The v2 rerun is in progress; once it completes, the
manuscript numbers will be regenerated from the v2 result artifacts and
verified with `scripts/check_numbers_v2.py` before the sources here are
updated.

Do not cite any numeric value from the current `main.tex`.

## Contents

| File | Note |
|---|---|
| `main.tex` | v1 manuscript source (numbers retracted, text to be revised for v2) |
| `references.bib` | bibliography |
| `figures/perlabel_f1_heatmap.pdf` / `.png` | v1 per-label F1 heatmap; the v2 version is produced by `scripts/analyze_v2.py` |

## Not included

- `main.tex` also references `feature_categories_donut.pdf` and
  `pareto_classical_vs_dl.pdf`; these v1 figures are not shipped here and
  will be added in their v2 form together with the revised manuscript.
- Venue style files (`icicpe.sty`, `ICICPEtran.bst`) are not redistributed
  in this repository; the document therefore does not compile as-is.
