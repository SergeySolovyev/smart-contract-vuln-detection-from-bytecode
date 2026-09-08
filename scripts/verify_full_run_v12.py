#!/usr/bin/env python3
"""Verify that the released version-12 run confirms the paper's Section 7.

Run it from the repository root:

    python scripts/verify_full_run_v12.py

It uses only the standard library and reads only files in this repository.
It trains nothing, loads no model and opens no dataset row.

Three questions are answered, in order:

1. Are the released artifacts the ones the paper was built from? Every file
   is rehashed against the identities recorded when they were exported.
2. Was the notebook actually executed to completion? Cell counts, execution
   counts and error outputs are read from the notebook itself.
3. Do the run's reported values follow from those artifacts? Each one is
   recomputed from the raw records and compared with its expected string.
   The subset that Section 7 of the paper prints is checked the same way.

Exit code 0 means all three hold. Exit code 1 lists what failed.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "full_run_v12"
NOTEBOOK = ROOT / "notebooks" / "ML_Experiments_v2_full_run.ipynb"

# What the released run must reproduce. Most of these values appear in the
# paper; the notebook and kernel identifiers are properties of the run itself
# and are documented here rather than in the text. The point of the script is
# that nothing below is transcribed from a run log: each value is recomputed
# from the released records and compared against these strings.
EXPECTED = {
    "binary LogReg, test F1": "0.890",
    "binary LogReg, test MCC": "0.610",
    "binary LogReg, test PR-AUC": "0.953",
    "multi-label LogReg, test macro-F1": "0.5017",
    "selected DL configuration": "C2",
    "selected DL, validation macro-F1": "0.6578",
    "selected DL, test macro-F1": "0.6539",
    "weakest DL configuration": "B2",
    "weakest DL, validation macro-F1": "0.6074",
    "DL mean validation macro-F1": "0.6353",
    "XGBoost, validation macro-F1": "0.7518",
    "XGBoost minus selected DL, p.p.": "9.40",
    "binary XGBoost, validation PR-AUC": "0.987",
    "binary XGBoost, validation MCC": "0.801",
    "B5 delta, smallest": "+0.0109",
    "B5 delta, largest": "+0.0258",
    "notebook cells": "105",
    "notebook code cells": "54",
    "model stages trained": "16",
    "DL configurations trained": "9",
    "DL epochs, fewest": "9",
    "DL epochs, most": "17",
    "token vocabulary entries": "1,405",
    "named opcode mnemonics in it": "16",
    "Kaggle kernel version": "12",
    "GPU": "Tesla T4",
}

FROZEN_SET = ["ML0_LogReg", "C2", "G3_LogReg"]
EXECUTED_NOTEBOOK_SHA = (
    "561275b2556cccadfe2705e52c77d87cc4f315afb3c63bde1b216c82833ae39d")
READER_NOTEBOOK_SHA = (
    "5e9af3ce6e7d94a3fa82e0c0a4b5672a1d443d30a2970d617e3d59e1ae7ce647")

checks: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    checks.append((bool(ok), name, detail))


def f64(hexstr: str) -> float:
    """Decode a stored little-endian IEEE-754 double."""
    return struct.unpack("<d", bytes.fromhex(hexstr))[0]


def load(rel: str):
    return json.loads((RUN / rel).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate(model_key: str, metric_name: str) -> float:
    """The corpus-level value of one metric for one model on the test split."""
    record = load(f"final/{model_key}.json")
    for row in record["metric_results"]:
        if row["label"] is None and row["metric_name"] == metric_name:
            if row["status"] != "OK":
                raise SystemExit(
                    f"{model_key}/{metric_name} has status {row['status']}")
            return f64(row["value_f64_hex"])
    raise SystemExit(f"{model_key}: no corpus-level {metric_name}")


# --------------------------------------------------------------- 1. files
meta = load("README.json")
missing = [e["file"] for e in meta["files"] if not (RUN / e["file"]).exists()]
check(not missing, "every recorded artifact is present", ", ".join(missing))
drift = [e["file"] for e in meta["files"]
         if (RUN / e["file"]).exists() and sha256(RUN / e["file"]) != e["sha256"]]
check(not drift, f"all {len(meta['files'])} artifacts match their recorded SHA-256",
      ", ".join(drift))

# ----------------------------------------------------------- 2. execution
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = nb["cells"]
reader_cells = [c for c in cells
                if c.get("metadata", {}).get("reader_copy_only")]
original = [c for c in cells
            if not c.get("metadata", {}).get("reader_copy_only")]
code = [c for c in original if c["cell_type"] == "code"]
counts = [c.get("execution_count") for c in code]
errors = sum(1 for c in code for o in c.get("outputs", [])
             if o.get("output_type") == "error")

check(len(original) == 105, "105 original cells preserved", f"found {len(original)}")
check(len(reader_cells) == 4, "4 added cells, each marked reader_copy_only",
      f"found {len(reader_cells)}")
check(len(code) == 54, "54 code cells", f"found {len(code)}")
check(counts == list(range(1, 55)),
      "execution counts are exactly 1..54, in order, none empty",
      f"got {counts}")
check(errors == 0, "no error output in any cell", f"found {errors}")
check(meta["executed_notebook"]["sha256"] == EXECUTED_NOTEBOOK_SHA,
      "the source executed notebook is the one recorded in the archive")
check(sha256(NOTEBOOK) == READER_NOTEBOOK_SHA,
      "the notebook in this repository is byte-for-byte the released copy",
      "a mismatch here usually means a line-ending conversion on checkout; "
      ".gitattributes marks this file so that cannot happen")

# The base template carries an inherited authorial_revision block that
# describes the UNEXECUTED candidate it was generated from. It is not a
# statement about this run, and it is left untouched so the file keeps the
# SHA-256 the paper and the archive journal pin. The execution evidence is
# the counts above and the controller state below.
inherited = nb.get("metadata", {}).get("authorial_revision", {})
check(inherited.get("execution_status") == "UNEXECUTED_CANDIDATE",
      "inherited template metadata is present and untouched (see README)")

# ------------------------------------------------------- 3. control state
state = load("run-state.json")
access = load("heldout-access.json")
frozen = load("selection-frozen.json")
b5 = load("b5-threshold-sensitivity.json")

check(state["state"] == "HELDOUT_EVALUATED", "run state is HELDOUT_EVALUATED",
      state["state"])
check(state["execution_status"] == "COMPLETE", "execution status is COMPLETE",
      state["execution_status"])
check(state["official_test_access_count"] == 1,
      "the test split was read exactly once",
      str(state["official_test_access_count"]))
check(state["logical_access_count"] == 1, "one logical heldout access",
      str(state["logical_access_count"]))
check(state["selection_fingerprint"] == access["selection_fingerprint"],
      "the heldout access carries the same selection fingerprint as the state")
check(frozen["final_evaluation_set"] == FROZEN_SET,
      "the frozen final set is ML0_LogReg, C2, G3_LogReg",
      str(frozen["final_evaluation_set"]))
check(b5["diagnostic_status"] == "DIAGNOSTIC_ONLY"
      and b5["execution_stage"] == "POST_SELECTION_FROZEN_PRE_HELDOUT",
      "B5 ran after the freeze and before the test, outside selection")

# ---------------------------------------------------------- 4. the numbers
ranking = load("full-ranking.json")
checkpoint = load("checkpoint.json")
probe = load("input-representation-probe.json")
preflight = load("resource-preflight.json")

dl = {e["model_key"]: f64(e["macro_f1_f64_hex"]) for e in ranking["multilabel_dl"]}
classical = {e["model_key"]: f64(e["macro_f1_f64_hex"])
             for e in ranking["multilabel_classical"]}
binary = {e["model_key"]: (f64(e["pr_auc_f64_hex"]), f64(e["mcc_f64_hex"]))
          for e in ranking["binary_classical"]}
best, worst = max(dl, key=dl.get), min(dl, key=dl.get)
epochs = {k: v["epochs"] for k, v in checkpoint["completed"].items()
          if k.startswith("TRAIN_")}
deltas = [f64(v) for v in b5["sensitivity_delta_f64_hex"].values()]
tokenizer = probe["saved_tokenizer"]

check(best == ranking["selected_dl"],
      "the validation winner is the model the run recorded as selected")

recomputed = {
    "binary LogReg, test F1": f"{aggregate('G3_LogReg', 'f1'):.3f}",
    "binary LogReg, test MCC": f"{aggregate('G3_LogReg', 'mcc'):.3f}",
    "binary LogReg, test PR-AUC": f"{aggregate('G3_LogReg', 'pr_auc'):.3f}",
    "multi-label LogReg, test macro-F1": f"{aggregate('ML0_LogReg', 'macro_f1'):.4f}",
    "selected DL configuration": best,
    "selected DL, validation macro-F1": f"{dl[best]:.4f}",
    "selected DL, test macro-F1": f"{aggregate(best, 'macro_f1'):.4f}",
    "weakest DL configuration": worst,
    "weakest DL, validation macro-F1": f"{dl[worst]:.4f}",
    "DL mean validation macro-F1": f"{sum(dl.values()) / len(dl):.4f}",
    "XGBoost, validation macro-F1": f"{classical['ML0_XGBoost']:.4f}",
    "XGBoost minus selected DL, p.p.":
        f"{(classical['ML0_XGBoost'] - dl[best]) * 100:.2f}",
    "binary XGBoost, validation PR-AUC": f"{binary['G3_XGBoost'][0]:.3f}",
    "binary XGBoost, validation MCC": f"{binary['G3_XGBoost'][1]:.3f}",
    "B5 delta, smallest": f"{min(deltas):+.4f}",
    "B5 delta, largest": f"{max(deltas):+.4f}",
    "notebook cells": str(len(original)),
    "notebook code cells": str(len(code)),
    "model stages trained": str(len(checkpoint["completed"])),
    "DL configurations trained": str(len(dl)),
    "DL epochs, fewest": str(min(epochs.values())),
    "DL epochs, most": str(max(epochs.values())),
    "token vocabulary entries": f"{tokenizer['vocabulary_entries']:,}",
    "named opcode mnemonics in it": str(tokenizer["named_token_count"]),
    "Kaggle kernel version": meta["what"].split("version ")[1].split(",")[0],
    "GPU": preflight["device"]["device_name"],
}

check(tokenizer["named_tokens_equal_ascii_hex_opcode_names"],
      "the named tokens are the opcodes of the hexadecimal ASCII digits")
check(probe["synthetic_comparison"]["disassemble_decoded_bytes"]
      == "PUSH1 0x1\nPUSH1 0x0",
      "decoding the synthetic input first yields two PUSH1 instructions")
check(len(probe["synthetic_comparison"]["legacy_disassemble_string"]
          .split("\n")) == 8,
      "the legacy path yields eight instructions for the same input")

# ------------------------------------------------------------- the report
print(f"Verifying the released version-12 run against the paper.\n"
      f"  artifacts: {RUN.relative_to(ROOT).as_posix()}\n"
      f"  notebook : {NOTEBOOK.relative_to(ROOT).as_posix()}\n")

width = max(len(k) for k in EXPECTED)
print(f"  {'quantity'.ljust(width)}   {'expected':>10}   {'recomputed':>10}")
print(f"  {'-' * width}   {'-' * 10}   {'-' * 10}")
for key, printed in EXPECTED.items():
    got = recomputed[key]
    ok = got == printed
    check(ok, f"expected value: {key}", f"expected {printed}, artifacts {got}")
    mark = " " if ok else " <- MISMATCH"
    print(f"  {key.ljust(width)}   {printed:>10}   {got:>10}{mark}")

failed = [(n, d) for ok, n, d in checks if not ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed.")
if failed:
    print("\nFAILED:")
    for name, detail in failed:
        print(f"  {name}" + (f"  [{detail}]" if detail else ""))
    sys.exit(1)
print("Every reported value follows from the released artifacts, and the")
print("notebook that produced them ran to completion without errors.")
