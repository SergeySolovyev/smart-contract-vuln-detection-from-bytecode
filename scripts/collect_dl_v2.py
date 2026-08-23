# -*- coding: utf-8 -*-
"""Paper v2 — Phase 3 harvester: gather DL configs from every source.

Three independent sources, merged by config name (first valid wins):
  1. runs_v2/            — anything already harvested locally
  2. kaggle kernels output of both trainers (fwd + rev)
  3. the ops-drop inbox on Heroku (populated live by v6 trainers, and the
     only source that survives a 12h-wall cancellation)

Prints exactly which configs are present, which are missing, and where each
came from — provenance matters, since v1's central defect was numbers whose
origin nobody could reconstruct.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import os as _os
import pathlib as _pathlib

# Working directory. Defaults to the results/ tree shipped in this
# repository so the analysis and emit scripts run straight from a checkout;
# set PAPER_V2_DIR to a full working tree (with the parquet splits and
# runs_v2/) to regenerate results from scratch.
_ROOT = _os.environ.get("PAPER_V2_DIR") or str(
    _pathlib.Path(__file__).resolve().parent.parent / "results")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V2 = Path(_ROOT)
RUNS = V2 / "runs_v2"
RUNS.mkdir(exist_ok=True)
KLOG = Path(r"C:\Users\1\AppData\Local\Temp\claude"
            r"\D--DeFi\78df77b8-631e-440a-a518-39fb9eedbda2\scratchpad\klog")
KLOG.mkdir(parents=True, exist_ok=True)

EXPECTED = ["A1_baseline", "B1_pos_weight", "B2_focal_g2",
            "B3_focal_pos_weight", "B4_asymmetric",
            "B5_threshold_tuning_on_best", "C1_transformer_4layers",
            "C2_dmodel_256", "C3_pure_cnn", "C4_pure_transformer"]

provenance = {}

# The trainer sometimes emits a shortened config name (observed: ops-drop
# key "B5_threshold_tuning" for the registry entry
# "B5_threshold_tuning_on_best"). Map prefixes back so a finished config is
# never counted as missing over a naming detail.
def canonical(name: str) -> str:
    for e in EXPECTED:
        if e == name or e.startswith(name) or name.startswith(e):
            return e
    return name


def take(name: str, payload: dict, src: str) -> None:
    name = canonical(name)
    if name in provenance:
        return
    if not isinstance(payload, dict) or "f1_per_label" not in payload:
        return
    (RUNS / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    provenance[name] = src

# ── 1. already local ────────────────────────────────────────────────────────
for p in RUNS.glob("*.json"):
    try:
        take(p.stem, json.loads(p.read_text(encoding="utf-8")), "local")
    except Exception:
        pass

# ── 2. kernel outputs ───────────────────────────────────────────────────────
for variant in ("fwd", "rev"):
    slug = f"sergeisolovyev/sc-vuln-dl-v2-trainer-{variant}"
    dest = KLOG / f"out_{variant}"
    try:
        shutil.rmtree(dest, ignore_errors=True)
        subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(dest)],
                       capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        print(f"  kernel {variant}: {e}")
        continue
    for p in dest.rglob("*.json"):
        if p.name == "dl_summary_v2.json":
            try:
                for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                    take(k, v, f"kernel-{variant}")
            except Exception:
                pass
        else:
            try:
                take(p.stem, json.loads(p.read_text(encoding="utf-8")),
                     f"kernel-{variant}")
            except Exception:
                pass

# ── 3. ops-drop inbox ───────────────────────────────────────────────────────
try:
    sec = [l.split("=", 1)[1].strip()
           for l in io.open(os.environ.get("DL_DROP_ENV", ""),
                            encoding="utf-8")
           if l.startswith("REVERT_OPS_SECRET=")][0]
    req = urllib.request.Request(
        os.environ.get("DL_DROP_URL", ""),
        headers={"x-ops-secret": sec})
    with urllib.request.urlopen(req, timeout=60) as r:
        for k, v in json.loads(r.read()).items():
            if k != "smoke":
                take(k, v, "ops-drop")
except Exception as e:  # noqa: BLE001
    print(f"  ops-drop: {e}")

# ── report ──────────────────────────────────────────────────────────────────
have = sorted(provenance)
missing = [c for c in EXPECTED if c not in provenance]
print(f"\ncollected {len(have)}/10 DL configs")
for c in EXPECTED:
    src = provenance.get(c)
    mark = "OK  " if src else "MISS"
    macro = ""
    if src:
        d = json.loads((RUNS / f"{c}.json").read_text(encoding="utf-8"))
        mf = d.get("macro_f1_external") or d.get("macro_f1")
        macro = f"macro={mf:.4f}" if mf else ""
    print(f"  {mark} {c:30s} {src or '-':12s} {macro}")
if missing:
    print(f"\nmissing: {missing}")
    print("-> rerun Phase 3 for these, or report the ablation as partial")
else:
    print("\nall 10 present — Phase 4 can run")
