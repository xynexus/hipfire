#!/usr/bin/env python3
"""Repeat the resident HFP ABI feed gate and write durable rows."""

import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RESULT = Path(
    os.environ.get(
        "R59_RESULT_CSV",
        HERE.parent / "results" / "r59-resident-weight-abi-20260713.csv",
    )
)
TRIALS = int(os.environ.get("R59_TRIALS", "3"))

rows = []
MODES = (
    "R34_SEPARATE_BOS",
    "R35_SEPARATE_BOS",
    "R34_BUNDLED_BO",
    "R35_BUNDLED_BO",
)
for mode in MODES:
    for trial in range(TRIALS):
        env = os.environ.copy()
        env["R59_ABI_MODE"] = mode
        process = subprocess.run(
            [sys.executable, "r59_resident_weight_abi.py"],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(process.stdout)
        sys.stderr.write(process.stderr)
        if process.returncode:
            raise SystemExit(f"R59 {mode} trial {trial} failed with {process.returncode}")
        record = next(
            (
                json.loads(line.removeprefix("R59_JSON "))
                for line in process.stdout.splitlines()
                if line.startswith("R59_JSON ")
            ),
            None,
        )
        if record is None:
            raise SystemExit(f"R59 {mode} trial {trial} emitted no R59_JSON row")
        record["trial"] = trial
        record["roles"] = json.dumps(record["roles"], sort_keys=True)
        rows.append(record)

RESULT.parent.mkdir(parents=True, exist_ok=True)
fields = sorted({key for row in rows for key in row})
with RESULT.open("w", newline="", encoding="utf-8") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

for mode in MODES:
    selected = [row for row in rows if row["mode"].endswith(mode)]
    wire = [row["wire_gbs"] for row in selected]
    packed = [row["packed_data_gbs"] for row in selected]
    print(
        f"R59_SUMMARY mode={mode} trials={TRIALS} "
        f"wire_median_gbs={statistics.median(wire):.6f} "
        f"wire_min_gbs={min(wire):.6f} wire_max_gbs={max(wire):.6f} "
        f"packed_median_gbs={statistics.median(packed):.6f} csv={RESULT}"
    )
