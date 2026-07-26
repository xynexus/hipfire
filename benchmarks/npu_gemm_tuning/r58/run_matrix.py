#!/usr/bin/env python3
"""Run identical packed feed-only and nibble-decode trials into one CSV."""

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
        "R58_RESULT_CSV",
        HERE.parent / "results" / "r58-nibble-decode-20260712.csv",
    )
)
TRIALS = int(os.environ.get("R58_TRIALS", "3"))

rows = []
MODES = (
    "PACKED_FEED_ONLY",
    "NIBBLE_DECODE",
    "COMPUTE_STAGE1",
    "COMPUTE_STAGE2",
)
for mode in MODES:
    for trial in range(TRIALS):
        env = os.environ.copy()
        env["R58_MODE"] = mode
        process = subprocess.run(
            [sys.executable, "r58_run.py"],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(process.stdout)
        sys.stderr.write(process.stderr)
        if process.returncode:
            raise SystemExit(f"{mode} trial {trial} failed with {process.returncode}")
        record = next(
            (
                json.loads(line.removeprefix("R58_JSON "))
                for line in process.stdout.splitlines()
                if line.startswith("R58_JSON ")
            ),
            None,
        )
        if record is None:
            raise SystemExit(f"{mode} trial {trial} emitted no R58_JSON row")
        record["trial"] = trial
        rows.append(record)

RESULT.parent.mkdir(parents=True, exist_ok=True)
fields = sorted({key for row in rows for key in row})
with RESULT.open("w", newline="", encoding="utf-8") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

medians = {
    mode: statistics.median(row["wire_gbs"] for row in rows if row["mode"] == mode)
    for mode in MODES
}
retention = medians["NIBBLE_DECODE"] / medians["PACKED_FEED_ONLY"]
compute_retention = medians["COMPUTE_STAGE1"] / medians["NIBBLE_DECODE"]
stage2_retention = medians["COMPUTE_STAGE2"] / medians["COMPUTE_STAGE1"]
print(
    f"R58_SUMMARY feed_median_gbs={medians['PACKED_FEED_ONLY']:.6f} "
    f"decode_median_gbs={medians['NIBBLE_DECODE']:.6f} "
    f"retention={retention:.6f} "
    f"compute_median_gbs={medians['COMPUTE_STAGE1']:.6f} "
    f"compute_retention={compute_retention:.6f} "
    f"stage2_median_gbs={medians['COMPUTE_STAGE2']:.6f} "
    f"stage2_retention={stage2_retention:.6f} csv={RESULT}"
)
