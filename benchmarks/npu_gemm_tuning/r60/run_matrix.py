#!/usr/bin/env python3
"""Run three locked R60 trials and persist the exact hardware rows."""

import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "results" / "r60-first-shared-input-mmul-20260713.csv"
TRIALS = int(os.environ.get("R60_TRIALS", "3"))
rows = []
stages = [stage.strip() for stage in os.environ.get(
    "R60_STAGES", "FIRST_MMUL,FULL_K_GROUP_STAGE1,THREE_GROUP_SCALED_STAGE2"
).split(",") if stage.strip()]
for stage in stages:
    for trial in range(1, TRIALS + 1):
        environment = os.environ.copy()
        environment["R60_STAGE"] = stage
        process = subprocess.run(
            [sys.executable, "r60_run.py"],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        sys.stdout.write(process.stdout)
        sys.stderr.write(process.stderr)
        if process.returncode:
            raise SystemExit(process.returncode)
        line = next(
            (line for line in process.stdout.splitlines() if line.startswith("R60_JSON ")),
            None,
        )
        if line is None:
            raise SystemExit("R60 run did not emit a result row")
        row = json.loads(line.removeprefix("R60_JSON "))
        row["trial"] = trial
        rows.append(row)

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summaries = []
for stage in stages:
    values = [row["wire_gbs"] for row in rows if row["mode"] == stage]
    summaries.append(
        f"{stage}={statistics.median(values):.6f}({min(values):.6f}-{max(values):.6f})"
    )
print(f"R60_MATRIX rows={len(rows)} {' '.join(summaries)} csv={RESULT}")
