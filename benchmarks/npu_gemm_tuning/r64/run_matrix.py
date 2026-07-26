#!/usr/bin/env python3
"""Run fresh-process shim traces for every R64 physical column."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "results" / "r64-full-qkv-shim-trace-20260713.csv"
TRIALS = int(os.environ.get("R64_TRIALS", "1"))
COLUMNS = [int(value) for value in os.environ.get("R64_COLUMNS", "0,1,2,3").split(",")]
if not COLUMNS or any(column < 0 or column > 7 for column in COLUMNS):
    raise SystemExit("R64_COLUMNS must contain columns in 0..7")
rows = []
for column in COLUMNS:
    cache = Path(
        os.environ.get(
            f"R64_CACHE_C{column}",
            f"~/.hipfire/npu/embgemma_r64_trace_shim{column}",
        )
    ).expanduser()
    for trial in range(1, TRIALS + 1):
        environment = os.environ.copy()
        environment.update(
            {
                "R61_ACTIVATION_MODE": "w4-native",
                "R61_OUTPUT_MODE": "physical",
                "R61_CACHE_DIR": str(cache),
                "R64_TRACE_SIZE": "16777216",
                "R64_TRACE_TILE": "shim",
                "R64_TRACE_START": str(column),
                "R64_TRACE_COLS": "1",
                "R64_TRACE_DIR": str(Path.home() / f".hipfire/r64-traces/c{column}"),
            }
        )
        process = subprocess.run(
            [str(Path.home() / ".venv/bin/python"), "../r61/r61_raw_run.py"],
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
            (value for value in process.stdout.splitlines() if value.startswith("R64_JSON ")),
            None,
        )
        if line is None:
            raise SystemExit(f"R64 column {column} did not emit a result")
        row = json.loads(line.removeprefix("R64_JSON "))
        row["trial"] = trial
        rows.append(row)

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

spans = [row["device_span_us"] for row in rows]
print(
    f"R64 rows={len(rows)} device_span_us={min(spans):.3f}-{max(spans):.3f} "
    f"csv={RESULT}"
)
