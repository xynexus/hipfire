#!/usr/bin/env python3
"""Run three fresh-process R63 compact-QKV control trials."""

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
        "R63_RESULT",
        HERE.parent / "results" / "r63-w4-native-compact-qkv-20260713.csv",
    )
).expanduser()
TRIALS = int(os.environ.get("R63_TRIALS", "3"))
rows = []
for trial in range(1, TRIALS + 1):
    environment = os.environ.copy()
    environment["R61_ACTIVATION_MODE"] = "w4-native"
    environment["R61_OUTPUT_MODE"] = "physical"
    environment["R61_KERNEL_VARIANT"] = os.environ.get(
        "R63_KERNEL_VARIANT", "current-no-spill"
    )
    environment["R61_CACHE_DIR"] = str(
        Path(
            environment.get(
                "R63_CACHE_DIR",
                "~/.hipfire/npu/embgemma_r63_w4_native_compact_qkv_m256_k768_n1280",
            )
        ).expanduser()
    )
    process = subprocess.run(
        [sys.executable, "../r61/r61_raw_run.py"],
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
        (line for line in process.stdout.splitlines() if line.startswith("R62_JSON ")),
        None,
    )
    if line is None:
        raise SystemExit("R63 compact-QKV run did not emit a result row")
    row = json.loads(line.removeprefix("R62_JSON "))
    row["mode"] = "FULL_W4_NATIVE_INPUT_OQ4_COMPACT_QKV_PHYSICAL"
    row["trial"] = trial
    rows.append(row)

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

values = [row["npu_ms"] for row in rows]
print(
    f"R63 rows={len(rows)} median_npu_ms={statistics.median(values):.6f} "
    f"range={min(values):.6f}-{max(values):.6f} csv={RESULT}"
)
