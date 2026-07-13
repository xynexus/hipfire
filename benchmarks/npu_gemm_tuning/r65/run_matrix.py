#!/usr/bin/env python3
"""Record fresh-process warmed R65 correctness and timing rows."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RESULT = Path(
    os.environ.get(
        "R65_RESULT",
        HERE.parent / "results" / "r65-w4-bf16-raw-attention-20260713.csv",
    )
).expanduser()
TRIALS = int(os.environ.get("R65_TRIALS", "3"))
rows = []
for trial in range(TRIALS):
    environment = os.environ.copy()
    environment.setdefault("R65_WARMUP", "2")
    environment.setdefault("R65_ITERS", "3")
    completed = subprocess.run(
        [sys.executable, str(HERE / "r65_run.py")],
        cwd=HERE,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = next(
        line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")
    )
    row = json.loads(payload)
    row["trial"] = trial
    rows.append(row)
    print(payload, flush=True)

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(RESULT)
