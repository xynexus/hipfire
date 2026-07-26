#!/usr/bin/env python3
"""Run each R57 column count in a fresh process and write stable CSV rows."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
MODE = os.environ.get("R57_MODE", "PRODUCTION_DMA").upper()
OUTPUT = Path(
    os.environ.get(
        "R57_OUTPUT",
        HERE.parent / "results" / f"r57-{MODE.lower().replace('_', '-')}-20260712.csv",
    )
)
COLUMNS = tuple(int(value) for value in os.environ.get("R57_COLUMNS", "1,2,4,8").split(","))
REPEAT = int(os.environ.get("R57_REPEAT", "3"))
TRACE_DIR = Path(os.environ.get("R57_TRACE_DIR", "~/.hipfire/r57-traces")).expanduser()
TRACE_DIR.mkdir(parents=True, exist_ok=True)


rows = []
for columns in COLUMNS:
    for repetition in range(REPEAT):
        env = os.environ.copy()
        env["COLS"] = str(columns)
        env["TRACE_TXT"] = str(TRACE_DIR / f"trace-r57-c{columns}-r{repetition}.txt")
        env["TRACE_JSON"] = str(TRACE_DIR / f"trace-r57-c{columns}-r{repetition}.json")
        process = subprocess.run(
            [sys.executable, "production_dma_run.py"],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        line = next(
            (line for line in process.stdout.splitlines() if line.startswith("R57_JSON ")),
            None,
        )
        if process.returncode != 0 or line is None:
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            raise SystemExit(
                f"R57 columns={columns} repetition={repetition} failed rc={process.returncode}"
            )
        row = json.loads(line.removeprefix("R57_JSON "))
        if row["mode"] != MODE:
            raise SystemExit(f"requested mode {MODE}, runner returned {row['mode']}")
        row["repetition"] = repetition
        rows.append(row)
        print(
            f"columns={columns} repetition={repetition} "
            f"wire={row['wire_gbs']:.3f} GB/s payload={row['semantic_unique_gbs']:.3f} GB/s "
            f"logical={row['logical_semantic_gbs']:.3f} GB/s "
            f"busy={row['mean_receive_busy']:.3f}"
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0])
with OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {OUTPUT}")
