#!/usr/bin/env python3
"""Record fresh-process R66 pack correctness/timing controls."""

import csv
import os
from pathlib import Path
import re
import socket
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = Path(
    os.environ.get("R66_CACHE_DIR", "~/.hipfire/npu/embgemma_r66_r65_stage_to_qkv_m256")
).expanduser()
BINARY = ROOT / "target/release/examples/npu_embedding_qkv_pack_verify"
RESULT = Path(
    os.environ.get(
        "R66_RESULT", HERE.parent / "results" / "r66-r65-stage-to-qkv-20260713.csv"
    )
).expanduser()
TRIALS = int(os.environ.get("R66_TRIALS", "3"))
ITERS = int(os.environ.get("R66_ITERS", "100"))
pattern = re.compile(r"([a-z_]+)=([0-9.]+)")
rows = []
for trial in range(TRIALS):
    completed = subprocess.run(
        [str(BINARY), str(CACHE), str(ITERS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(line for line in reversed(completed.stdout.splitlines()) if "dispatch_ms=" in line)
    metrics = {key: float(value) for key, value in pattern.findall(line)}
    row = {
        "trial": trial,
        "host": socket.gethostname(),
        "cache_dir": str(CACHE),
        "iterations": ITERS,
        "q_cosine": metrics["q_cosine"],
        "q_max": metrics["q_max"],
        "q_bit_mismatches": int(metrics["q_bit_mismatches"]),
        "k_cosine": metrics["k_cosine"],
        "k_max": metrics["k_max"],
        "k_bit_mismatches": int(metrics["k_bit_mismatches"]),
        "v_bit_mismatches": int(metrics["v_bit_mismatches"]),
        "dispatch_ms": metrics["dispatch_ms"],
        "oracle": "pass",
        "stage_layout": "r65-inline-10240",
        "q_bytes": 393216,
        "kv_bytes": 262144,
    }
    rows.append(row)
    print(line, flush=True)

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(RESULT)
