#!/usr/bin/env python3
"""Measure production-wrapper parity/timing for the controlled R63 variants."""

import csv
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT = HERE.parent / "results" / "r63-production-wrapper-ab-20260713.csv"
TRIALS = int(os.environ.get("R63_WRAPPER_TRIALS", "3"))
ITERS = int(os.environ.get("R63_WRAPPER_ITERS", "3"))
MODEL = Path(
    os.environ.get(
        "R63_MODEL",
        "~/.hipfire/models/embeddinggemma-300m/EmbeddingGemma-300M.npu.oq4.hfq",
    )
).expanduser()
HFP = Path(
    os.environ.get(
        "R63_QKV_HFP",
        "~/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp",
    )
).expanduser()
VERIFY = ROOT / "target/release/examples/npu_opus_hfp_verify"
ROLES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
]
VARIANTS = [
    (
        "current-no-spill",
        Path("~/.hipfire/npu/embgemma_r63_w4_native_compact_qkv_m256_k768_n1280").expanduser(),
    ),
    (
        "pre-3db7a1497-spill",
        Path("~/.hipfire/npu/embgemma_r63_oldspill_compact_qkv_m256_k768_n1280").expanduser(),
    ),
    (
        "historical-cache",
        Path("~/.hipfire/npu/embgemma_aie2p_whole8_w4-scaled_m256_kg3_n1280").expanduser(),
    ),
]
PARITY = re.compile(r"mismatches=(\d+) max_abs=([0-9.eE+-]+)")
TIMING = re.compile(r"wrapper_ms=([0-9.]+) logical_tops=([0-9.]+)")

rows = []
for variant, cache in VARIANTS:
    for trial in range(1, TRIALS + 1):
        process = subprocess.run(
            [
                str(VERIFY),
                str(MODEL),
                str(cache),
                str(HFP),
                *ROLES,
                "--iters",
                str(ITERS),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        sys.stdout.write(process.stdout)
        sys.stderr.write(process.stderr)
        if process.returncode:
            raise SystemExit(process.returncode)
        parity = PARITY.search(process.stdout)
        timing = TIMING.search(process.stdout)
        if not parity or not timing or int(parity.group(1)) != 0:
            raise SystemExit(f"invalid wrapper result for {variant} trial {trial}")
        rows.append(
            {
                "variant": variant,
                "trial": trial,
                "iterations": ITERS,
                "mismatches": int(parity.group(1)),
                "max_abs": float(parity.group(2)),
                "wrapper_ms": float(timing.group(1)),
                "logical_tops": float(timing.group(2)),
                "cache": str(cache),
                "hfp": str(HFP),
            }
        )

RESULT.parent.mkdir(parents=True, exist_ok=True)
with RESULT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for variant, _ in VARIANTS:
    values = [row["wrapper_ms"] for row in rows if row["variant"] == variant]
    values.sort()
    print(
        f"R63_WRAPPER variant={variant} median_ms={values[len(values) // 2]:.4f} "
        f"range={values[0]:.4f}-{values[-1]:.4f}"
    )
print(f"R63_WRAPPER csv={RESULT}")
