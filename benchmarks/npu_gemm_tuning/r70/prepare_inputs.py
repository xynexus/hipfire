#!/usr/bin/env python3
"""Prepare loader-padded activations and valid inline R70 attention records."""

import math
import os
from pathlib import Path
import struct

from ml_dtypes import bfloat16
import numpy as np


SOURCE = Path(
    os.environ.get("R70_SOURCE_INPUT_DIR", "~/.hipfire/npu/r69-chain-input")
).expanduser()
OUT = Path(os.environ.get("R70_INPUT_DIR", "~/.hipfire/npu/r70-chain-input")).expanduser()

ROWS, BLOCKS = 4, 18
SOURCE_A_BLOCK, A_BLOCK = 8192, 10240
PAIR, PAIRS_PER_ROLE, ROLES = 10240, 48, 5
VALUES, CS, PARAMS = 4096, 4096, 2048


def bf16_bits(value):
    return int(np.asarray([value], dtype=bfloat16).view(np.uint16)[0])


source_activations = (SOURCE / "activations.bin").read_bytes()
if len(source_activations) != ROWS * BLOCKS * SOURCE_A_BLOCK:
    raise SystemExit("source activation geometry changed")
padded = bytearray(ROWS * BLOCKS * A_BLOCK)
for block in range(ROWS * BLOCKS):
    source = block * SOURCE_A_BLOCK
    target = block * A_BLOCK
    padded[target : target + SOURCE_A_BLOCK] = source_activations[
        source : source + SOURCE_A_BLOCK
    ]

stage = bytearray(ROLES * PAIRS_PER_ROLE * PAIR)
params = bytearray(PARAMS)
for index in range(256):
    struct.pack_into("<H", params, index * 2, bf16_bits(0.83 + (index % 29) * 0.004))
    struct.pack_into(
        "<H", params, 512 + index * 2, bf16_bits(0.91 + (index % 23) * 0.003)
    )
struct.pack_into("<f", params, 1024, 1.0e-6)

for role in range(ROLES):
    for pair in range(PAIRS_PER_ROLE):
        base = (role * PAIRS_PER_ROLE + pair) * PAIR
        stage[base + VALUES + CS : base + PAIR] = params
        m_macro, within = divmod(pair, 16)
        core_row, token_group = divmod(within, 4)
        if token_group >= 3:
            continue
        token0 = m_macro * 96 + core_row * 24 + token_group * 8
        for row in range(8):
            token = token0 + row
            if token >= 256:
                continue
            for dim in range(128):
                frequency = 1.0 / math.pow(10_000.0, (2 * dim) / 256.0)
                angle = token * frequency
                struct.pack_into(
                    "<H",
                    stage,
                    base + VALUES + (row * 256 + dim) * 2,
                    bf16_bits(math.cos(angle)),
                )
                struct.pack_into(
                    "<H",
                    stage,
                    base + VALUES + (row * 256 + 128 + dim) * 2,
                    bf16_bits(math.sin(angle)),
                )

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "activations.bin").write_bytes(padded)
(OUT / "weights.bin").write_bytes((SOURCE / "weights.bin").read_bytes())
(OUT / "stage-seed.bin").write_bytes(stage)
print(OUT)
