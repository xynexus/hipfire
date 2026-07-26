#!/usr/bin/env python3
"""Replace the dumped R68 stage sentinel with valid persistent RoPE/params."""

import math
import os
from pathlib import Path
import struct

from ml_dtypes import bfloat16
import numpy as np


OUT = Path(os.environ.get("R69_INPUT_DIR", "~/.hipfire/npu/r69-chain-input")).expanduser()
PAIR, PAIRS_PER_ROLE, ROLES = 8192, 37, 5
VALUE_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
STAGE_BYTES = VALUE_BYTES + 2048
stage = bytearray(STAGE_BYTES)


def bf16_bits(value):
    return int(np.asarray([value], dtype=bfloat16).view(np.uint16)[0])


for role in range(ROLES):
    for pair in range(PAIRS_PER_ROLE):
        base = (role * PAIRS_PER_ROLE + pair) * PAIR + 4096
        for row in range(8):
            token = pair * 8 + row
            if token >= 256:
                continue
            for dim in range(128):
                frequency = 1.0 / math.pow(10_000.0, (2 * dim) / 256.0)
                angle = token * frequency
                struct.pack_into("<H", stage, base + (row * 256 + dim) * 2, bf16_bits(math.cos(angle)))
                struct.pack_into("<H", stage, base + (row * 256 + 128 + dim) * 2, bf16_bits(math.sin(angle)))

params = VALUE_BYTES
for index in range(256):
    struct.pack_into("<H", stage, params + index * 2, bf16_bits(0.83 + (index % 29) * 0.004))
    struct.pack_into("<H", stage, params + 512 + index * 2, bf16_bits(0.91 + (index % 23) * 0.003))
struct.pack_into("<f", stage, params + 1024, 1.0e-6)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "stage-seed.bin").write_bytes(stage)
print(OUT)
