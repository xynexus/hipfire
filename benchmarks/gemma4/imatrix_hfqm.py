#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors

"""Minimal native HFQM writer for imatrix-only calibration packages."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


def write_u32(output: BinaryIO, value: int) -> None:
    output.write(struct.pack("<I", value))


def write_u64(output: BinaryIO, value: int) -> None:
    output.write(struct.pack("<Q", value))


def write_imatrix_hfqm(
    path: Path,
    metadata: dict,
    sums: dict[str, np.ndarray],
    counts: dict[str, int],
) -> None:
    """Write normalized projection-input statistics in native HFQM v1 form."""

    names = sorted(sums)
    if not names:
        raise ValueError("cannot write an empty imatrix package")
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    index = bytearray(struct.pack("<I", len(names)))
    payloads: list[np.ndarray] = []
    for name in names:
        count = counts.get(name, 0)
        if count <= 0:
            raise ValueError(f"imatrix {name} has no contributing tokens")
        values = np.asarray(sums[name] / count, dtype="<f4")
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"imatrix {name} must be a nonempty vector")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"imatrix {name} contains invalid activation energy")
        entry_name = f"{name}.imatrix".encode()
        if len(entry_name) > 0xFFFF:
            raise ValueError(f"imatrix entry name is too long: {name}")
        index.extend(struct.pack("<H", len(entry_name)))
        index.extend(entry_name)
        index.extend(struct.pack("<BB", 2, 1))  # F32, one dimension
        index.extend(struct.pack("<I", values.size))
        index.extend(struct.pack("<I", 0))  # group_size
        index.extend(struct.pack("<Q", values.nbytes))
        payloads.append(values)

    metadata_offset = 32
    data_start = metadata_offset + len(metadata_json) + len(index)
    data_offset = (data_start + 4095) & ~4095
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(b"HFQM")
            write_u32(output, 1)
            write_u32(output, 24)  # Gemma 4 architecture ID
            write_u32(output, len(names))
            write_u64(output, metadata_offset)
            write_u64(output, data_offset)
            output.write(metadata_json)
            output.write(index)
            output.write(bytes(data_offset - data_start))
            for values in payloads:
                output.write(values.tobytes(order="C"))
            output.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
