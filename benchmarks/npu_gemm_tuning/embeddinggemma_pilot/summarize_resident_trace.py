#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert EmbeddingGemma resident trace output into durable CSV evidence.

The input is combined stdout/stderr from ``embed_e2e_npu_opus`` with
``HIPFIRE_EMBED_TRACE_RESIDENT=1`` and, optionally, ``HIPFIRE_XDNA_TRACE=1``.
One process may contain several encodes; a repeated/decreasing layer number
starts the next sample.  Finalize records delimit samples when available.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PAIR = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
PHASE_PREFIX = "embeddinggemma_resident_phase "
FINALIZE_PREFIX = "embeddinggemma_resident_finalize "
DISPATCH_PREFIX = "xdna_dispatch_cumulative "

LAYER_FIELDS = (
    "setup_ms",
    "attn_prepare_ms",
    "attn_pack_sync_ms",
    "attn_run_ms",
    "attention_ms",
    "unit_rms_ms",
    "ffn_ms",
    "tail_ms",
    "next_prep_ms",
    "residual_prep_ms",
    "output_materialize_ms",
    "prep_and_output_ms",
    "total_ms",
)


@dataclass
class Sample:
    index: int
    layers: list[dict[str, str]] = field(default_factory=list)
    finalize: dict[str, str] = field(default_factory=dict)
    dispatch: dict[str, str] = field(default_factory=dict)


def pairs(line: str) -> dict[str, str]:
    return dict(PAIR.findall(line))


def parse(lines: Iterable[str]) -> list[Sample]:
    samples: list[Sample] = []
    current: Sample | None = None
    previous_layer: int | None = None
    latest_dispatch: dict[str, str] = {}

    def ensure_sample() -> Sample:
        nonlocal current
        if current is None:
            current = Sample(len(samples))
            samples.append(current)
        return current

    for raw in lines:
        line = raw.strip()
        if line.startswith(DISPATCH_PREFIX):
            latest_dispatch = pairs(line)
        elif line.startswith(PHASE_PREFIX):
            values = pairs(line)
            layer = int(values["layer"])
            if current is not None and previous_layer is not None and layer <= previous_layer:
                current = Sample(len(samples))
                samples.append(current)
            sample = ensure_sample()
            sample.layers.append(values)
            previous_layer = layer
        elif line.startswith(FINALIZE_PREFIX):
            sample = ensure_sample()
            sample.finalize = pairs(line)
            sample.dispatch = latest_dispatch.copy()
            current = None
            previous_layer = None
    if current is not None and not current.dispatch:
        current.dispatch = latest_dispatch.copy()
    return [sample for sample in samples if sample.layers or sample.finalize]


def dispatch_delta(current: dict[str, str], previous: dict[str, str]) -> tuple[int, float, float]:
    def number(record: dict[str, str], key: str) -> float:
        return float(record.get(key, "0"))

    return (
        int(number(current, "count") - number(previous, "count")),
        number(current, "submit_ms") - number(previous, "submit_ms"),
        number(current, "wait_ms") - number(previous, "wait_ms"),
    )


def write_layers(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample", "state", "layer", *LAYER_FIELDS))
        writer.writeheader()
        for sample in samples:
            for layer in sample.layers:
                writer.writerow(
                    {
                        "sample": sample.index,
                        "state": "cold" if sample.index == 0 else "primed",
                        "layer": layer["layer"],
                        **{name: layer.get(name, "") for name in LAYER_FIELDS},
                    }
                )


def write_samples(path: Path, samples: list[Sample]) -> None:
    fields = (
        "sample",
        "state",
        "layers",
        "layer_total_ms",
        "final_norm_mean_ms",
        "dense_l2_ms",
        "finalize_total_ms",
        "dispatches",
        "submit_ms",
        "wait_ms",
    )
    previous_dispatch: dict[str, str] = {}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            count, submit_ms, wait_ms = dispatch_delta(sample.dispatch, previous_dispatch)
            if sample.dispatch:
                previous_dispatch = sample.dispatch
            writer.writerow(
                {
                    "sample": sample.index,
                    "state": "cold" if sample.index == 0 else "primed",
                    "layers": len(sample.layers),
                    "layer_total_ms": f"{sum(float(row['total_ms']) for row in sample.layers):.3f}",
                    "final_norm_mean_ms": sample.finalize.get("final_norm_mean_ms", ""),
                    "dense_l2_ms": sample.finalize.get("dense_l2_ms", ""),
                    "finalize_total_ms": sample.finalize.get("total_ms", ""),
                    "dispatches": count if sample.dispatch else "",
                    "submit_ms": f"{submit_ms:.3f}" if sample.dispatch else "",
                    "wait_ms": f"{wait_ms:.3f}" if sample.dispatch else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="combined stdout/stderr trace log")
    parser.add_argument("--layers-out", type=Path, required=True)
    parser.add_argument("--samples-out", type=Path, required=True)
    args = parser.parse_args()

    samples = parse(args.trace.read_text().splitlines())
    if not samples:
        raise SystemExit("no EmbeddingGemma resident trace records found")
    args.layers_out.parent.mkdir(parents=True, exist_ok=True)
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    write_layers(args.layers_out, samples)
    write_samples(args.samples_out, samples)
    print(
        f"parsed {len(samples)} samples / {sum(len(sample.layers) for sample in samples)} layers"
    )


if __name__ == "__main__":
    main()
