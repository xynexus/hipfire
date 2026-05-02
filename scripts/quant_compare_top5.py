#!/usr/bin/env python3
"""Compare greedy_dump_top5 CSV/token outputs across quantization modes."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def load_tokens(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().splitlines() if x.strip()]


def load_top5(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: quant_compare_top5.py <baseline-prefix> <mode=prefix>...",
            file=sys.stderr,
        )
        return 2

    baseline_prefix = Path(sys.argv[1])
    base_tokens = load_tokens(baseline_prefix.with_suffix(".tokens"))
    base_top5 = load_top5(baseline_prefix.with_suffix(".top5.csv"))

    rows = []
    for spec in sys.argv[2:]:
        if "=" not in spec:
            print(f"bad mode spec: {spec}", file=sys.stderr)
            return 2
        mode, prefix_s = spec.split("=", 1)
        prefix = Path(prefix_s)
        toks = load_tokens(prefix.with_suffix(".tokens"))
        top5 = load_top5(prefix.with_suffix(".top5.csv"))

        n = min(len(base_tokens), len(toks), len(base_top5), len(top5))
        first_div = None
        top1_match = 0
        top5_overlap_sum = 0
        max_top1_delta = 0.0
        margin_at_div = None
        for i in range(n):
            if base_tokens[i] == toks[i]:
                top1_match += 1
            elif first_div is None:
                first_div = i
                margin_at_div = float(base_top5[i]["margin_top12"])

            base_ids = {int(base_top5[i][f"r{r}_id"]) for r in range(1, 6)}
            ids = {int(top5[i][f"r{r}_id"]) for r in range(1, 6)}
            top5_overlap_sum += len(base_ids & ids)
            max_top1_delta = max(
                max_top1_delta,
                abs(float(base_top5[i]["r1_logit"]) - float(top5[i]["r1_logit"])),
            )

        row = {
            "mode": mode,
            "steps_compared": n,
            "top1_agreement": top1_match / n if n else math.nan,
            "mean_top5_overlap": top5_overlap_sum / n if n else math.nan,
            "first_divergence": first_div,
            "baseline_margin_at_divergence": margin_at_div,
            "max_top1_logit_delta": max_top1_delta,
            "baseline_tokens": len(base_tokens),
            "mode_tokens": len(toks),
        }
        rows.append(row)

    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
