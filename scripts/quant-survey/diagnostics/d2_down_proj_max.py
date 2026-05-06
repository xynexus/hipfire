"""
D2: Per-expert (or per-tensor) down_proj absmax + ratio statistics.

For each weight tensor (typically a `down_proj.weight` per layer or
per (layer, expert)):
  1. Per-row absmax: `row_max[i] = max_j |W[i, j]|` over the K (input) axis.
  2. Per-row median absmax: `row_med[i] = median_j |W[i, j]|`.
  3. Per-row tail ratio: `ratio[i] = row_max[i] / max(row_med[i], 1e-9)`.
  4. Distribution stats on both arrays: mean, p50, p90, p99, p99_9, max.

The 2026-05-05 finding "down_proj p99 max ~ 37M" was the **ratio**, not the
absolute weight magnitude (per `expert_absmax_stats.py:124`). This module
reports BOTH so neither signal is hidden in the synthesis step.

Outlier classification at the model level (synthesis) uses `ratio_p99`
z-score across tensors of the same projection, NOT raw absmax.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class D2Record:
    n_rows: int
    row_max_mean: float
    row_max_p50: float
    row_max_p90: float
    row_max_p99: float
    row_max_p99_9: float
    row_max_max: float

    row_med_mean: float
    row_med_p50: float

    ratio_mean: float
    ratio_p50: float
    ratio_p90: float
    ratio_p99: float
    ratio_p99_9: float
    ratio_max: float

    def to_json(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "row_max": {
                "mean": self.row_max_mean,
                "p50": self.row_max_p50,
                "p90": self.row_max_p90,
                "p99": self.row_max_p99,
                "p99_9": self.row_max_p99_9,
                "max": self.row_max_max,
            },
            "row_med": {
                "mean": self.row_med_mean,
                "p50": self.row_med_p50,
            },
            "ratio": {
                "mean": self.ratio_mean,
                "p50": self.ratio_p50,
                "p90": self.ratio_p90,
                "p99": self.ratio_p99,
                "p99_9": self.ratio_p99_9,
                "max": self.ratio_max,
            },
        }


def _percentile_set(arr: np.ndarray, percentiles: list[float]) -> dict[float, float]:
    """One-shot percentile computation; returns {p: value} float dict."""
    if arr.size == 0:
        return {p: float("nan") for p in percentiles}
    out = np.percentile(arr.astype(np.float64), percentiles)
    return {p: float(v) for p, v in zip(percentiles, out)}


def run_d2(weights: np.ndarray) -> D2Record:
    """Compute D2 on a 2D tensor [M, K].

    For 3D-stacked MoE tensors of shape [n_experts, M, K], the runner
    is expected to slice expert-by-expert and call this function once
    per expert, NOT pass the 3D array directly. This function asserts
    weights is 2D to enforce that contract.
    """
    if weights.ndim != 2:
        raise ValueError(
            f"D2 expects 2D weight matrix [M, K]; got shape {weights.shape}. "
            "For stacked-3D MoE tensors, slice on the leading expert axis "
            "before calling run_d2."
        )

    abs_w = np.abs(weights.astype(np.float32, copy=False))
    row_max = abs_w.max(axis=1)
    row_med = np.median(abs_w, axis=1)
    # Avoid division-by-zero on rows that are entirely zero.
    safe_med = np.maximum(row_med, np.float32(1e-9))
    ratio = row_max / safe_med

    rmax_p = _percentile_set(row_max, [50.0, 90.0, 99.0, 99.9])
    ratio_p = _percentile_set(ratio, [50.0, 90.0, 99.0, 99.9])

    return D2Record(
        n_rows=int(weights.shape[0]),
        row_max_mean=float(row_max.mean()),
        row_max_p50=rmax_p[50.0],
        row_max_p90=rmax_p[90.0],
        row_max_p99=rmax_p[99.0],
        row_max_p99_9=rmax_p[99.9],
        row_max_max=float(row_max.max()),

        row_med_mean=float(row_med.mean()),
        row_med_p50=float(np.median(row_med)),

        ratio_mean=float(ratio.mean()),
        ratio_p50=ratio_p[50.0],
        ratio_p90=ratio_p[90.0],
        ratio_p99=ratio_p[99.0],
        ratio_p99_9=ratio_p[99.9],
        ratio_max=float(ratio.max()),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    rng = np.random.default_rng(0)

    # Plain Gaussian, no outliers: ratio should be modest (~5-20).
    w = (rng.standard_normal((128, 1024)).astype(np.float32) * 0.05)
    rec = run_d2(w)
    print(f"[d2 self-test] gaussian({w.shape}): "
          f"rmax(p50={rec.row_max_p50:.4f} max={rec.row_max_max:.4f}) "
          f"ratio(p50={rec.ratio_p50:.1f} p99={rec.ratio_p99:.1f} max={rec.ratio_max:.1f})")
    assert rec.n_rows == 128
    assert rec.ratio_p50 < 100.0, "gaussian ratio_p50 should be modest"

    # Insert one extreme outlier in row 7: ratio_max should jump.
    w_out = w.copy()
    w_out[7, 42] = 1000.0
    rec_out = run_d2(w_out)
    print(f"[d2 self-test] outlier({w_out.shape}): "
          f"rmax(p99={rec_out.row_max_p99:.4f} max={rec_out.row_max_max:.4f}) "
          f"ratio(p99={rec_out.ratio_p99:.1f} max={rec_out.ratio_max:.1f})")
    assert rec_out.row_max_max > 999.0, "outlier should appear in row_max_max"
    # 1000.0 outlier vs ~0.034 median (gaussian * 0.05 scale, half-normal mean
    # of |x|) gives ratio ~29k. The 2026-05-05 data showed 37M ratios on
    # actual MoE down_proj — those rows have a much wider absmax-to-median
    # gap than this synthetic test. Just verify the outlier is detected.
    assert rec_out.ratio_max > 1e4, \
        f"outlier ratio_max should be at least 10000; got {rec_out.ratio_max}"
    assert rec_out.ratio_p99 > rec.ratio_p99, "outlier should bump ratio_p99"

    # All-zeros row: must not crash on division-by-zero.
    w_zero = w.copy()
    w_zero[3, :] = 0.0
    rec_z = run_d2(w_zero)
    print(f"[d2 self-test] zeroed-row: ratio_max={rec_z.ratio_max:.1f}")
    assert np.isfinite(rec_z.ratio_max), "zero row must not produce NaN/inf"

    # 3D shape rejection.
    try:
        run_d2(rng.standard_normal((4, 16, 32)).astype(np.float32))
    except ValueError as e:
        print(f"[d2 self-test] 3D rejection: OK ({e!s:.80}...)")
    else:
        raise AssertionError("D2 should reject 3D input")

    print("[d2 self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
