"""
D4: Per-group FWHT pre/post absmax comparison.

For every 256-element group of the (flattened) tensor:
  1. pre_max[g]  = max(|W[g*256:(g+1)*256]|)  before rotation
  2. post_max[g] = max(|FWHT(signs1, signs2, W[...])|)  after rotation
  3. ratio[g]    = post_max[g] / max(pre_max[g], 1e-9)

A ratio < 1 means FWHT redistributed energy and reduced the per-group
peak (good for asymmetric quant: the per-group scale fits a smaller range,
so quant noise on the bulk distribution shrinks). A ratio close to 1
means FWHT didn't help. A ratio > 1 means FWHT made things worse for
this group.

Reports tensor-level mean / p99 / max of the ratio. The interpretation
in synthesis: if mean reduction < 0.5 across the whole model, FWHT is
absorbing outlier energy effectively and the quant cliff cannot be
weight-side. If close to 1.0, the per-group outliers persist post-rotation
and downstream NRMSE damage tracks them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from quant_ops import (  # noqa: E402
    GROUP_SIZE,
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    cpu_fwht_256_batch,
    gen_fwht_signs,
)


@dataclass
class D4Record:
    n_groups: int
    pre_max_mean: float
    pre_max_p99: float
    pre_max_max: float
    post_max_mean: float
    post_max_p99: float
    post_max_max: float
    ratio_mean: float       # mean(post_max / pre_max) over groups
    ratio_p50: float
    ratio_p99: float
    ratio_max: float        # worst-case group: where FWHT helped least or hurt most
    n_groups_ratio_below_0_5: int
    n_groups_ratio_above_1_0: int

    def to_json(self) -> dict:
        return {
            "n_groups": self.n_groups,
            "pre_max": {
                "mean": self.pre_max_mean,
                "p99": self.pre_max_p99,
                "max": self.pre_max_max,
            },
            "post_max": {
                "mean": self.post_max_mean,
                "p99": self.post_max_p99,
                "max": self.post_max_max,
            },
            "reduction_ratio": {
                "mean": self.ratio_mean,
                "p50": self.ratio_p50,
                "p99": self.ratio_p99,
                "max": self.ratio_max,
                "n_groups_below_0_5": self.n_groups_ratio_below_0_5,
                "n_groups_above_1_0": self.n_groups_ratio_above_1_0,
            },
        }


def run_d4(weights: np.ndarray,
           signs1: np.ndarray | None = None,
           signs2: np.ndarray | None = None) -> D4Record:
    """Compute D4 on a tensor of any shape. Flattens row-major, pads to
    a multiple of 256 with zeros, processes per group via the vectorized
    batch FWHT path.

    The pad zeros land in their own (final) group; their ratio is 0/0,
    treated as 1.0 (no rotation effect on a zero block).

    Vectorized: pre_max + rotated + post_max are all batch ops on
    (n_groups, 256). For a 5.7M-element tensor this is ~0.5s vs 43s
    for the scalar per-group Python loop.
    """
    if signs1 is None:
        signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    if signs2 is None:
        signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    n = flat.shape[0]
    n_padded = ((n + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    n_groups = n_padded // GROUP_SIZE

    padded = np.zeros(n_padded, dtype=np.float32)
    padded[:n] = flat
    grouped = padded.reshape(n_groups, GROUP_SIZE)

    pre_max = np.abs(grouped).max(axis=1).astype(np.float32)
    rotated = cpu_fwht_256_batch(grouped, signs1, signs2)
    post_max = np.abs(rotated).max(axis=1).astype(np.float32)

    safe_pre = np.maximum(pre_max, np.float32(1e-9))
    ratio = post_max / safe_pre
    # Pad-only groups have pre_max==0, ratio == post_max / 1e-9 → huge.
    # Reset those to 1.0 (zero rotates to zero; ratio is meaningless).
    ratio[pre_max < np.float32(1e-9)] = 1.0

    p99 = float(np.percentile(ratio.astype(np.float64), 99))
    p50 = float(np.percentile(ratio.astype(np.float64), 50))

    return D4Record(
        n_groups=n_groups,
        pre_max_mean=float(pre_max.mean()),
        pre_max_p99=float(np.percentile(pre_max, 99)),
        pre_max_max=float(pre_max.max()),
        post_max_mean=float(post_max.mean()),
        post_max_p99=float(np.percentile(post_max, 99)),
        post_max_max=float(post_max.max()),
        ratio_mean=float(ratio.mean()),
        ratio_p50=p50,
        ratio_p99=p99,
        ratio_max=float(ratio.max()),
        n_groups_ratio_below_0_5=int((ratio < 0.5).sum()),
        n_groups_ratio_above_1_0=int((ratio > 1.0).sum()),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    rng = np.random.default_rng(0)

    # Gaussian: FWHT preserves L2 norm exactly; per-group max often drops
    # because the L_inf norm is lowered by averaging through the butterfly.
    w = rng.standard_normal((4, 1024)).astype(np.float32) * 0.05
    rec = run_d4(w)
    print(f"[d4 self-test] gaussian({w.shape}): "
          f"groups={rec.n_groups} "
          f"ratio(mean={rec.ratio_mean:.3f} p50={rec.ratio_p50:.3f} "
          f"p99={rec.ratio_p99:.3f} max={rec.ratio_max:.3f}) "
          f"below_0_5={rec.n_groups_ratio_below_0_5}/{rec.n_groups}")

    # One extreme outlier in a group: FWHT spreads the outlier energy
    # across all 256 elements so post_max drops to outlier/16; ratio ~ 1/16.
    w_out = (rng.standard_normal(256).astype(np.float32) * 0.001)
    w_out[42] = 1.0
    rec_out = run_d4(w_out)
    print(f"[d4 self-test] one-outlier(256): "
          f"pre_max={rec_out.pre_max_max:.4f} post_max={rec_out.post_max_max:.4f} "
          f"ratio={rec_out.ratio_mean:.4f}")
    # Per FWHT theory, ratio should be ~1/16 (each output element gets
    # outlier/16 contribution; bulk is small so post_max ~ outlier/16).
    assert rec_out.ratio_mean < 0.10, \
        f"single-outlier group should rotate to ratio ~ 1/16; got {rec_out.ratio_mean}"

    # Many similar-magnitude values: FWHT can't reduce the max (worst case).
    w_many = (rng.standard_normal(256).astype(np.float32) * 0.0)
    w_many[:] = 0.5  # uniform large values
    rec_many = run_d4(w_many)
    print(f"[d4 self-test] uniform-large(256): "
          f"pre_max={rec_many.pre_max_max:.4f} post_max={rec_many.post_max_max:.4f} "
          f"ratio={rec_many.ratio_mean:.4f}")
    # Uniform constant input: FWHT(constant) yields one big peak (sum) +
    # 255 zeros; post_max = sum * 1/16 = 256 * 0.5 / 16 = 8.0, vs pre_max
    # 0.5. Ratio is 16. So FWHT *hurts* this case.
    assert rec_many.ratio_mean > 1.0, \
        f"uniform-large should rotate to higher peak; got {rec_many.ratio_mean}"

    print("[d4 self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
