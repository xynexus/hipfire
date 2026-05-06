"""
D1: Per-tensor NRMSE between bf16 reference and MQ4G256-FWHT round-trip.

For every weight tensor:
  1. Apply production MQ4G256-FWHT pipeline (seeds 42 / 1042) to a copy of
     the f32 reference, then dequantize.
  2. Compute NRMSE = sqrt(MSE) / sqrt(var(reference)) over the entire tensor.
  3. Compute mean cosine similarity over the flattened tensor (for compatibility
     with 2026-05-05 quant_recon_results.json).
  4. Report sample size = total element count (no subsampling).

Output is one JSON record per tensor; survey_runner emits as JSONL.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Import sibling modules without requiring an installed package.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from quant_ops import (  # noqa: E402
    GROUP_SIZE,
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    gen_fwht_signs,
    quantize_then_dequantize_mq4g256_fwht_vectorized as _qd_vectorized,
    nrmse,
    mean_cosine_similarity,
)


@dataclass
class D1Record:
    """Output of run_d1 on a single tensor."""
    n_elements: int
    nrmse_mq4g256_fwht: float
    cos_sim_mq4g256_fwht: float
    # Dequantized array dropped before serialization; only summary metrics shipped.

    def to_json(self) -> dict:
        return {
            "n_elements": self.n_elements,
            "nrmse_mq4g256_fwht": self.nrmse_mq4g256_fwht,
            "cos_sim_mq4g256_fwht": self.cos_sim_mq4g256_fwht,
        }


def run_d1(weights: np.ndarray,
           signs1: np.ndarray | None = None,
           signs2: np.ndarray | None = None) -> D1Record:
    """Compute D1 on a single tensor (any shape, any size).

    Caller is responsible for FP cast (the runner already returns f32).
    Memory: O(2 * n_elements) for the dequant copy. The runner streams
    one tensor at a time so the peak is bounded.
    """
    if signs1 is None:
        signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    if signs2 is None:
        signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    flat = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    recon = _qd_vectorized(flat, signs1, signs2)
    n = flat.shape[0]

    return D1Record(
        n_elements=n,
        nrmse_mq4g256_fwht=nrmse(flat, recon),
        cos_sim_mq4g256_fwht=mean_cosine_similarity(flat, recon),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    rng = np.random.default_rng(0)

    # Standard normal: should give ~0.99+ cos sim.
    x = rng.standard_normal(2048).astype(np.float32) * 0.1
    rec = run_d1(x)
    print(f"[d1 self-test] gaussian(2048): n={rec.n_elements} "
          f"NRMSE={rec.nrmse_mq4g256_fwht:.4f} cos={rec.cos_sim_mq4g256_fwht:.4f}")
    assert rec.cos_sim_mq4g256_fwht > 0.99, "gaussian round-trip cos too low"

    # Outlier-heavy row: cos still high; NRMSE is actually LOWER because
    # var(reference) is dominated by the outlier (one big value bumps var
    # far more than it bumps MSE), so MSE/var shrinks. Keep the assertion
    # on cos sim only — that's the metric that doesn't degenerate under
    # variance dominance.
    y = (rng.standard_normal(2048).astype(np.float32) * 0.05)
    y[7] = 5.0  # one large outlier
    rec_y = run_d1(y)
    print(f"[d1 self-test] outlier(2048): n={rec_y.n_elements} "
          f"NRMSE={rec_y.nrmse_mq4g256_fwht:.4f} cos={rec_y.cos_sim_mq4g256_fwht:.4f}")
    assert rec_y.cos_sim_mq4g256_fwht > 0.99, "outlier row cos sim too low"

    # Tiny 256-element tensor (matches one block exactly).
    z = rng.standard_normal(256).astype(np.float32) * 0.1
    rec_z = run_d1(z)
    print(f"[d1 self-test] block(256): n={rec_z.n_elements} "
          f"NRMSE={rec_z.nrmse_mq4g256_fwht:.4f} cos={rec_z.cos_sim_mq4g256_fwht:.4f}")
    assert rec_z.n_elements == 256

    print("[d1 self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
