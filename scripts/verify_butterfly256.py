"""Phase 1 verification: butterfly256 + identity-residual reduces to FWHT.

Three independent checks per IMPLEMENTATION_PLAN.md Phase 1:

  (1) Identity: butterfly256(theta=0, x) == x byte-equal.
      At the bisectable init point, the residual must be a no-op.

  (2) Round-trip: butterfly256_inverse(butterfly256(x, theta), theta) ≈ x
      for 1000 random thetas. Verifies the inverse correctness (and that
      the forward map is orthonormal up to F32 roundoff).

  (3) FWHT residual composition: butterfly256(FWHT(x), theta=0) == FWHT(x)
      byte-equal. Combined with (1), confirms theta=0 reduces the full
      pipeline T = B_residual ∘ FWHT to the current FWHT.

Phase 1 stop condition: any check failing → halt and document. The math is
load-bearing for all downstream phases.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.butterfly_core import (  # noqa: E402
    LAYER_COUNT,
    N,
    PAIRS_PER_LAYER,
    butterfly256,
    butterfly256_inverse,
    cpu_fwht_256,
    fwht_residual_init,
    fwht_signs_from_seed,
)

IDENTITY_TOL = 1e-5
ROUNDTRIP_TOL = 1e-4
COMPOSITION_TOL = 1e-5


def check_identity_residual(n_trials: int = 1000, seed: int = 2026) -> float:
    rng = np.random.default_rng(seed)
    theta = fwht_residual_init()
    max_err = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal(N).astype(np.float32)
        y = butterfly256(x, theta)
        err = float(np.abs(y - x).max())
        if err > max_err:
            max_err = err
    return max_err


def check_inverse_roundtrip(n_trials: int = 1000, seed: int = 2026) -> float:
    rng = np.random.default_rng(seed + 1)
    max_err = 0.0
    for _ in range(n_trials):
        theta = rng.uniform(-np.pi, np.pi, size=(LAYER_COUNT, PAIRS_PER_LAYER)).astype(
            np.float32
        )
        x = rng.standard_normal(N).astype(np.float32)
        y = butterfly256(x, theta)
        x_back = butterfly256_inverse(y, theta)
        err = float(np.abs(x_back - x).max())
        if err > max_err:
            max_err = err
    return max_err


def check_fwht_composition(n_trials: int = 1000, seed: int = 2026) -> float:
    rng = np.random.default_rng(seed + 2)
    signs1 = fwht_signs_from_seed(42)
    signs2 = fwht_signs_from_seed(1042)
    theta = fwht_residual_init()
    max_err = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal(N).astype(np.float32)
        fwht_only = cpu_fwht_256(x, signs1, signs2)
        composed = butterfly256(fwht_only, theta)
        err = float(np.abs(composed - fwht_only).max())
        if err > max_err:
            max_err = err
    return max_err


def main() -> int:
    print("Phase 1 verification: butterfly256 + identity residual")
    print("=" * 60)

    err1 = check_identity_residual()
    pass1 = err1 < IDENTITY_TOL
    print(
        f"[{'PASS' if pass1 else 'FAIL'}] (1) butterfly256(theta=0, x) == x       "
        f"max_err = {err1:.3e}  (tol {IDENTITY_TOL:.0e})"
    )

    err2 = check_inverse_roundtrip()
    pass2 = err2 < ROUNDTRIP_TOL
    print(
        f"[{'PASS' if pass2 else 'FAIL'}] (2) inv(bfly(x, θ), θ) ≈ x              "
        f"max_err = {err2:.3e}  (tol {ROUNDTRIP_TOL:.0e})"
    )

    err3 = check_fwht_composition()
    pass3 = err3 < COMPOSITION_TOL
    print(
        f"[{'PASS' if pass3 else 'FAIL'}] (3) bfly(FWHT(x), θ=0) == FWHT(x)       "
        f"max_err = {err3:.3e}  (tol {COMPOSITION_TOL:.0e})"
    )

    all_pass = pass1 and pass2 and pass3
    print("=" * 60)
    print(f"{'ALL CHECKS PASS' if all_pass else 'PHASE 1 HALT — FIX MATH BEFORE PROCEEDING'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
