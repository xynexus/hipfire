"""Butterfly residual transform — reference Python implementation.

Residual form per ButterflyQuant (arXiv:2509.09679). The full quantization
rotation applied to MQ4G256 activations is:

    T(theta) = B_residual(theta) ∘ FWHT_canonical

where:
  - FWHT_canonical = D2 · H_256 · D1   (matches hipfire's cpu_fwht_256)
  - B_residual(theta) = product over 8 stride-doubling layers, each applying
    128 independent SO(2) Givens rotations parameterized by theta[layer, pair].
  - theta has shape [8, 128], one angle per (layer, pair).
  - At theta = 0 → B_residual = I → T = FWHT_canonical (byte-equal current pipeline).

Mathematical aside on the residual form choice:
H_256 (normalized) has det = +1 but each H_2 block has det = -1 (a reflection,
not a rotation). A single Givens block R(theta) has det = +1 for all theta, so
pure-Givens butterflies cannot continuously connect to H_2 — H_2 lives in the
disconnected (reflection) component of O(2). The residual form sidesteps this:
we fix the base transform as the existing FWHT (which already encodes the
reflection structure) and learn a small additive rotation on top.

This matches ButterflyQuant's bisectable design: residual = 0 is a known-good
init (= current MQ4+AWQ baseline), and optimization can only improve from
there (in expectation under the calibration loss).
"""

import numpy as np

LAYER_COUNT = 8
N = 256
PAIRS_PER_LAYER = N // 2  # 128


def _pair_indices(layer: int, pair_idx: int) -> tuple[int, int]:
    """Map (layer, pair_idx) to (i, j) for the stride-doubling butterfly.

    Layer l has stride = 1 << l. Within a block of 2*stride elements there
    are `stride` independent pairs. The flattened pair_idx in [0, 128)
    enumerates: block_idx * stride + within_block.
    """
    stride = 1 << layer
    block_idx = pair_idx // stride
    within_block = pair_idx % stride
    i = block_idx * (2 * stride) + within_block
    j = i + stride
    return i, j


def butterfly256(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Apply B_residual(theta) to x along the last axis.

    Args:
        x: float ndarray of shape [..., 256].
        theta: float ndarray of shape [8, 128].

    Returns:
        y = B_residual(theta) · x  (rotation applied on last axis).
    """
    assert x.shape[-1] == N, f"Expected last dim {N}, got {x.shape[-1]}"
    assert theta.shape == (LAYER_COUNT, PAIRS_PER_LAYER), (
        f"theta must be [{LAYER_COUNT}, {PAIRS_PER_LAYER}], got {theta.shape}"
    )

    y = x.astype(np.float32, copy=True)
    for layer in range(LAYER_COUNT):
        cos_t = np.cos(theta[layer]).astype(np.float32)
        sin_t = np.sin(theta[layer]).astype(np.float32)
        for pair_idx in range(PAIRS_PER_LAYER):
            i, j = _pair_indices(layer, pair_idx)
            a = y[..., i].copy()
            b = y[..., j].copy()
            c = cos_t[pair_idx]
            s = sin_t[pair_idx]
            y[..., i] = c * a - s * b
            y[..., j] = s * a + c * b
    return y


def butterfly256_inverse(y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Apply B_residual(theta)^T = B_residual(-theta) to y.

    Inverts butterfly256 exactly up to numerical roundoff.
    """
    assert y.shape[-1] == N
    assert theta.shape == (LAYER_COUNT, PAIRS_PER_LAYER)

    x = y.astype(np.float32, copy=True)
    for layer in range(LAYER_COUNT - 1, -1, -1):
        cos_t = np.cos(theta[layer]).astype(np.float32)
        sin_t = np.sin(theta[layer]).astype(np.float32)
        for pair_idx in range(PAIRS_PER_LAYER):
            i, j = _pair_indices(layer, pair_idx)
            a = x[..., i].copy()
            b = x[..., j].copy()
            c = cos_t[pair_idx]
            s = sin_t[pair_idx]
            x[..., i] = c * a + s * b
            x[..., j] = -s * a + c * b
    return x


def fwht_residual_init() -> np.ndarray:
    """Identity butterfly residual: theta = 0.

    At this theta, butterfly256 is the identity transform, so composing with
    FWHT_canonical reproduces FWHT_canonical byte-equal. This is the
    bisectable init point (= current MQ4+AWQ baseline).
    """
    return np.zeros((LAYER_COUNT, PAIRS_PER_LAYER), dtype=np.float32)


def cpu_fwht_256(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Reference for hipfire's cpu_fwht_256 = D2 · H_256 · D1 · x.

    Matches the math of the existing cpu_fwht_256 kernel for Phase 1
    verification purposes. Byte-equality vs the actual hipfire C kernel is a
    Phase 8 concern (Rust port) — here we only need internal self-consistency.

    Args:
        x: float ndarray with last dim = 256.
        signs1: ±1 ndarray length 256 (input diagonal D1).
        signs2: ±1 ndarray length 256 (output diagonal D2).
    """
    assert x.shape[-1] == N
    assert signs1.shape == (N,) and signs2.shape == (N,)

    y = (x.astype(np.float32) * signs1.astype(np.float32)).copy()
    h = 1
    while h < N:
        for i in range(0, N, 2 * h):
            for j in range(i, i + h):
                a = y[..., j].copy()
                b = y[..., j + h].copy()
                y[..., j] = a + b
                y[..., j + h] = a - b
        h *= 2
    y = y / np.float32(np.sqrt(N))
    y = y * signs2.astype(np.float32)
    return y


def fwht_signs_from_seed(seed: int) -> np.ndarray:
    """Generate ±1 sign table of length 256 from a seed.

    NOTE: matches numpy's default_rng PRNG, NOT hipfire's exact LCG. Sufficient
    for Phase 1 math verification; Phase 8 (Rust port) must use the production
    PRNG to achieve byte-equality with the C kernel.
    """
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=N, dtype=np.int8) * 2 - 1
    return signs.astype(np.float32)
