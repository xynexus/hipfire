
---

# Stage A2e — the activation sweep, re-run on the deployed (rotated) grid

Stage A2/A2b re-run with `maybe_quant_act` quantizing in the codec's FWHT basis.
Everything else identical: weight tier `oq4`, KV clean, real text, LR 1e-4
warmup+cosine, 120 steps. **These supersede the A2/A2b activation arms.**

`a16` is carried over unchanged and was not re-run: `maybe_quant_act` returns
before rotating when the tier is `None`, so that arm is byte-identical.

| activation | deploy loss | after | **recovered** | best | wall |
|---|---|---|---|---|---|
| `a16` | 0.2211 | 0.1392 | **37.0%** | @120 | (carried over) |
| `a8` | 0.2209 | 0.1399 | **36.7%** | 0.1393 @100 | 1980 s |
| `a4o16` | 0.2713 | 0.1908 | **29.7%** | 0.1799 @100 | 2037 s |
| `a4o8` | 0.2784 | 0.1970 | **29.3%** | 0.1908 @80 | 2093 s |
| `a4` | 0.3557 | 0.2499 | **29.7%** | 0.2480 @100 | 2036 s |

`a8` landing within 0.09% of `a16` on the deploy loss (0.2209 vs 0.2211), from an
independently executed run, is a useful consistency check on the harness.

## Correction: promotion does NOT restore recoverability

Stage A2b concluded that outliers destroy *recoverability*, not just accuracy,
because on the unrotated grid recovery went 23.0% (`a4`) → 36.8% (`a4o8`) →
38.9% (`a4o16`), i.e. back to the A16 rate.

On the deployed grid that effect **disappears**:

| | `a4` | `a4o8` | `a4o16` |
|---|---|---|---|
| unrotated (superseded) | 23.0% | 36.8% | 38.9% |
| **rotated (deployed)** | **29.7%** | **29.3%** | **29.7%** |

Flat. Every A4 variant recovers ~29–30% regardless of how many values are
promoted. **The rotation already handles the outliers**; what remains at A4 is a
uniform 4-bit precision limit that promotion reduces in *magnitude* but does not
make more recoverable. The A2b claim was an artifact of measuring an unrotated
grid where the outliers were still present.

What survives: A4 does recover less than A8/A16 (29.7% vs 36.7%), so four-bit
activations are genuinely harder to recover from — just not because of outliers,
and not fixable by promoting them.

## Promotion still helps the loss — but loses to `iu4x2` A8

The whole activation budget is the **0.1348 nats** between `a4` (0.3557) and
`a8` (0.2209). Against it:

- `a4o8` closes **57%** of the gap (0.3557 → 0.2784), +0.125 bits/value.
- `a4o16` closes **63%** (→ 0.2713), +0.25 bits/value. Knee still at 8; the
  8→16 step buys 0.0071 nats against 0→8's 0.0773.

That is more than the SNR probe predicted (41% / 54%), so the probe understated
promotion — worth noting, since it means SNR is a conservative proxy for KL here.

**But `a8` (0.2209) still beats `a4o16` (0.2713) outright, and `a8` is reachable
with no mask at all.** `gemm_oq_compact_iu4x2_wmma.hip` carries the int8
activation as two radix-16 int4 digits through the same iu4 core, and its
weight tile is IDENTICAL across the two passes — so A8 costs one extra activation
plane and WMMA issue, and nothing in weight traffic. Two int4 planes are exactly
the bytes of one int8 activation.

**Recommendation: do not build a per-group activation-promotion kernel.** It
needs a runtime position mask, lands short of A8 on quality, and the existing
two-pass `iu4x2` path already reaches A8 at close to A4's weight bandwidth. The
per-group promotion machinery stays valuable as a *measurement* tool and on the
weight side (where `oq4.25` is a clear win), but not as an activation kernel.

## Ironic resolution

The original question was whether activation quant should be split into two
passes across the rotation. The answer is that a two-pass activation scheme *is*
the right answer — just not that one. `iu4x2`'s radix-16 digit split is a
two-pass scheme over the same weight tile, and it wins.
