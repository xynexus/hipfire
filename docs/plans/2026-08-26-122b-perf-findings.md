# Qwen3.5-122B-A10B on halo — where it actually stands (2026-08-26)

Measured on gfx1151, `122b-lmbf16.hfq` (69.3 GB) and
`Qwen3.5-122B-A10B--oq4.25++fix.hfq` (68.6 GB).

## 1. It loads, and the memory problem is gone

    weights loaded: 64.56 GiB payload, total_resident_bytes = 68.99 GB

The 3.5x GTT blowup that used to make this model unloadable (expanded experts +
2 MiB rounding, ~137 GiB resident for a 63.9 GiB artifact) is fixed: compact
residency holds and resident tracks the artifact. Loads in 34-65s at 1.0-2.0
GiB/s.

## 2. It is still INCOHERENT — perf work on it is premature

    <think>\n\nHere's a thinking'skeyider'\n<think>\nHere's a thinking, [\n</think>\n\n theur\n\n the.焄

BYTE-IDENTICAL garbage from both artifacts, including the lm_head->BF16 one, so
the lm_head fix is not the (current) cause. This matches the commit log:
`a51be9b78 fix(moe): two real layout bugs found chasing the 122B; the 122B
itself is NOT fixed`. Later commits exonerate compact residency specifically
(`f3d3a5efd` 106 real-weight checks, `ce7a3d25f` 626 unrotated tensors perfect).

CONTROL: `Qwen3.6-35B-A3B--oq4.25++` — same arch, same kernels — is perfectly
coherent at 61.5 tok/s decode. So this is 122B-specific, not a MoE-path
regression. Note the garbage is a CORRUPTED version of the A3B's correct
opening ("Here's a thinking process:" -> "Here's a thinking'skeyider'"), which
reads as slightly-wrong numerics rather than a broken path.

## 3. Baseline numbers

| | 122B | 35B-A3B | 27B (dense-ish) |
|---|---|---|---|
| prefill tok/s | **29.1** | 69.6 | 306 |
| decode tok/s | 23-27 | 61.5 | 69.9 |

**Prefill is FLAT at 29.1 tok/s** — n=61, n=92 and n=658 all measure 29.0-29.1.
Flat in n is the signature of no amortisation at all: ~34 ms/token, ~8.6 GB read
per token against ~5.3 GB of active weights.

## 4. Root cause of the prefill ceiling: mixed layers cannot use grouped prefill

The artifact is MIXED-PRECISION per layer — 1288 expert tensors are `Oq8G256`
while 23,288 are `OqPlusCompact`, and per the loader comment that is **37 of 48
layers**. Two consequences, one known and one not:

**a. The batched-prefill gate declines the whole model.**
`classify_routed_expert_dtypes` has three variants — `Uniform`,
`QuantWithFullPrecisionFallback{quant, full}` (quant + F16/BF16 only) and
`Invalid`. A two-QUANT mix (compact + Oq8) matches none, so it is `Invalid`, and
`routed_supported` maps `Invalid => false`. Meanwhile `routed_oq_mixed_compact`
is computed on the very next lines, is TRUE, and says the compact family can
serve the layer from its per-expert stride table. The admission never consults
it.

Confirmed by instrumenting the predicate:

    [moe-admit] router=BF16 router_ok=true scalar_gate=BF16 shared_gate_ok=true
                profile=Invalid oq_mixed_compact=true gu=OqCompactG256 down=OqCompactG256

**b. Admitting it changes NOTHING, and that is the real finding.** Patched
experimentally to admit mixed-compact: `[pbs-gate] verdict=true`, and prefill
stayed at 29.0 -> 29.0 (n=92) and 29.1 -> 29.1 (n=658).

rocprofv3 says why. Every dominant kernel is a **GEMV**, and the call counts are
one per token per layer (658 tokens x 48 layers = 31,584 ~ 31,680):

    gemv_oq_compact_grouped_v3                            213840   5624.6 ms  26.8%
    gemv_oq_compact_moe_gate_up_k8_indexed_batched         31680   4121.7 ms  19.7%
    gemv_oq_compact_moe_down_k8_indexed_batched_expanded   31680   3040.4 ms  14.5%
    gemv_oq_compact_grouped_v3_splitk                      31680   2393.2 ms  11.4%
    gemm_bf16_x_bf16_wmma                                  63360   1749.3 ms   8.3%

**THE KERNEL ASYMMETRY.** `gemm_oq_compact_moe_grouped_wmma` takes
`int block_stride` as a **launch-wide scalar**. The decode GEMV
(`oqc_moe_row_dot`) takes a **per-expert stride table** with `block_stride == 0`
as the Oq8 sentinel. So mixed layers are dispatchable at DECODE and not at
PREFILL, and they fall back to per-token indexed GEMVs regardless of what the
admission gate says.

## 5. What would actually fix the prefill ceiling

Give `gemm_oq_compact_moe_grouped_wmma` (and its `_f32` sibling) the same
per-expert stride table the decode GEMV already has, then let the admission
consult `routed_oq_mixed_compact` instead of requiring a valid profile. Both
halves are needed; either alone does nothing, which is exactly what the
experiment above demonstrates.

Sizing: grouped MoE was measured at +31% prefill on the 27B (`43653ea3d`), but
the 122B is starting from FULLY per-token, and its uniform-compact sibling
(35B-A3B) prefills at 69.6 vs this model's 29.1. Note also E=256/K=8 means
tokens-per-expert is 8x lower than the 27B at equal n, so the grouped crossover
sits much higher than that commit's N>=64.

⚠️ Do this AFTER coherence. A faster incoherent model is not progress, and the
mixed-layer path is precisely the region the open coherence bug lives in.

# Coherence bisect against the 35B control (2026-08-27)

## The control

`Qwen3.6-35B-A3B--oq4.25++` — same arch (qwen3_5_moe), same kernels, same
quant family — is COHERENT at 61.5 tok/s decode. Census of the two:

| | expert dtypes | gate_up AWQ missing | down AWQ missing |
|---|---|---|---|
| 35B (coherent) | **uniform** `OqPlusCompact` | 0 (0%) | 0 (0%) |
| 122B (incoherent) | **mixed** compact + Oq8 | 644 (5.2%) | 2018 (16.4%) |

The control never exercises a mixed layer, nor an expert without an AWQ
sidecar. 37 of the 122B's 48 layers are mixed.

## Ruled out

- **lm_head.** `--fix` and `122b-lmbf16` (which differs ONLY in lm_head:
  2 Bf16Lut3 tensors vs 1) emit BYTE-IDENTICAL garbage.
- **OQ8 router path.** `HIPFIRE_OQ8_ROUTER=1` changes nothing.
- **Per-expert missing AWQ.** The design is documented ("a 0 entry means that
  expert has no sidecar") and `rotate_x_mq_awq_indexed_batched` honours it:
  `awq = ptrs == nullptr ? nullptr : ptrs[topk_indices[slot]]`, then
  `if (awq != nullptr)`. Null → plain rotation, per expert. Correct.
- **Compact residency / quantization of unrotated data.** Already exonerated
  upstream: `f3d3a5efd` (106 real-weight checks), `ce7a3d25f` (626 unrotated
  tensors at cosine 1.000000).

## What the artifacts say — the sharpest evidence

Three 122B artifacts exist, and which ones LOAD is itself the clue:

| artifact | routed experts | profile | residency | outcome |
|---|---|---|---|---|
| `--oq4.25++fix` / `122b-lmbf16` | compact + **Oq8G256** | `Invalid` | compact-resident, 68.99 GB | loads, **INCOHERENT** |
| `122b-passA` | compact + **BF16** | `QuantWithFullPrecisionFallback` (supported) | EXPANDED | 148.7 GiB, refuses to load |
| `--oq4.25++` | compact + Q8F16 | — | — | (the original digit-soup artifact) |

`servable_by_stride_table` accepts `OqCompactG256 | Oq8G256` only, so the
BF16-fallback variant cannot be compact-resident and does not fit, while the
Oq8-fallback variant rides the per-expert stride table and does.

**The only loadable 122B is the one on the newest, least-covered path.** The
stride table (`cc532499d`) is what makes a mixed layer dispatchable at all, its
`block_stride == 0` sentinel selects the Oq8 arm inside a compact GEMV, and
nothing else in the tree exercises it — the 35B is uniform, and no tiny fixture
has mixed experts (`fixture.rs` builds `qwen3_5_moe` and `qwen3_5_moe_indexed`,
both uniform).

## Next step, and it is a fixture not a kernel

Build a tiny fixture with MIXED compact + Oq8 routed experts. That is the one
thing that would reproduce this in seconds instead of a 65-second load of a
69 GB artifact, and its absence is why the path shipped uncovered. The same
pattern as the DeltaNet contraction bug: the invariant existed, nothing ran it.

Only once it reproduces small is it worth deciding between:
  - the stride-table Oq8 arm inside `oqc_moe_row_dot` reading an activation
    rotated for the compact basis, and
  - the promoted Oq8 experts having been quantized in a basis the runtime does
    not reproduce (they carry NO awq_scale sidecar — exactly 644 of them).
