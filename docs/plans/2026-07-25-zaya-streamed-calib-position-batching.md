# ZAYA streamed calibration: position-slice batching

Status: planned 2026-07-25. Supersedes the per-token inner loop in
`crates/hipfire-arch-zaya/src/calibration_stream.rs` (`zaya-stream-v1`).

## Why

Measured on ZAYA1-8B, 226,506 rows x 40 blocks, `--sequences 128 --context 2048`:

| | value |
|---|---|
| wall clock | 4m04s per block, 2h35m for 40 |
| effective FLOPs | ~34 GFLOP/s against a ~50 TFLOP/s bf16 peak (**0.07%**) |
| `gpu_busy_percent` | 99% |
| weight bytes touched per token per block | **37.0 MB** |
| implied bandwidth | ~35 GB/s, i.e. **14%** of ~256 GB/s LPDDR5X |

The GPU reads as saturated while doing almost no arithmetic. The cause is not
launch overhead and not sync stalls (both were earlier misdiagnoses of mine): at
batch size 1 **every token re-reads the entire block's weights**. `expert_gate_up`
(16.8 MB) plus `expert_down` (8.4 MB) alone are 68% of that traffic.

Batching a position-slice of `S` tokens amortises the weight read across `S`:

| | batch 1 | batch 128 |
|---|---|---|
| weight bytes / token | 37.0 MB | 0.289 MB |
| weight traffic / block | 8.37 TB | 65 GB |
| at 256 GB/s | 32.7 s/block | 0.25 s/block |

Compute floor is ~0.4 s/block (8.2 TFLOP at ~20 TFLOP/s), so the forward's floor
is roughly **30 s for all 40 blocks**. Expect 5-15 min in practice, because
Hessian accumulation then dominates: `H = XᵀX` at hidden 2048 is ~4.2M MAC per
row, and ~6 dense capture points per block put it near 1.1e13 FLOP/block —
*more* than the forward. Batching `capture_by_id` to `n=S` batches that too.

## Why the scheduler already permits it

`MicrobatchPlanner::plan` emits rows **position-major**:

```rust
for position in time_start..time_end {
    for (sample_index, sample) in samples[sequence_start..sequence_end] { ... }
}
```

So each contiguous run of rows at one `position` is a set of tokens from
*different* sequences — mutually independent, sharing only the within-sequence
sequential dependency that attention and the CCA convolution care about. That run
is the natural batch. Two consequences fall out for free:

- every token in a slice is at the **same** position, so
  `zaya_rope_partial_qk_posbuf_f32` and one `positions` tensor need no change;
- ragged tails are automatic — a slice is however many sequences still have a
  token at that position.

## No new HIP kernels required

Already batched and family-neutral:

- `hipfire_runtime::weights::weight_gemm(gpu, w, x, y, batch_size)` — dense GEMM
  with the calibration capture tap already wired in
- `Gpu::rmsnorm_batched`
- `CalibCollector::capture_by_id(…, n, k)` already accepts `n` rows
- `hipfire_runtime::moe::grouped::GroupedMoeRoutingPlan`

**Correction (verified 2026-07-25, after this plan was first written): the batched
attention and KV-write kernels do NOT fit this use.** `attention_f32_batched`
declares `k_cache`/`v_cache` as `[max_seq, n_kv_heads, head_dim]` — a *single*
sequence's history, with `blockIdx.y` selecting the query row. Likewise
`kv_cache_write_f32_batched` writes `batch_size` rows into **one** `dst` at
`positions[b]`. Both are batched-*prefill* semantics: many query positions
against one sequence. A position-slice is the transpose of that — one position
across many *independent* sequences, each with its own cache.

Options, in preference order:
1. Keep attention and the KV write in the per-sequence loop alongside the CCA
   conv ops. Costs `S` launches per slice but **no** extra weight traffic: KV
   reads average ~2.1 MB/token against 37 MB of weights (~6% of the total) and
   are irreducible, since each sequence must read its own history.
2. `Gpu::kv_cache_write_f32_routed_batched(dst_ptrs, src, row_session_indices,
   positions, …)` scatters rows into *per-session* cache pointers in one launch —
   the right shape for the KV write. Worth adopting once (1) works.
3. A per-session batched f32 attention does not exist; do not write one until
   profiling shows the `S` attention launches matter after the GEMMs are fixed.

So the first increment batches the **GEMMs only** and leaves attention, the KV
write and the CCA conv ops per-sequence. That still captures ~94% of the weight
traffic, because it is the weights — not the activations — that were being
re-read `S` times.

Kernels that stay per-sequence, deliberately: `zaya_conv_window_f32`,
`zaya_conv1d_valid_f32`, `zaya_value_assemble_decode_f32`,
`zaya_qk_prep_decode_f32`. They carry per-sequence CCA state, but their weights
are 0.65 MB of 37 MB, so looping them costs ~2% of the win and avoids writing and
validating four new batched HIP kernels. Revisit only if profiling says the loop's
launch count matters after the GEMMs are fixed.

Kernels already taking an `n`/`s` and needing no change: `zaya_affine_residual_f32`,
`zaya_router_prep_f32` (EDA state becomes `[S, rh]`), `zaya_bias_gelu_f32`,
`zaya_add_conv_residual_qk_f32`, `zaya_qk_l2norm_qk_f32`.

## Shape of the change

1. `Scratch` grows a leading batch dim: `max_rows` from
   `job.options.max_rows`, buffers `[max_rows * dim]`.
2. `boundary` becomes `[max_rows * row_width]`: one `memcpy_htod` per slice in,
   one `download_f32` per slice out, replacing two transfers *per token*.
3. `SeqState` becomes per-slice-indexed: KV caches `[S, max_seq, kv_dim]`, conv
   rings `[S, conv_ch, pad]`, delayed values `[S, v_half]`, and a `positions`
   `GpuTensor` instead of a 4-byte `pos_buf`.
4. `forward_token` -> `forward_position_slice(rows: &[SampleRow], …)`.
5. MoE: download the `[S, n_route]` router logits once, do the existing host
   softmax/argmax per row (preserving bit-identical routing), group rows by
   winning expert, gather each group into contiguous scratch, one `weight_gemm`
   per *active* expert, scatter-accumulate with the per-row gate. With 16 experts
   and S=128 that is <=16 GEMMs of ~8 rows rather than 128 GEMVs — an 8x cut on
   the 68% of traffic the experts own, improving as S grows.
6. Bump the adapter version to `zaya-stream-v2`. The version is part of the run
   fingerprint, so this **deliberately invalidates existing checkpoints** rather
   than letting a v1 boundary be resumed by v2 math.

## Validation gate

Non-negotiable, because this rewrites the numerics' execution order:

1. `cargo test -p hipfire-arch-zaya --lib` (naming/capture-registry invariants).
2. Re-run the resident parity comparison exactly as for v1: streamed 1 sample x
   256 tokens, then `collect_artifacts --job-from` on the converted bf16 `.hfq`,
   then `artifact compare-calibration`. **Block 0 must stay numerically
   identical**; that was the v1 result and it is the strongest available signal
   that the block math is right.
3. Expect *more* drift than v1 beyond block 0, not less: GEMM reduction order
   differs from a GEMV's, so the 0.75% top-1 route-flip figure will move. Record
   the new number rather than asserting parity.
4. Re-measure GFLOP/s and weight bandwidth to confirm the win is real.

## Follow-ons not in this change

- `--kldref-rows`: KLDREF costs ~2.4e14 FLOP over all 226k rows, but a KLD
  *reference* is statistically ample over a few thousand positions. Subsetting
  cuts the dominant term 20-50x. `logZ` is a full-vocabulary logsumexp, so the
  two-stage lm-head shortlist cannot help — every logit is needed for the
  denominator.
- Fold the KLD reduction into block 39's microbatch loop: block 39's output rows
  *are* the final hidden states, so this saves re-reading 2.09 GB of boundary.
  Minutes against the GEMM's hours.
- Thread tokens through the grouped-MoE dispatch callback so qwen3.5 also gets
  token/stratum profiles.
