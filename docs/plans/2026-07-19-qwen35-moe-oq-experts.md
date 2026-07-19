# qwen35 MoE OQ experts — design + blockers (2026-07-19)

Goal axis P1 (Opus quant W{2,4,8}A8). Make the **qwen35** MoE family serve OQ
routed experts, mirroring the LFM2 MoE OQ landing (commit `996ac446d`) and the
minimax reference. **Status: BLOCKED, not landed.** The quantizer + loader are
correct and verified (an OQ qwen35 MoE model quantizes and LOADS cleanly), but
the dispatch cannot be validated end-to-end this session — see Blockers.

## What was designed + verified (then reverted, uncommitted)

Applied across 5 files, compiled clean, then reverted to keep the tree clean:

1. **Quantizer** (`hipfire-quantize/src/main.rs`, qwen35 per-expert 3D block
   ~L8543). New high-precedence OQ arm (after fp16/bf16, before the MQ variants,
   so it wins uniformly and ignores kmap MQ-promotion): `--format oq4` →
   `quantize_oq4g256` → `Oq4G256`; `--format oq8`/`oq8+` or `--w8-top` →
   `quantize_oqplus_compact` → `OqPlusCompact` (default 1% int8, expands to
   `Oq8G256` at load). Verified: `hipfire-quantize --input <fixture> --format oq4`
   emits OQ experts (label `OQ4-EXP`), model writes.

2. **Loader** (`hipfire-arch-qwen35/src/qwen35/loading.rs`). Ported minimax's
   per-expert repack — `oq4_ondisk_to_moe_blocks` (130 B → 132 B `[f32|128nib]`)
   and `oqplus_compact_to_moe_oq8_blocks` (→ 260 B `[f32|256 i8]`) — plus a
   `load_moe_expert` helper that peeks qt via `find_tensor_info`, repacks OQ
   (34/36) into the indexed-MoE kernel block layout and uploads raw tagged
   Oq4/Oq8 (the dense `oq4_arch_load`/`oq8_combined` layouts are the WRONG
   contract for `gemv_oq*_moe_*`), else falls through to `load_weight_tensor`.
   Paged path (`load_moe_ffn_paged`) refuses OQ with an honest error (block
   strides not threaded). **Verified: OQ model loads cleanly, no repack error.**

3. **Dispatch** (`moe_decode.rs`, `prefill_chunk.rs`, `mod.rs`). Added
   `MoeDecodeIndexedRoutedPath::{Oq4,Oq8}` + `routed_dtype_indexable_oq{4,8}`
   flags (`needs_x_rot_local` includes OQ — FWHT-rotated like MQ4). Routed
   gate_up/down OQ arms in decode (`gemv_oq{4,8}g256_moe_gate_up_k8_indexed` +
   `_down_..._batched_expanded`) and prefill Path 1 (`_batched` variants).
   Admissibility: added Oq4/Oq8 to `moe_prefill_quant_family_supported_for_arch`
   (RDNA, exclude gfx9). Left OQ OUT of `moe_grouped_gemm_supported_for_dtype` on
   purpose — OQ has no grouped-WMMA MoE kernel, so path2 eligibility stays false
   and OQ routes to the indexed Path 1; admissibility passes via
   `shared_matches_routed` (shared + routed both OQ).

## Blockers (why it can't land/validate yet)

1. **No OQ shared-expert-down kernel.** The shared expert down uses
   `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` (fused down GEMV + sigmoid
   shared-gate scale + residual). There is **no** `gemv_oq{4,8}g256_residual_sigmoid_scaled`
   sibling. `FusedGateUpOq4G256`/`Oq8G256` keys DO exist (shared gate+up is
   wireable), but without the down kernel the shared expert can't be OQ. And
   **every shipping qwen35 MoE has a shared expert** (`config.rs`:
   `has_shared_expert = shared_expert_intermediate_size > 0`; A3B=512, A10B,
   A17B). Keeping the shared expert in MQ/Q8 while routed is OQ breaks the
   `shared_matches_routed` admissibility (would need a mixed-layout redesign).

2. **Routed-only MoE forward is pre-existing unvalidated infra.** The only way to
   dodge the shared expert (a fixture with `shared_expert_intermediate_size=0`,
   `num_experts_per_tok=8`) hits an explicit gate in
   `generate.rs`:~2894 ("routed-only Qwen3 MoE forward is not validated yet …
   execution is gated to avoid GPU faults"), and forcing it with
   `HIPFIRE_QWEN35_ROUTED_ONLY_MOE_FORWARD=1` panics in `gemv.rs:118`
   (`gemv_f32` on a 1-D tensor) — a bug in that gated path, independent of OQ.

## Next steps to unblock

1. Write `gemv_oq4g256_residual_sigmoid_scaled_gpu_batched` (+ oq8) by cloning
   the hfq4 sibling and swapping the weight decode to the OQ grouped iu4/iu8
   path; add a parity test vs CPU oracle.
2. Wire the shared expert: gate+up via `run_fused_gate_up_key(FusedGateUpOq4G256/
   Oq8G256)` (prefill_chunk.rs ~L303), down via the new kernel (~L569), plus the
   decode shared-expert path and the scalar-gate dispatch.
3. Re-apply the dispatch/loader/quantizer design above.
4. Validate on `Qwen3.5-35B-A3B` (`/srv/huggingface/models--Qwen--Qwen3.5-35B-A3B`,
   67 GB source) `--format oq4` — coherence on halo (gfx1151).

## Reference

- Landed pattern: LFM2 MoE OQ commit `996ac446d`.
- minimax loader template: `hipfire-arch-minimax/src/minimax.rs` (`oq4_ondisk_to_moe_blocks`
  L284, `oqplus_compact_to_moe_oq8_blocks` L319, `upload_wt_oq` L257, expert loop
  L728-815) + dispatch `forward.rs` L425-635.
