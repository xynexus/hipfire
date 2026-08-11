# Serving the 122B: paged experts load, but decode is blocked

Opened 2026-08-11 after discovering the `Qwen3.5-122B-A10B--oq4.25++.hfq`
artifact had never been executed — only verified by size, format, hash and
tensor count. It does not run.

## What works

`HIPFIRE_QWEN35_RESIDENCY_MODE=qwen_moe_modules` (or
`HIPFIRE_QWEN35_PAGED_EXPERTS=1`) makes the 122B **load** on a 124 GB box:

| | fully resident | paged experts |
|---|---|---|
| 122B peak GTT | ~115 GB, then OOM on lm_head | **9.32 GB** |
| streamed | 71.91 GiB (all) | **6.22 GiB of 71.91** |
| result | `hipMalloc: out of memory` | `{"type":"loaded"}`, 48 layers |

12,288 routed expert modules register against an 8 GiB cache
(`HIPFIRE_QWEN35_EXPERT_CACHE_MB`, default 8192). This is the mechanism that
makes large MoE viable here at all, and it reframes the earlier "a 397B needs
~800 GB resident" projection: under paging the resident set is dense weights
plus a bounded cache, not the whole model.

Paged experts are OFF by default (`qwen35_paged_experts_enabled`, mod.rs:171).

## What blocks decode

Generation panics at `generate.rs:2950` with
`moe.decode-routed-dtype-unsupported-no-fallback`. The chain:

1. `check_moe_decode_supported` (dispatch/pipeline/mod.rs:198) rejects when
   `!use_gpu_topk && !routed_experts_resident`.
2. Paging deliberately leaves `ffn.experts` empty (layout.rs:87), so
   `routed_experts_resident` is false — **the CPU fallback is gone by design**
   and GPU-top-K is the only path.
3. GPU-top-K for OQ needs `routed_dtype_indexable_oq4`/`oq8`, both gated on
   `oq_indexed_decode` (mod.rs:1788).
4. That reads `HIPFIRE_QWEN35_MOE_OQ_INDEXED`, whose comment states the indexed
   routed OQ decode kernels are disabled **"while debugging their finite-KLD
   failure"** (mod.rs:1838). Gated in 753df2b27 (2026-07-22); the commit has no
   body and there is no other documentation of the failure.

**So paged decode depends on a kernel switched off for producing wrong numbers.**
Setting `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1` will make the 122B emit tokens. Do not
treat that as a fix — it re-enables a known correctness bug, and "it generates"
is not evidence the numbers are right.

## Dead ends — do not re-derive

- **BF16 experts are NOT the decode blocker.** The 122B carries 644 experts kept
  at source precision by `--expert-coverage-policy preserve-undercovered`, and
  the paged loader rejects quant type 16 at load. That is real, and
  `paged_moe_quant_hint` now explains it at the point of rejection. But it is
  only a load-time gate: the 35B (`moe-a-nointra.hfq`, zero preserved experts,
  pure OQ) loads under paging and fails decode with the identical error. So
  "requantize so every routed expert is OQ" would NOT fix this.
- **Mapping BF16 -> F16 in `paged_moe_dtype_for_quant` was tried and reverted.**
  The model then loads and panics mid-generation instead of failing cleanly at
  load — strictly worse. See the doc comment there.
- **`MoePrefillDtypes::from_ffn` handles paged layers correctly.** It reads
  `expert_gate_up_dtypes`, which the loader populates for paged layers; the empty
  `ffn.experts` vector does not break dtype detection.

## Also worth knowing

- `load_any_as_f32` (loading.rs) dequantizes ANY quant type to **F32** on CPU,
  including `3 => dequant_q8f16`. The lm_head is stored `qt=3` (Q8F16) and then
  widened on load — stored narrow, resident wide. If the head is moved to a
  two-pass form (coarse 4-bit + bf16-lut3), this is the path that has to change
  with it.
- Fully-resident footprint is ~2 bytes/param regardless of the 4.25-bit on-disk
  format: 2B measured 4.27 GB (predicted 3.76), 35B measured 69.15 GB (predicted
  69.32). Steady-state, not a load transient — the series is flat through
  generation. Whether that is dequantization or accounting was not resolved,
  because paging made it moot for the load question.

## Kernel review of the indexed OQ decode path (2026-08-11)

Static review of the kernels behind `HIPFIRE_QWEN35_MOE_OQ_INDEXED`. **No
arithmetic bug found.** Recording what was CLEARED so the next person does not
re-read it:

- **FWHT signs match.** Quantizer `gen_fwht_signs(42, 256)` / `(1042, 256)`
  (main.rs:4374) is byte-identical to the runtime's `ensure_mq_signs`
  (dispatch/mod.rs:2251). The "MQ sign LUT" is a shared FWHT LUT and is correct
  for OQ. OQ weights ARE stored FWHT-rotated (`oqplus_compact_ldlq_pack` calls
  `cpu_fwht_256`), so this mattered.
- **`gemv_oq8g256_moe_gate_up_k8_indexed`** — 260 B stride, f32 scale at offset 0
  (260 is 4-aligned so the load is legal), 32 lanes x 8 int8 = 256/group,
  `base` stays in `[0,K)`, wave32 shuffle reduction correct for lane 0.
- **`gemv_oq8g256_moe_down_k8_indexed_batched_expanded`** — same, plus a correct
  `((bid*K_TOP)+krank)*M + row` output offset. Atomic-free by design.
- **`gemv_oq4g256_moe_gate_up_k8_indexed`** — the `DOG` macro extracts all eight
  nibbles in the right order (low nibble of each byte -> even element), matching
  the packer's `(q4[2i] & 0xf) | (q4[2i+1] << 4)`. Note it correctly keeps TWO
  different offsets: `boff = tid*4` (bytes into the nibble array) and
  `base = g*256 + tid*8` (elements into x). Quad loop + `tail` covers
  `groups_per_row % 4` exactly.
- **`oqplus_compact_to_moe_oq8_blocks`** — f16->f32 scale, nibble order,
  `sign_extend_i4` against the packer's +-7 clamp, and outliers correctly
  overwriting their int4 slot. Stride agrees with the kernel.

### Smells found (real, none yet proven causal)

1. **Zero test coverage.** Nothing in `crates/*/tests` or `tests/` references the
   indexed OQ kernels. They were shipped disabled and never differentially
   validated against the non-indexed path — which is why the failure is still
   undiagnosed a month later.
2. **`oqplus_compact_to_moe_oq8_blocks` is duplicated in at least four crates**
   (`hipfire-runtime/src/oq_moe.rs`, `arch-lfm2moe`, `arch-minimax`, +qwen35). A
   binary format contract maintained by copy-paste; if one copy drifts the result
   is silent wrongness, not a compile error.
3. **The paged down path drops AWQ.** `run_paged_mixed_routed_decode`
   (moe_decode.rs:565) calls `gpu.rotate_x_mq_batched(...)` when the AWQ-aware
   `rotate_x_mq_batched_for` exists and safely falls back when no sidecar is
   present. Currently HARMLESS — the 122B has zero routed-expert `awq_scale`
   sidecars — but latent the moment routed experts carry AWQ. Note the gate_up
   comment's argument ("all experts consume the same post-rmsnorm x, so identical
   AWQ scales") does NOT extend to down_proj, whose input is each expert's own
   hidden state.
4. **Invariant comment drift.** `x_rot_local.expect("use_gpu_topk implies
   routed_gate_up_{mq4,mq6,mq2_lloyd,paro} ...")` omits the OQ cases that now
   reach it.

### MEASURED (2026-08-11): the failure is real, severe, and NOT in the kernels

End-to-end, same artifact (`moe-a-nointra.hfq`), same reference, same 8 chunks,
only the flag differs:

| | mean_kld | ppl |
|---|---|---|
| flag OFF (CPU fallback) | 0.030367 | 7.46 |
| flag ON (indexed OQ) | **5.108296** | **1171.67** |

**168x worse.** The disable is fully justified and NOT stale. Decode is broken
too, so this is not prefill-specific: `"The capital of France 是斯"` — degenerate
and drifting into CJK, the same "structural attractor" shape the softmax comments
in moe_decode.rs describe.

Yet every kernel on that path is numerically exact. Two new differential
harnesses (`cargo run --release -p hipfire-rdna --example ...`):

- `parity_gemv_oq8_moe_indexed` — gate_up non-batched AND batched (the batched
  one is what prefill uses), each vs a CPU oracle AND vs the production
  `gemv_oq8_grouped`. Three-way so a disagreement is attributable: `grouped` is a
  path a real model already generates coherent text through, so if IT disagrees
  the ORACLE is wrong, not the kernel.
- `parity_moe_down_combine_oq8_indexed` — the batched down GEMV, plus
  `moe_down_combine_k8_batched` (where the top-K routing weight is applied, since
  neither GEMV applies it). Residual seeded NON-ZERO so an overwrite cannot hide
  behind a zeroed buffer; routing weights non-uniform and NOT summing to 1 so a
  dropped/doubled weight cannot hide behind normalisation.

All pass at ~1e-4 against ~0.15-0.25 tolerances, across shapes including the real
122B ones (gate_up M=2048 K=3072, down M=3072 K=1024) and edge cases (single
group K=256, n_exp=2, N=1..4).

**So the bug is in what feeds the kernels or where their outputs go, not in the
arithmetic.** The harnesses fabricate `expert_ptrs`, `topk_indices`,
`topk_weights` and `x` by hand — which is exactly why they pass while the real
path fails.

### Cleared during this pass

- **Top-K.** `moe_topk_renorm_k8` documents a "PRE-SOFTMAXED probs" contract, and
  both branches honour it (`gpu.softmax_f32(router_logits)` immediately before,
  moe_decode.rs:1008 and :1030). Raw logits would have given correct SELECTION
  with wrong WEIGHTS — a perfect finite-KLD candidate — but it is not happening.
  The kernel hardcodes `K_TOP 8` and takes no `k` argument; harmless here since
  this model is top-8, but it is WRONG for the 397B at top-10.
- **Layout consistency.** With the flag on, the loader converts routed experts to
  the interleaved 260 B MoE-block layout (loading.rs:6304), and prefill dispatches
  the same indexed kernels (prefill_chunk.rs:1246/1668). Decode and prefill agree.
- **FWHT signs**, kernel arithmetic, and the OQ+C -> OQ8 conversion (see above).

### BUG FOUND (latent, not the cause of the above)

`loading.rs:6304` gates the MoE-block conversion on
`env == Some("1")`, while `qwen35_moe_oq_indexed_decode_enabled()`
(mod.rs:1838) accepts `Some("1") | Some("on")`. Setting
`HIPFIRE_QWEN35_MOE_OQ_INDEXED=on` therefore enables indexed DISPATCH while
leaving experts in CANONICAL layout — the indexed kernels then read the int8 code
array's first four bytes as an f32 scale. Guaranteed garbage, silently. Either
accept both spellings in both places or reject anything but "1" in both.

### ROOT CAUSE (2026-08-12): per-expert AWQ scales vs one shared rotation

`--ldlq` stores W·s per routed expert; the forward must compute (W·s)·(x/s).
The indexed MoE path rotates x ONCE per layer — using the SHARED expert's
awq_scale (`prefill_chunk.rs:148`) or none at all — and hands that single buffer
to every routed expert. That is only valid if all routed experts share one AWQ
scale. **They do not.** Extracted from the 35B-A3B oq4.25++ (`hfq extract`,
control: same tensor extracted twice hashes identically):

| tensor | awq_scale payload sha |
|---|---|
| layer0 experts.0.gate_up | `4c91eb73933a0f91` |
| layer0 experts.0.gate_up (control re-extract) | `4c91eb73933a0f91` |
| layer0 experts.1.gate_up | `e9f11b566d0968f0` |
| layer0 experts.7.gate_up | `53213482a95463b9` |
| layer0 shared_expert.gate_proj | `2ba22a862df684ec` |

Routing is exactly why: each expert sees a different token subset, so a
different imatrix, so a different s. Every routed expert was off by
s_e/s_shared per channel — which is the measured layer-0 divergence (cosine
0.244, norm 16x) and the 168x KLD blowup.

The MQ4 comment asserting "identical imatrix -> byte-identical AWQ scales" is
false for routed experts; the OQ path inherited that assumption.

### Fix in progress — Option 1 (per-expert rotation)

Chosen over (2) one shared AWQ scale per layer at quantize time and (3) no AWQ
on routed experts, because it is the only option that does not require
re-quantizing existing artifacts. Cost: prefill x_rot scratch goes from
`[N x K]` to `[N x K_TOP x K]` (+176 MB at N=2048, dim=3072) and 8x the rotation
bandwidth; decode is free (+0.1 MB).

LANDED AND VERIFIED:

1. `rotate_x_mq_awq_indexed_batched` (kernel + dispatch wrapper). Reads
   `topk_indices` on-device, selects each slot's scale from a pointer table,
   expands `[N x K] -> [N x K_TOP x K]`. Null pointer -> plain rotation, so mixed
   artifacts stay correct. `parity_rotate_x_mq_awq_indexed` is BIT-EXACT
   (0.00000000) vs the trusted `rotate_x_mq_awq_batched` driven one slot at a
   time, with the null arm asserted to be exercised.
2. `expert_gate_up_awq_ptrs` / `expert_down_awq_ptrs` on `MoeFfnWeights`, built
   like `expert_gate_up_ptrs`. `None` when no expert has a sidecar (plain
   rotation stays byte-identical). Paged mode passes `None` and WARNS if the
   artifact has expert sidecars rather than degrading silently.
3. All four indexed gate_up kernels (oq4/oq8 x batched/non-batched) now index x
   PER SLOT. `parity_gemv_oq8_moe_indexed` updated to feed a different x per
   slot — a shared-x kernel can no longer pass it — and passes on three shapes.

NOT YET WIRED (the reason the flag must stay off):

4. Scratch buffer `[N x K_TOP x dim]` + the `rotate_x_mq_awq_indexed_batched`
   call, in BOTH `prefill_chunk.rs` and `moe_decode.rs`. The scratch lives in
   three structs (`qwen35/mod.rs:525`, `prefill_batch.rs:93`,
   `mtp_head.rs:350`), each with its own alloc/free path.
5. Re-run the steer trace and confirm layer-0 cosine goes to ~1.0, then the KLD.

**Landmine while 4 is outstanding:** the kernels now expect a per-slot x, but
the call sites still pass the shared `pbs.x_rot_batch`. Unreachable today (the
flag gates it off), but anyone enabling `HIPFIRE_QWEN35_MOE_OQ_INDEXED` before
step 4 lands gets garbage from the new contract rather than the old one. Finish
step 4 or revert step 3 before enabling.

### Where to look next

Per-kernel arithmetic is the wrong place — it has been checked. Remaining
unexamined: the expert pointer table, `moe_down_combine_k8_batched` and where the
top-K routing weight is applied (neither GEMV applies it, so it must be in the
combine — a double-apply or missing-apply there is exactly a finite-KLD error),
the OQ4 down kernel, and the `_batched` gate_up variant.

The efficient move is a **differential test**, not more reading: run the same
expert weights and activations through the indexed and non-indexed paths and diff
the outputs. That localises in one run what static review cannot, and it is the
coverage gap in smell 1. Enable the flag inside that harness only — never to
"see if it generates".

## Next

Deciding whether to debug the indexed OQ decode kernels is a correctness call,
not a config change. Until then the 122B artifact cannot be served on this box,
and neither could a 397B built the same way — which is worth settling BEFORE
spending 33-42 h quantizing one.
