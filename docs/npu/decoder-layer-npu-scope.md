# Scoping a resident DECODER-layer NPU path (llama / qwen3.6-moe)

Written 2026-07-30 after measuring the existing NPU primitives end to end and
deconstructing FastFlowLM. Companion to `wire-in-r6-prefill-offload.md`, which covers
the GEMM primitive; this covers what is missing to make a *served model* run on the NPU.

## Where the NPU actually sits today

- **Nothing NPU is wired into causal-LLM serving.** The only serving touchpoints are
  `hipfire-runtime/src/dflash.rs` (drafter) and
  `hipfire-serving-core/src/qwen3_embedding.rs`. `NpuGemmMp` is referenced only by
  examples and `hipfire-rdna/src/gtt_share.rs`.
- **Layer-level resident executors already exist — for EmbeddingGemma only.**
  `resident_ffn.rs` (R25), `resident_attention_w8.rs`, `resident_embedding_layer.rs`
  (R34). These are the FLM-shaped design: whole stages resident on the array, weights
  uploaded once, activations streamed.
- **The zero-copy path exists.** `gtt_share.rs::SharedGttBuffer` PRIME-exports a GTT
  dma-buf so CPU, NPU and GPU address the same physical pages on this UMA APU. The
  caveat in its header stands: a GPU *compute* kernel touching it still needs
  `hipHostRegister` on `as_mut_slice()`.

## Why the EmbeddingGemma executors do not just work

`NpuResidentFfnW4::load_cached` validates a manifest against hard-coded
`m=256 k=768 intermediate=1152 out=768` and refuses anything else. The shapes are baked
into the compiled schedule, not read from the artifact. Target shapes are far larger:

| | EmbeddingGemma | llama-3.2-1B | qwen3.6-35B-A3B (per expert) |
|---|---|---|---|
| k (hidden) | 768 | 2048 | 4096 |
| intermediate | 1152 | 8192 | routed, 256 experts/layer |

So this is not a parameterisation change. Each target needs its own
generated-and-cached xclbin set, and the R25/R34 L1 budgets do not carry over — the R6
sweep already showed L1 is the binding constraint (`MT=16 KCHUNK=32` fails to build at
all; see `wire-in-r6-prefill-offload.md`).

## The work, in dependency order

1. **Generalise the resident executors to read shapes from their manifest** instead of
   asserting one shape. Prerequisite for every other model; mechanical.
2. **Generate + cache decoder-layer xclbins per (model, shape) set**, the way
   `r6_cache.sh` does for the GEMM. Offline (aiecc/Python stays out of the hot path per
   AGENTS.md).
3. **Upload path — DONE and verified 2026-07-30.** `OpusPackedMatrix::from_payload`
   consumes a raw `.hfq` tensor payload directly, and `group_dense_i8` expands OQ4 to
   dense signed bytes in `-8..=7` — exactly `NpuGemm`'s `w_int4` contract. Proven by
   `crates/hipfire-runtime/examples/npu_artifact_gemv.rs`, which reads a REAL oq4++
   artifact, uploads it with `NpuGemm::upload_weights`, runs on the NPU and checks every
   output column against a CPU reference over the same expanded weights:

   | artifact | tensor | shape | result |
   |---|---|---|---|
   | `Llama-3.2-1B-Instruct-nc--oq4++` | `layers.0.mlp.gate_proj` | K=2048 N=8192 | PASS, 8192/8192 cols |
   | `Qwen3.6-35B-A3B-nc--oq4++` | `layers.0.mlp.shared_expert.down_proj` | K=512 N=2048 | PASS, 2048/2048 cols |

   Both carry their AWQ sidecar and expand fully inside int4 range.

   **SCOPE (important):** that PASS verifies the *integer* GEMM — correct int8xint4 dot
   products over real oq4++ weight bytes — against a CPU reference that likewise omits
   scales. It does NOT verify dequantized output. Oq4G256 carries a per-256-element group
   scale, and `OpusPackedMatrix::reference_f32` shows the required math is
   **per-group int32 -> `* act_scale[group][row] * weight_scale[col]` -> accumulate in
   f32**. `NpuGemm::run_resident` sums int32 across all K-chunks (and at KCHUNK=32 a chunk
   already spans two groups), which destroys that scaling and therefore cannot produce
   correct oq4 output.

   **Consequence: use `NpuGemmMp`, not `NpuGemm`.** `NpuGemmMp::run_resident_batch`
   already emits "one int32 matrix per K group for the caller's format-independent scale
   reconstruction" — exactly the contract oq4 needs. This is the second time surveying
   the crate first would have saved work (see the resident-weights duplication above).
   The upload path is real; the *numerics* still need the per-group form.
4. **Route the arch through it**, gated behind a flag (`HIPFIRE_NPU_PREFILL` was the name
   the wire-in doc proposed), with the GPU path as fallback.

## Which axis to target first — prefill, and why

Measured end to end on the GPU path (see `/home/sadara/flm-benchmarks.md`), hipfire
already beats FLM on decode and TTFT for both models and loses only on prefill:

| | hipfire (GPU) | FLM (NPU) |
|---|---|---|
| llama-1B decode / prefill | 75.96 tok/s / 80.8 t/s | 60.1 / ~2750 |
| 35B-A3B decode / prefill | 32.77 tok/s / 36.4 t/s | 13.4 / ~290 |

Prefill is the only losing axis, and it is where the NPU primitive is *already* strong:
2.17 TOPS measured end-to-end through `NpuGemm`, against the ~1.74 TOPS that FLM's 290
t/s on the 35B implies. **A prefill-only offload for the MoE is therefore the smallest
change that flips the last axis**, and the MoE is `prefill = "full"` in
`docs/model-support.toml` where llama is only `partial`.

Decode does NOT need the NPU: 75.96 and 32.77 tok/s already beat FLM by 1.26x and 2.45x
on the GPU. Moving decode to the NPU would be a regression unless a fused decoder layer
lands — the measured per-linear NPU decode ceiling is ~38 tok/s for the 1B
(`wire-in-r6-prefill-offload.md`, "Real projection shapes").

## Known blockers outside this work

- `load_f16_tensor` (and qwen35's `loading::hfq_plain_tensor_as_f32`) call
  `tensor_data`, not `tensor_data_vec`, so `Bf16Lut3`(49)/`Bf16Huff`(50) tensors panic.
  `--bf16-codec` defaults to `huff`, so every artifact carrying the codec is unloadable.
  `tensor_data_vec` already decodes via `decode_bf16_packed`.
- The 35B's oq4++ is oq4++ in name only on its experts: LDLQ landed 120/20600 because
  `qwen3.6-35b-a3b-128tok.calib.hfq` activates almost no experts. A router-aware or much
  longer calibration is needed before any MoE quant-quality claim.
- The 35B needs a per-model `max_seq` cap or the KV allocation OOMs at the configured
  262144.

## Prefill: a WRONG diagnosis, and what is actually known (2026-07-30)

**Retracted.** An earlier revision of this document claimed the prefill deficit was a
missing `Oq4G256` entry in `is_batchable_la`, i.e. that oq4 models fell back to a
per-token prefill loop. That was wrong on both halves and is corrected here so nobody
acts on it:

- **The batched OQ4 arm already exists.** `weights.rs::weight_gemm` has a full
  `DType::Oq4G256` arm (rotate via `rotate_x_mq_batched_for`, then
  `gemm_oq4_residual_mmq` for `batch >= 64`, else `gemm_oq4_grouped_act_batched`), with
  an `HIPFIRE_OQ4_PREFILL_ACT_BITS` override. `llama.rs::prefill_forward` calls it for
  every projection with `batch = n`.
- **`prefill_forward` is fully batched** — `rope_batched_f32`,
  `kv_cache_write_q8_0_batched`, batched attention, batched projections. There is no
  per-token loop.
- **`is_batchable_la` is not even consulted by llama** — only `qwen35/ep.rs` calls it.
- **The evidence was misread.** Flat t/s vs prompt length (84.0 @ pp128, 80.9 @ pp512) is
  what an *efficient* batched prefill looks like: total time scales with tokens, so
  tokens/s is constant. It is not a signature of sequential execution.

What IS established: prefill gains almost nothing over decode (pp512 80.8 t/s vs tg128
76.0 t/s on llama-3.2-1B oq4++), and **the projection GEMMs are not the cause** —
forcing `HIPFIRE_OQ4_PREFILL_ACT_BITS` to 4, 8 and 16 gives 80.70 / 80.90 / 80.80 t/s,
i.e. swapping the int4, int8-MMQ and f16-WMMA GEMM paths changes nothing measurable.

A controlled comparison settles the quant question. Same model, same
`--bf16-codec none`, only the weight format differs:

| artifact | pp512 t/s | tg128 t/s | prefill gain over decode |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-nc--mq4` | 107.75 | 100.14 | 1.076x |
| `Llama-3.2-1B-Instruct-nc--oq4++` | 80.80 | 75.94 | 1.064x |

Both land at ~7%. **Batched prefill is nearly ineffective on the llama path regardless of
weight format** — it is not an OQ4 problem and not a GEMM-selection problem. (Note also
that mq4 is simply the faster artifact here: 100.1 tok/s decode = 1.67x FastFlowLM,
against oq4++'s 1.26x.)

**Scaling analysis points at weight-read amortisation.** Prefill time is linear in prompt
length (llama-3.2-1B mq4, one rep each):

| pp | prefill_ms | ratio vs previous | ms/token |
|---|---|---|---|
| 128 | 1124.7 | - | 8.8 |
| 256 | 2277.8 | 2.03x | 8.9 |
| 512 | 4752.4 | 2.09x | 9.3 |
| 1024 | 10004.9 | 2.11x | 9.8 |

Every doubling of the prompt doubles the time, so per-token cost is ~constant at 9-10 ms
— the same per-token cost as decode. Attention's O(n^2) term shows up only as the slight
drift (2.03 -> 2.11) and is not the limiter at these lengths.

## PERF/WATT — measured, and it reverses this document's recommendation (2026-07-30)

Earlier revisions argued "the GPU already beats FLM on decode, so do not build an NPU
decode path." That compared throughput only and was wrong. Measured on halo, same model
(llama-3.2-1B), package power sampled from
`/sys/class/drm/card1/device/hwmon/hwmon5/power1_input` at 5 Hz (idle baseline 6.20 W):

| | tok/s | watts | tok/s/W | marginal tok/s/W (minus idle) |
|---|---|---|---|---|
| hipfire GPU, oq4++ | 75.66 | 57.98 | 1.30 | 1.46 |
| FastFlowLM, NPU | 59.48 | 21.83 | **2.72** | **3.81** |

**The NPU gives 79% of the throughput at 38% of the power: 2.09x better perf/watt, 2.60x
on marginal power.** Even at the measured aie2p aggregate feed ceiling (~55 GB/s, below),
an NPU decode path would land near ~65 tok/s at ~22 W = ~2.95 tok/s/W — still >2x the GPU.
The feed ceiling caps NPU *speed*, not its efficiency advantage.

So an NPU decode path IS worth building; it just should not be justified or judged on
tok/s. For a thermally- or battery-constrained target the NPU is the correct engine, and
"beat FLM" should be read as beating 2.72 tok/s/W, not 59.5 tok/s.

## ARCH HYGIENE: aie2 vs aie2p — do not mix the numbers

`docs/npu/NPU-RESULTS.md`'s "Platform" section is **nix1 / NPU1 Phoenix / AIE2-AIE-ML /
16 TOPS / 4 compute columns** — a different machine AND a different architecture. This
host (halo) is **aie2p / npu2, 8 columns, 58 TOPS**. Numbers do not transfer:

- `benchmarks/npu_gemm_tuning/r1/README.md` is halo aie2p (line 51 states it).
- `r1b_trace_run.py` defaults `DEV=npu2`, i.e. aie2p — verify `NPU_DEV` before citing a run.
- The 0.4-0.6 TOPS W4A8 figures in `r6/README.md` are XDNA1/Phoenix and must NOT be
  compared against halo results.

## Measured aie2p feed ceiling (trace unit, `r1b_cols_trace_run.py`)

| COLS | AGG GB/s | per-col | MEAN_BUSY |
|---|---|---|---|
| 1 | 13.4 | 13.4 | 0.93 |
| 2 | 25.8 | 12.9 | 0.90 |
| 4 | 44-45 | 11.0-11.3 | 0.77-0.80 |
| 8 | 54-56 | 7.0 | 0.47-0.49 |

Aggregate saturates at **~55 GB/s** — the shared LPDDR5X/NoC/mem-controller knee, not a
per-port limit (one shim DMA channel is ~40 GB/s at 32 B/cyc; the 14.4 GB/s I traced is a
single compute tile's S2MM receive port at 8 B/cyc). r1's own conclusion: prefill with
M >= ~512 is compute-bound, and only decode-shaped work stays feed-bound.

## NPU feed ceiling, measured with the AIE trace unit (2026-07-30)

Every NPU number I reported earlier came from wall-clock timing in my own harnesses, which
cannot separate host overhead from on-NPU behaviour. The repo already has the right
instrument: `benchmarks/npu_gemm_tuning/r1/r1b_trace_run.py` traces the compute tile's
S2MM ch0 port with `PORT_RUNNING/STALLED/IDLE`. Run it as:

```bash
source ~/.venv/bin/activate && source /opt/xilinx/xrt/setup.sh   # the DefaultNPURuntime
export MLIR_AIE_INC=$(python -c 'import mlir_aie;print(list(mlir_aie.__path__)[0])')/include
python benchmarks/npu_gemm_tuning/r1/r1b_trace_run.py
```

Live on halo it reproduces the recorded/"SEALED" result exactly:

```
BUSY_FRAC 0.914  RUNCYC 131072  STALLCYC 216  IDLECYC 12169
FEED_GBS_SPAN 13.1568  FEED_GBS_RUN 14.4000
```

**One compute tile's receive DMA is capped at 8 B/cycle = 14.4 GB/s**, 91% busy with
negligible stalling — bandwidth-bound at the port, not starved upstream.

Calibrating my own measurements against it:

| | rate | vs per-port cap (8 tiles) | vs ~55 GB/s fabric |
|---|---|---|---|
| this doc's NPU decode best (COLS=8) | 32.5 GB/s (4.06/tile) | 28% | 59% |
| FastFlowLM (measured 46.4 GB/s) | 46.4 GB/s | 40% | 84% |
| hipfire GPU oq4++ / mq4 | 48.4 / 63.8 GB/s | — | above it |

**Conclusion for decode: do NOT move it to the NPU.** The GPU already runs at or above the
NPU's practical aggregate feed rate, so an NPU decode path is capped below where the GPU
already is. FLM's kernel does feed better than the R6 path (46.4 vs 32.5) but that only
closes a gap to a ceiling the GPU has already cleared.

The NPU case therefore rests entirely on **compute-bound prefill**, where the relevant
figures are the 15.7 TOPS int8 reference in `findings.md` against the ~6.8 TFLOP/s that
FLM's 2750 t/s prefill implies — not on bandwidth.

## MEASURED: the prefill path dispatches no batched GEMM at all (2026-07-30)

rocprofv3 over a real ~540-token prompt through a real oq4++ artifact
(`examples/infer_hfq`, in-process so the profiler can see it):

| kernel | dispatches | total ms | avg us | LDS |
|---|---|---|---|---|
| `gemv_oq4_grouped` | 85008 | 4162.4 | 49.0 | 0 |
| `gemv_f32` | 759 | 3960.4 | 5217.9 | 1024 |
| `attention_q8_0_kv` | 12144 | 558.2 | 46.0 | 2692 |
| `rotate_x_mq_awq` | 85008 | 148.1 | 1.7 | 0 |
| `rmsnorm_f32_gfx1151` | 25047 | 147.2 | 5.9 | 32 |

**No `gemm_*` kernel appears at all** — not `gemm_oq4_residual_mmq`, not any batched GEMM.
85008 = 112 GEMVs (7 projections x 16 layers) x 759 forward steps, and `gemv_f32` is
dispatched exactly 759 times. The prompt was processed one position at a time through the
DECODE path.

That means `prefill_forward` — which is genuinely batched — is not being called on this
path, and the earlier hypothesis of per-token prefill was right in substance even though
the `is_batchable_la` mechanism proposed for it was wrong.

**Second finding: the lm_head dominates.** `gemv_f32` averages 5.2 ms per dispatch over
759 dispatches = 3960 ms, **43% of total runtime**. That is the [128256, 2048] f32 lm_head
evaluated at EVERY prompt position, when prefill needs logits only at the last one. Even
after the per-position issue is fixed, skipping lm_head for non-final prompt positions is
a large independent win. (FLM ships a separate `lm_head.xclbin` and hipfire has a
two-stage coarse lm_head — `#201` — for exactly this cost.)

**Caveat:** this profiles `examples/infer_hfq`, not the daemon. The daemon's measured
pp512 ~= tg128 is what per-position GEMV predicts, so the same shape of problem is likely
present in serving, but confirming it requires profiling the daemon through its JSON-lines
protocol (see the `run-model` skill). Do that before changing serving code.

## Why `bench_batched_gemm` plateaus at 1.1x — and why that does NOT explain the model

**Scope warning, read first.** What follows explains the microbenchmark
`bench_batched_gemm`, which exercises `gemm_hfq4g256`. It does NOT explain the served
model's prefill: an oq4/oq4++ model at batch>=64 dispatches
`gemm_oq4_residual_mmq` instead, and that kernel IS a properly tiled llama.cpp-style MMQ
with `__shared__` tiles for both operands (`tile_x`, `tile_y`, `MMQ_TILE_X_K`). So the
activation-re-read story below is real for the plain HFQ4 kernel and irrelevant to the
model-level 80.8 t/s. The model-level cause remains UNKNOWN and needs the daemon profiled
directly, not a microbenchmark of a different kernel.

`kernels/src/gemm_hfq4g256.hip` is activation-bound, and structurally so.

The kernel assigns one output row per block (`row = blockIdx.x`) and tiles the batch at
`#define BATCH_TILE 8`, grid `(M, ceil(batch/BATCH_TILE))`. Weights for a K-group are
unpacked once and reused across the 8 batch rows — that part is fine. The problem is the
activation side: the innermost loop reads `x` straight from global memory,

```c
for (int b = 0; b < local_bs; b++) {
    const float* xb = x + (batch_start + b) * K;
    acc[b] += w0 * xb[base_k] + w1 * xb[base_k + 1] + ... ;
}
```

with **no `__shared__` staging at all** (confirmed by rocprofv3: `group_segment_size = 0`
for `gemm_hfq4g256`). So every one of the `M` row-blocks re-reads the whole `batch x K`
activation slab. For the 8B FFN shape in `bench_batched_gemm` (M=12288, K=4096, batch=32):

| | bytes per GEMM |
|---|---|
| weights | 25.2 MB |
| activations, ACTUAL (`M*K*batch*4`) | **6.44 GB** |
| activations, per `profile.rs::gemm_hfq4g256_bytes` (`batch*(K+M)*4`) | 2.1 MB |

Activation traffic is **3072x** what the repo's own traffic model assumes and **255x** the
weight traffic. The GEMM is activation-bound, not weight-bound — which is why
`profile.rs`'s "weight read once, B input/output vectors" comment does not describe the
kernel's real behaviour.

**This explains the microbenchmark's 1.1x plateau (only).** Per-vector activation traffic is `M*K*4`,
*independent of batch size*, so per-vector cost cannot fall as the batch grows:

```
  GEMV (1 vector):  40.9us
  batch= 1:   77.2us/vec       batch= 8:   38.8us/vec   <- plateau at BATCH_TILE
  batch= 4:   43.8us/vec       batch=32:   37.6us/vec
```

It plateaus exactly at `BATCH_TILE=8` and never improves, because raising the batch adds
activation traffic in proportion. It does NOT carry over to the served model, whose OQ4 path uses the tiled MMQ kernel.

(The reason it is not catastrophically slow is that the re-reads hit cache: 6.44 GB in
1.2 ms is ~5.4 TB/s, far above DRAM, so the `batch x K` slab — 512 KB here — is L2-resident.
Cache saves it from disaster but the traffic still sets the rate.)

Fixing `gemm_hfq4g256` (stage the activation tile in LDS, as
`gemm_oq4_residual_mmq` already does) would help any path that still selects it — but it is
NOT the fix for the measured model prefill, because that path already uses the tiled
kernel.


## The per-linear decode floor, measured (2026-07-30)

`crates/hipfire-runtime/examples/npu_linear_oq4.rs` runs one real `.hfq` linear
through `NpuOpusExecutor::run_f32` and checks it against
`OpusPackedMatrix::reference_f32`. llama-3.2-1B `layers.0.self_attn.q_proj`
(K=2048 N=2048, oq4++), **0/2048 mismatches, max_abs=0.0** — the first correct
oq4 dequantization on the NPU from a real artifact.

Decode-shape cost, same tensor at M=1:

| path | ms | weight GB/s |
|---|---|---|
| per-group (8 dispatches) | 1.32 | 1.58 |
| full-K m64/COLS=8 | 0.83 | 2.53 |
| full-K m8/COLS=1 | **0.53** | 3.99 |

### Where the 0.53 ms goes

`HIPFIRE_XDNA_TRACE=1`, per dispatch: **submit 21.5 us (4%), wait 392 us (74%),
host ~126 us (23%)**. It is kernel execution, not submit overhead. The work does
not scale with rows either — M=1 0.529 ms, M=2 0.561, M=4 0.587, M=8 0.639,
i.e. ~0.016 ms per real row on top of ~0.51 ms fixed.

### Widening the array makes decode WORSE

Each cache built at its minimum legal M (`M % (COLS*8) == 0`):

| COLS / cache M | ms @ M=1 |
|---|---|
| 1 / 8 | 0.545 |
| 2 / 16 | 0.593 |
| 4 / 32 | 0.671 |
| 8 / 64 | 0.830 |

The unscaled full-K path reads back `groups * chunk_rows * N` i32 where
`chunk_rows` is the cache's M. Widening the array raises the minimum M, so it
only enlarges a readback of padded rows decode never uses: 8 groups x 64 rows x
2048 x 4 = 4.2 MB to compute one row.

### Conclusion

The per-linear kernel family floors at ~0.53 ms/linear at decode = 112 linears
= ~59 ms/token = **~17 tok/s against FastFlowLM's 60.1**, which needs
~0.15 ms/linear. No COLS/M/kernel-variant knob closes that gap. **The fused
decoder layer is required, not merely preferable** — it is the only structure
that amortises the ~0.51 ms fixed cost across a layer's seven linears
(~0.075 ms/linear effective => ~119 tok/s).

### Blocked: scaled output

`r6_fullk_cache.sh w4-scaled` returns f32 scaled on the array — `chunk_rows*N`
f32 instead of `groups*chunk_rows*N` i32, i.e. 8x less readback, and it would
make wide COLS pay off. `pack_matrix` already feeds it the per-group weight
scales, and it dispatches, but the output is wrong (6.756 vs 1.449 at index 0).
Why this was not caught earlier: `examples/npu_embeddinggemma_fullk_sweep.rs`
exercises `run_resident_scaled` with all-1.0 activation AND weight scales, so it
is a throughput probe that cannot detect a scale-application bug. The scaled
full-K path has no correctness user in the repo.

### Cache recipe traps

Three ways to build a cache that loads, dispatches, and returns plausible noise,
none of which fail loudly:
- W4 per-group built from the default `r6_gemm.cc` instead of
  `R6_KERNEL_SRC=r6_gemm_ts.cc`
- MT=16 instead of MT=4
- N-parallel `r6_gen.py` instead of `R6_GEN=r6_gen_mp.py`
Copy `benchmarks/npu_gemm_tuning/embeddinggemma_aie2p/build_opus_caches.sh`.
Also: the r118/r129 "staged" full-K kernel belongs to `NpuGemmStagedFullK`, not
`NpuGemmFullK` — pairing them builds and runs and returns garbage. And `GROUPS`
is a bash builtin (the caller's gid array), so assigning it in a build script
silently yields 1000.

### Scaled full-K: alignment is implicated, but padding alone does not fix it

Tested 2026-07-30. `r6_scale_accum.cc` reads
`weight_scales = activation_scales + ROWS` with `ROWS = MT*4 = M/COLS`. At
M=64/COLS=8 that is ROWS=8 floats = a **32-byte** offset, while
`aie::load_v<16>` on floats is a 64-byte vector load wanting a 64-byte aligned
address. That would be invisible in exactly one case — a uniform-1.0 payload,
where any misaligned read still yields 1.0 — which is the one case that passes.

Padding the activation-scale region to 16 floats (updating `ROWS_PADDED` in the
kernel, `SE` in `r6_gen_mp_fullk_scaled.py`, and `scale_bytes` /
`copy_scale_payload` / the `input_bytes` allocation in `gemm_fullk.rs`) CHANGES
the result — the weight-only case moved from -9.0 to -4.5 against a want of
385.5 — so the payload layout is genuinely implicated. But it does not correct
it, and the remaining error cannot be explained by group assignment or scale
indexing either, because every failing case uses UNIFORM scales under which any
permutation or mis-assignment is unobservable.

REVERTED, deliberately. `copy_scale_payload` and the `SE` contract are shared
with the EmbeddingGemma whole-scaled/full-K paths, whose cached xclbins were
built against the old size; landing a half-migrated scale ABI risks breaking a
working consumer to chase a broken one. Reproduce with
`examples/npu_fullk_scaled_bug` before changing it again, and re-run the
EmbeddingGemma sweeps after — noting that those sweeps use all-1.0 scales and
therefore cannot detect a regression in this exact area.

## qwen3.6-35B-A3B on the NPU: what the llama hook does NOT cover (2026-07-30)

Measured, not inferred. Running the 35B with `HIPFIRE_NPU_DECODE=1` and the
residency probe reports **zero weights made NPU-resident**, and decode is
34.90 tok/s against a 34.70 tok/s GPU baseline — i.e. noise, no offload.

Two independent reasons, both structural:

1. **The MoE hot path bypasses `weight_gemv`.** The `npu_linear` hook sits on
   `weights::weight_gemv`, which qwen35 does use (29 sites in
   `decode_layers.rs`, 12 in `moe_decode.rs`) — but the individual
   `weight_gemv` calls in `moe_decode.rs` are the MIXED-DTYPE FALLBACK. A
   uniform-dtype MoE layer, which is what this artifact is, runs the fused
   gate-side GEMV plus the indexed `gemv_oq4g256_moe_*` routed kernels. Those
   are where essentially all the FLOPs are, and they never reach the hook.
2. **Attention is not oq4.** Layer 0's `self_attn` / `linear_attn` tensors are
   `qt=3` (Q8F16 -> `DType::Q8_0`), not `Oq4G256`, so they are rejected on dtype
   before any cache lookup. Only the shared expert and the routed experts are
   qt=34.

So NPU support for the MoE is not an extension of the llama path; it needs
(a) an offload hook inside the MoE dispatch itself — the routed-expert indexed
GEMV is the target, one expert-group GEMM per active expert — and (b) either a
Q8_0 NPU path for attention or an oq4-quantised attention artifact. The
`r6_fullk_cache.sh w8` mode exists for (b) but has not been exercised here.

Cache shapes that WOULD be needed for the shared expert alone (built, unused
until a hook exists): K=2048 N=512 (gate/up) and K=512 N=2048 (down).

### Scaled full-K, localised: delivery is correct, the ARITHMETIC is wrong

`benchmarks/npu_gemm_tuning/r6/r6_scale_dump_probe.cc` replaces the scale math
with a copy of the received `scale_payload` into the output, so the host can see
exactly what the core got. Build a cache with it in place of
`r6_scale_accum.cc` and run `npu_fullk_scaled_bug` with
`HIPFIRE_DUMP_PAYLOAD=1`:

| case | activation scales received | weight scales received |
|---|---|---|
| all 1.0 | `[1.0 x8]` | `[1.0 x8]` |
| weight 0.5 uniform | `[1.0 x8]` | `[0.5 x8]` |
| activation 0.25 uniform | `[0.25 x8]` | `[1.0 x8]` |

Exactly what `copy_scale_payload` wrote, at the expected offsets. So:

- **NOT** a routing fault — the core receives the right bytes.
- **NOT** a payload-layout or offset fault — `weight_scales =
  activation_scales + ROWS` reads the correct values.
- **NOT** a delivery-alignment fault — scalar reads at those offsets return the
  written data. (An earlier padding experiment changed the result, which had
  suggested alignment; this supersedes it. Padding perturbs the arithmetic path,
  not the delivery.)
- **NOT** the accumulate — the all-1.0 case sums all 8 groups exactly (771).

What remains is the float math in `scale_impl`: `aie::to_float(integers)`,
`aie::mul(values, weight_scale)`, `aie::mul(scaled, row_scale)`, and their
`.to_vector<float>()` conversions. Multiplying by 1.0 gives the exact answer;
multiplying by 0.5 or 0.25 does not, and not by a constant factor (the sign
flips). That points at the accumulator/vector-conversion semantics of the AIE
API rather than at anything in this repo's plumbing, and is the next thing to
check against the aie_api documentation for aie2p.

Two failed approaches, recorded so they are not repeated: replacing the whole
body with scalar C++ produced all zeros, and replacing only the weight-scale
`aie::load_v<16>` with an `aie::broadcast` of one scalar produced infinities —
both broke the previously-exact all-1.0 case, so neither isolated anything.

### ROOT CAUSE: the weight-scale load reads the ACTIVATION-scale region

`r6_scale_stage_probe.cc` parks the multiply intermediates in the output buffer
(init pass only, with `r6_scale_accum` made inert so later groups cannot
accumulate over them) and `npu_fullk_scaled_bug --HIPFIRE_DUMP_STAGE=1` prints
them. Lanes 0..3 of the first block:

| case | act scale | weight scale | value LOADED as weight_scale |
|---|---|---|---|
| all 1.0 | 1.0 | 1.0 | 1.0 |
| weight 0.5 uniform | 1.0 | 0.5 | **1.0** |
| activation 0.25 uniform | 0.25 | 1.0 | **0.25** |

In both failing cases the vector loaded as the weight scale carries the
ACTIVATION scale's value. `aie::load_v<16>(weight_scales + block*16)` is
reading inside the activation-scale block rather than past it.

This explains every observation at once, including why the all-1.0 case is
exact: when the whole payload is 1.0, reading the wrong region still yields 1.0.
It is consistent with the earlier padding experiment changing the result
(padding moves the boundary) without fixing it (the load is still short), and
with delivery being provably correct (the bytes are there; the load addresses
the wrong ones).

`weight_scales = activation_scales + ROWS` with ROWS = MT*4 = M/COLS = 8 floats
should be correct C pointer arithmetic, so the next step is to print ROWS and
the byte delta `(const int8*)weight_scales - scale_payload` from inside the
kernel and compare against the expected 32. Candidates: MT not reaching this
translation unit (the header defaults `#define MT 8`, which would make ROWS=32),
or the 64-byte `load_v<16>` being lowered to an aligned load that truncates the
32-byte-aligned address downward.

The second candidate would also explain why padding to 16 floats moved the
answer without fixing it: it makes the address 64-byte aligned but the earlier
`ROWS` offset is still what the pointer arithmetic used.

### Fault 1 CONFIRMED and its fix verified; a SECOND fault remains

Applying the padding (activation-scale region rounded to 16 floats, in
`r6_scale_accum.cc`'s `ROWS_PADDED`, `SE` in `r6_gen_mp_fullk_scaled.py`, and
`scale_bytes` / `copy_scale_payload` / `input_bytes` in `gemm_fullk.rs`) and
re-running the STAGE probe shows the load is repaired:

| case | weight_scale loaded, before | after padding |
|---|---|---|
| weight 0.5 uniform | 1.0 (wrong — an activation scale) | **0.5** |
| activation 0.25 uniform | 0.25 (wrong) | **1.0** |

and `after_mul1` becomes 11.0 x 0.5 = 5.5, correct. So the 64-byte
`aie::load_v<16>` truncating a 32-byte-aligned address is real, and padding
fixes it.

BUT the end-to-end result is STILL wrong with padding applied: weight-only reads
-4.5 against a want of 385.5. The stage probe runs with the accumulate entry
point inert, so it only proves group 0's first multiply. Everything downstream of
that is still suspect, and by elimination the remaining fault is in the
**cross-group accumulate** — `aie::add(scaled, aie::load_v<16>(output + offset))`
over groups 1..7 — or in the second multiply by `row_scale`. Note the all-1.0
case accumulates all 8 groups exactly, so the accumulate is not broken
unconditionally.

REVERTED again, for the same reason as before: this is a shared ABI and landing
a half-fix is worse than landing none.

**Implication worth checking independently of this work:** if any production
EmbeddingGemma path runs the scaled full-K or whole-scaled kernels with REAL
(non-unit) scales, it is silently wrong today — fault 1 alone corrupts every
weight-scale lane. `npu_opus.rs` does call `run_f32` with real matrices. The
sweeps cannot detect it because they pass all-1.0 scales.

### Fault 2 localised: groups 1..7 mis-pair partials with scales

Two further probes, both with the fault-1 padding applied:

- **No-add probe** (accumulate overwrites instead of adding, so the output holds
  only the LAST group): all-1.0 gives 102.0, exactly the last group's
  contribution — correct. Weight-0.5 gives -4.5 where the last group alone
  should give 51.0.
- **Accum-payload dump** (dump the payload the ACCUMULATE pass receives; the
  earlier dump covered only the init pass, i.e. group 0): groups 1..7 receive
  the correct activation and weight scales at the padded offsets.

So for groups 1..7: the scales arrive correctly, the load is fixed, and yet the
product is wrong — while group 0's product is right (stage probe) and the
all-1.0 case is right for every group. Uniform scales make a wrong scale VALUE
unobservable, so the natural suspect was the PAIRING between each group's
r6_mac partial on `@fr` and its scale payload on `@fs`.

**That hypothesis is eliminated by reading the generator.** The GEMM core
produces `@fr` as `for slab { for group { acquire/produce } }`, and the scale
core consumes it as `for slab { init(group 0); for group 1..KGROUPS-1 {...} }` —
one `@fr` and one `@fs` per (slab, group), in the same nesting order. Both
streams therefore advance in lockstep by construction. In the COMBINED (COLS=8)
layout they even originate from a single host stream that
`objectfifo.link [@fx] -> [@fa, @fs] ([] [0, AW])` splits, so they cannot skew.

So fault 2 is NOT mis-pairing and NOT scale delivery (dumped correct for the
accumulate groups). What is left is the accumulate arithmetic itself under
non-unit scales, or the reuse of the single `@fc` output buffer across the
init and accumulate calls within a slab. That is the next thing to probe.

Note this is invisible in the all-1.0 case for the same reason as fault 1: every
group's payload is identical, so mis-pairing changes nothing.


## THE SCALED FULL-K BUG IS DISPATCH DESYNCHRONISATION, NOT SCALES

Everything above treats "all-1.0 passes, non-unit scales fail" as the signal.
**That framing is wrong**, and `HIPFIRE_CASE_ORDER=reverse` in
`npu_fullk_scaled_bug` proves it: reversing the order of the three cases moves
which one passes.

| order | first case run | all-1.0 result |
|---|---|---|
| normal | all-1.0 | mostly correct (mean_ratio 0.875) |
| reversed | activation-only | **fails**, got -9.0, mean_ratio -0.027 |

All-1.0 is not special. The FIRST `run_resident_scaled` of a process is
(mostly) correct and every later one is corrupted. The scale-core and GEMM-core
loops are `scf.for %i = 0 to INF` over their objectfifos, so core state persists
across dispatches; a dispatch that does not consume exactly what the previous one
left desynchronises the `@fr`/`@fs`/`@fc` streams for every subsequent call.

This retroactively explains the whole investigation:
- the accumulate-count probe seeing 7, 1 and 4 accumulates for three cases that
  differ only in scale VALUES (which that probe ignored entirely);
- padding the activation region "changing" the result without fixing it — it
  changes buffer sizes, hence the desync pattern;
- why fault 1 (the genuinely misaligned weight-scale load, confirmed by direct
  observation and fixed by padding) did not repair the end-to-end result.

Fault 1 is real and its fix is verified. But it is not sufficient, and the
remaining fault is not in the arithmetic at all — it is that
`run_resident_scaled` is not safe to call more than once. Any consumer calling
it per layer or per token would be silently wrong from the second call onward.

NEXT: check whether the runtime sequence drains/resets the fifos per dispatch,
and compare against the UNSCALED full-K path (`r6_gen_mp_fullk.py`), which is
repeatedly called by `npu_linear` in the working llama path and does NOT show
this — so whatever it does differently is the fix.

### Generator diff: the dispatch pattern is the SAME, so the desync is in the link

Comparing `r6_gen_mp_fullk.py` (unscaled, called ~12k times per llama run by
`npu_linear` without desyncing) against `r6_gen_mp_fullk_scaled.py`:

Both runtime sequences follow the same shape — configure and start the input
task(s) and `@fw`, configure `@fc` with `issue_token`, `dma_await_task` on `@fc`
only, then free. Neither awaits its INPUT task before freeing it. So the
start/await/free pattern is not the difference.

What the scaled schedule adds is a second consumer: under COLS=8 one host
stream `@fx` is split by `objectfifo.link [@fx] -> [@fa, @fs] ([] [0, AW])` into
the GEMM core's activations and the scale core's payload, and a second core
(`%s{col}`) with its own `@fr`/`@fs`/`@fc` acquire pattern (`@fc` once per slab,
`@fr`/`@fs` once per group). Element counts balance per dispatch — NB slabs,
NB*KGROUPS groups, `@fc` repeat_count = NB-1 — so the imbalance is not in the
counts.

That leaves the link's buffering, or the two consumers of `@fx` draining at
different rates and leaving residue at dispatch end. Worth testing directly:
call `run_resident_scaled` twice with IDENTICAL inputs and compare — the
reversal experiment implies call 2 differs from call 1.

## MEASURED: the scaled path is a PREFILL lever, not a decode lever

The reason to chase `w4-scaled` was "on-array scaling removes the i32 readback,
which is what forces COLS=1, which caps decode". **Measured, that is wrong.**
q_proj K=2048 N=2048, per call (scaled timings taken on the first-dispatch-
correct configuration; the desync affects values, not latency):

| config | M=1 (decode) | M=64 (prefill) |
|---|---|---|
| unscaled COLS=1 m8 | **0.529 ms** | — |
| unscaled COLS=8 m64 | 0.884 | 2.135 |
| scaled COLS=8 m64 | 1.329 | **1.467** |

At decode the scaled path is **2.5x SLOWER** than the current best. The readback
argument does not apply at M=1: unscaled COLS=1/m8 reads back
8 groups x 8 rows x 2048 x 4 = 512 KB, and scaled COLS=8/m64 reads back
64 rows x 2048 x 4 = 512 KB. Identical. There was never a decode win to capture
— the win only appears once M is large enough that the i32 block
(groups x M x N) outgrows the f32 block (M x N), i.e. at prefill, where scaled
COLS=8 does beat unscaled COLS=8 by 1.46x.

CONSEQUENCE: repairing the scaled full-K desync is worth doing for PREFILL and
is irrelevant to the decode number this project has been trying to move. The
decode lever remains the fused decoder layer (amortising the ~0.51 ms fixed cost
across a layer's seven linears); see the per-linear floor section above.

Recorded because roughly eight turns of investigation were spent on this path
under the assumption it gated decode. Measure the lever before repairing it.

### UG1079's "eight lanes" is AIE1/AIE-ML, NOT aie2p — do not rewrite to 8-lane

`docs/npu/ug1079-2026.1-AIE-programming-manual/036-floating-point-operations.md`
states: "The AI Engine vector unit provides eight lanes of single-precision
floating-point multiplication and accumulation." That reads like an explanation
for fault 1 — `r6_scale_accum.cc` uses 16-lane float vectors, so `load_v<16>`
is a 64-byte load and the weight scales at `activation_scales + ROWS` = +32
bytes are misaligned. Rewriting the kernel to native 8-lane ops would then make
the existing offset naturally aligned and need no ABI change.

**Tested: it produces infinities.** The reason is in the aie_api headers —
`detail/aie2p/mul_acc32_fp.hpp` dispatches `mul_elem_16` for `Elems <= 16` on
fp32, i.e. aie2p's native single-precision vector width is **16 lanes**. The
UG1079 text describes AIE1 / AIE-ML (NPU1), not AIE-ML v2 / aie2p (NPU2).

This is the same generation trap already recorded for benchmark numbers
("never mix aie2 with aie2p") applied to the ISA documentation. When UG1079 and
`include/aie_api/detail/aie2p/` disagree, the headers win for this target.

So fault 1's fix remains the payload padding, which was independently proven to
repair the load by direct observation. Use
`~/AMD_AI_DOCS/install-help-aie-ml-v2-intrinsics.sh` (AIE-ML **v2** = aie2p) for
intrinsic-level questions rather than UG1079.

### objectfifo semantics relevant to the desync (from the MLIR-AIE bindings)

`_aie_ops_gen.py` docstrings, which are the authoritative local reference:

- **`aie.objectfifo` is a circular buffer plus LOCKS**, lowered by
  `AIEObjectFifoStatefulTransformPass` into `aie.buffers` + `aie.locks` in the
  memory module. Depth = number of objects. Every fifo in
  `r6_gen_mp_fullk_scaled.py` is created with depth **1**, forcing strict
  producer/consumer alternation.
- **`objectfifo.acquire` elides redundant acquires**: "the operation only
  performs new acquires if necessary… if two objects have been acquired in the
  past and none have yet to be released by the same process, performing another
  acquire… will not result in any new use_lock operations." So the lowering
  tracks per-process held-object state.

Locks live in the memory module and are configured once, not per dispatch. A
dispatch that does not leave every lock back at its initial state desynchronises
every later one — which matches the observed "only the first
`run_resident_scaled` of a process is correct".

Tested and REJECTED: raising `@fs`/`@fr` to depth 2 to tolerate skew. It builds
but returns infinities — the `objectfifo.link [@fx] -> [@fa, @fs]` split and the
runtime DMA sizing both assume matched depth-1 buffering, so depth is not a
drop-in knob.

### TRAP: full-K cache directory names omit COLS

`r6_fullk_cache.sh` names its output
`embgemma_aie2p_fullk_submit_{mode}_m{M}_kg{K/256}_n{N}` — **COLS is not in the
name**. Building the same (mode, M, K, N) at a different COLS silently
OVERWRITES the existing cache, and `NpuGemmFullK::load_cached(dir, cols)` takes
cols as a caller argument, so the mismatch is not detected: it loads a cache
built for one column count and runs it as another, producing wrong numbers with
no error.

This invalidated a COLS sweep here (building COLS=2/M=8 clobbered the COLS=1/M=8
cache that the end-to-end decode path in `npu_linear.rs` depends on, which then
failed parity). Earlier sweeps that varied M alongside COLS — (1,8), (2,16),
(4,32), (8,64) — were unaffected because their M values differ.

If you need to compare COLS at a FIXED M, give each build a distinct cache
directory, or add COLS to the name. Re-verify with
`examples/npu_linear_oq4 ... --fullk COLS --fullk-m M` (it checks parity, so a
mismatched cache shows up immediately).

## MLIR-AIE source (~/build/mlir-aie) answers the desync question

### Locks are configured once, not per dispatch

`lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` creates the
objectfifo locks with STATIC initial values:

```cpp
int prodLockValue = (numElem - initValues) * repeatCount;
int consLockValue = initValues * repeatCount;
```

These are `aie.lock` init values baked into the design and applied at device
configuration (xclbin load / hardware-context creation) — **not per dispatch**.
So any dispatch that does not return every lock to its initial value skews every
later one, which is exactly the observed "only the first `run_resident_scaled`
of a process is correct".

### depth > 1 with repeat_count is UNSUPPORTED — my depth-2 test was invalid

`programming_guide/section-2/section-2b/04_Repeat/README.md`: with a compute-tile
producer the lowering adjusts acquire/release values to account for the repeat,
and "Doing this adjustement for Object FIFOs of depth larger than 1 is
non-trivial and **currently not supported**." The scaled schedule gives `@fc`
(produced by the scale core, a compute tile) `repeat_count = NB - 1`, so depth
MUST stay 1. The earlier depth-2 experiment was invalid by construction.

### The likely mechanism: freeing input BDs without awaiting them

`include/aie/Dialect/AIEX/IR/AIEX.td` and
`programming_guide/section-2/section-2d/DMATasks.md`:

> `dma_free_task` allows the compiler to reuse the BDs of a task WITHOUT
> synchronization. Using `dma_free_task(X)` before task `X` has completed will
> lead to a race condition and unpredictable behavior. Only use it in
> conjunction with some other means of synchronization — for example after
> `dma_await_task(Y)` if `Y` can only complete after `X`.

The scaled runtime sequence configures its input stream `%tx{col}` WITHOUT
`issue_token` (so it cannot be awaited), awaits only `%tc{col}`, then frees
`%tx`/`%tw`. That is only sound if awaiting `@fc` implies the input drained —
and `%tc` carries `repeat_count = NB - 1`, so one await may return after the
first of NB transfers rather than all of them, leaving input BDs live when they
are freed.

NEXT: give `%tx` `issue_token = true` and await it before freeing, or await
`%tc` once per repeat. Both are runtime-sequence changes in
`r6_gen_mp_fullk_scaled.py`, no kernel or ABI change.

### Scaled output does NOT help decode — now tested across the corner

| config | ms @ M=1 |
|---|---|
| unscaled COLS=1 m8 | **0.514** |
| scaled COLS=4 m16 | 0.833 |
| scaled COLS=8 m32 | 0.790 |
| scaled COLS=8 m64 | 1.329 |

Confirms the earlier single-point result: repairing the scaled path is a PREFILL
win (1.46x at M=64) and does nothing for decode at any COLS/M tested.

### Tested and REJECTED: issue_token + await on the scaled input stream

The docs-motivated fix — give `%tx{col}` `issue_token = true` and
`dma_await_task` it before `dma_free_task` — was applied to a sandbox copy of
`r6_gen_mp_fullk_scaled.py`. The generated MLIR is correct (all eight `%tx`
blocks carry `{issue_token = true}`, `%tc` keeps
`{issue_token = true, repeat_count = 31}`), it builds, and it returns
**infinities in every case including the first** — i.e. it breaks the schedule
outright rather than repairing the cross-dispatch desync.

So the "free before await" reading is not the whole story, or a token on an MM2S
input channel is not compatible with this schedule's BD allocation. The other
candidate from the same docs — awaiting `@fc` once per repeat rather than once
per dispatch — is untested.

Running tally of REJECTED fixes for the scaled path, all rebuilt and measured:
padding alone (fixes the load, end-to-end still wrong), 8-lane float rewrite
(wrong generation — aie2p fp32 is 16-lane), objectfifo depth 2 (unsupported with
repeat_count on a compute-tile producer), issue_token+await on the input (breaks
the schedule).
