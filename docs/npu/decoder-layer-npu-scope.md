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

