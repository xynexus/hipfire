# DFlash / DSpark drafter on the NPU — implementation plan

Phased, gated build plan for running the DFlash block-diffusion speculator body
(and the DSpark head variant) on the AIE2 NPU (npu1 / Phoenix / gfx1103, nix1).

Companion docs:
- `docs/npu/dflash-drafter-npu-kernels.md` — the kernel design (op→kernel map).
- `docs/npu/npu-kernel-design-guide.md` rule 11 — why it is viable (verify budget).
- `tools/npu/aiecost/` — cost model; `design.py drafter` reproduces the numbers.

## 0. Goal, scope, and definition of done

**Goal.** Run the DFlash draft block forward on the NPU, fast enough to hide under
the target's GPU verify, validated numerically against the Rust reference; then add
the DSpark heads.

**Primary target (decided).** The real trained z-lab drafter: 5-layer Qwen3 block,
dim 4096, FFN 12288, GQA 32 q / 8 kv × 128, block_size 16, 8-bit (Q8F16) weights.
Chosen over the tiny greenfield config because it (a) hides under real verify with
3.5–20× margin, (b) has trained weights for end-to-end validation, (c) maps onto
the existing Qwen3 NPU kernels (8 kv / 128 head_dim). The tiny config is a
stretch/warm-up datapoint, not the deliverable.

**Non-goals (this plan).** Runtime/daemon integration of an NPU execution backend;
int4 (OQ4/MQ4) re-quant of the sidecar; npu2/Strix-Halo port. All are follow-ons.

**Done when.** The fused NPU draft block forward matches `hipfire-runtime::dflash`
`draft_forward` (F32 GPU reference) per block position within the per-kernel test
tolerance, on a staged z-lab sidecar, at ≤ the model's fused-per-layer dispatch
count; DSpark heads reproduce the confidence/markov outputs; both are committed
with tests under `tools/npu/`.

## 1. Success metrics (from the cost model + measurement)

- **Latency.** Fused block ≤ verify budget. Real body model: ~16 ms/block int8-ish
  on npu1; measured 9B verify budget 57 ms (27B ~155, 31B ~345). Target: stay well
  under 57 ms so it hides even on the fastest (9B) target.
- **Fusion.** ≤ ~3 dispatches/layer (≈15/block), vs ~40 unfused. Track actual
  dispatch count and compare to `design.py drafter` fused-per-layer row.
- **Numerics.** Per-position parity vs reference within existing kernel tolerances
  (rmsnorm/attn tests use atol ~1e-2 on bf16 paths).
- **Energy (secondary).** Report package-delta energy/block per the E1 method; the
  model predicts ~107 mJ/block (int4) / more at int8.

## 2. Phases

Each phase has an explicit **gate** — do not start the next phase until it passes.

### Phase A — Reference + harness (no NPU kernels yet)

Stage the trained weights and lock a golden reference to validate against.

1. Convert the smallest real sidecar to HFQ:
   ```
   dflash_convert --input /srv/huggingface/models--z-lab--Qwen3.5-9B-DFlash/<snapshot> \
                  --output ~/.hipfire/models/qwen3.5-9b-mq4.dflash.hfq
   ```
   (target `qwen3.5-9b-mq4.hfq` is already staged.)
2. Golden dump: run `crates/hipfire-runtime/examples/dflash_smoke.rs` (or a new
   `dflash_ref_dump` example) that calls `dflash::draft_forward_opts` on a fixed
   seed block + fixed `target_hidden`, and dumps per-layer inputs/outputs and the
   final `[block_size, hidden]` block hidden to `.npy` in the scratchpad.
3. Extract the per-op golden tensors (post-rmsnorm, q/k/v, post-rope, attn out,
   o-proj, ffn) so each NPU primitive can be checked in isolation, not just the
   whole body.

**Gate A.** Golden reference dumps exist and are deterministic across two runs
(byte-identical), for both a random-init and the trained sidecar.

**Gate A — DONE (2026-07-18).** Deliverables landed:
- `crates/hipfire-runtime/examples/dflash_ref_dump.rs` — dumps deterministic
  seeded inputs + final `[16,4096]` block hidden to `.npy`. Byte-identical across
  two runs.
- `HIPFIRE_DFLASH_GOLDEN_DIR` env hook in `dflash::draft_forward_opts` — dumps
  per-op GPU intermediates (`rust_target_hidden_proj`, `rust_l0_input_norm`,
  `rust_l0_{q,k}_roped`, `rust_l0_v`, `rust_l0_attn_out`, `rust_l0_attn_proj`,
  `rust_l{0..4}_out`, `rust_final_block_hidden`). No-op when unset.
- `tools/npu/dflash_ref.py` — numpy float reference. Reads the same safetensors
  (bf16→f32, bit-exact to the HFQ) + dumped inputs, reproduces every per-op
  intermediate, writes them to `<ref_dir>/golden/`, and validates its final vs
  the GPU dump. **This numpy reference is the authoritative per-op golden for
  Phase B** (exact float math from real weights, no kernel/quant error).

Result vs the **F16** sidecar (production WMMA path): every primitive matches at
cos = 1.000000; full-body `max_abs = 0.017`, `mean_abs = 0.0015` (bf16 tol).

**Golden uses the F16 sidecar, NOT F32.** The F32 draft path
(`gemm_f32_batched`) has a latent transpose bug for batch>1: it writes output
feature-major `Y[n*batch + m]`, correct only at batch=1 (decode), so the F32
DFlash body (batch=block_size=16) comes out transposed (cos≈0 vs reference,
matching RMS). Production DFlash uses the F16/MQ4 kernels, which are correct, so
this never bit inference — but it means the F32 sidecar is NOT a usable golden.
See the tracked runtime finding; fix is out of scope for this NPU plan.

Sidecars: `~/.hipfire/drafts/Qwen3.5-9B.dflash.f16.hfq` (golden),
`Qwen3.5-9B.dflash.f32.hfq` (kept for the bug repro only).

### Phase B — Per-primitive parity at block granularity (M=16)

Confirm every existing primitive matches the reference at the drafter's dims. Most
already have `test_*_npu.py`; re-point them at drafter shapes.

| primitive | kernel | check |
|---|---|---|
| projection GEMM (int8 W8A8) | `oq_gemm_design.matmul_npu` per-group G256 | ✅ int32 bit-exact, all 8 shapes, halo + nix1 |
| rmsnorm | `build_qwen35_rmsnorm --hidden-size 4096` | ✅ nix1 PASS (max_rel 0.019) |
| head norm (q/k) | `build_qwen35_headnorm --n-heads 32 --n-kv-heads 8 --head-dim 128` | ✅ nix1 PASS (Q+K) |
| RoPE (FULL neox) | `build_qwen3_dflash_rope` (`dflash_rope_bf16.cc`) — NEW | ✅ nix1 PASS (theta 1e7, full head_dim) |
| SiLU/SwiGLU | `build_qwen35_swiglu --hidden-size 12288` | ✅ nix1 PASS (max_rel 0.022) |
| softmax | `build_qwen35_softmax --n-heads 32 --ctx-len 48` | ✅ nix1 PASS (max_rel 0.082) |

**RoPE needed a NEW kernel:** the existing `build_qwen35_rope`/`rope_rotate_bf16.cc`
is hard-coded to Qwen3.5 partial rotary (`n_rot = head_dim/4`); the DFlash draft
(Qwen3) rotates the FULL head_dim (neox), so `dflash_rope_bf16.cc` +
`build_qwen3_dflash_rope.py` + `test_dflash_rope_npu.py` were added (n_rot=head_dim).

**Gate B — PASSED (2026-07-18, nix1 RyzenAI-npu1/aie2).** All primitives pass at
drafter shapes on device: int8 projection (all 8 shapes int32 bit-exact + ~40 dB
W8A8), rmsnorm[16,4096], headnorm Q[32×128]/K[8×128], full-neox RoPE (theta 1e7),
SwiGLU[16,12288], softmax[32h,48ctx]. Primitive builders use the STOCK `~/.venv`
toolchain (`aie.iron.algorithms.transform`); the int8 projection uses the FORK
`~/mlir-aie-312` (`@iron.jit`) — two toolchains, both on nix1.

**Projection GEMM validated on device (2026-07-18).** `tools/npu/test_dflash_projection_npu.py`
runs the TRUE int8 W8A8 projection (per-group G256, `aie::mmul<4,8,8,int8,int8,acc32>`
via `oq_gemm_design.matmul_npu`) at every drafter shape on the halo Strix-Halo NPU:
q/k/v/o/gate/up/down/fc all **int32 bit-exact** vs numpy int64. W8A8 rescale = +40 dB
(confirmed on-device at N=1024 and on nix1 numpy for all N; halo numpy 2.x mis-reports
float SNR at N≥4096 — kernel proven by the integer check). Toolchain caveat: the
mlir-aie `single_core` `@iron.jit` example WEDGES the firmware (status 8, health-report);
`oq_gemm_design`'s design is the proven-good one. Recover a wedged NPU with
`sudo modprobe -r amdxdna && sudo modprobe amdxdna`.

**Also validated on nix1's own NPU (RyzenAI-npu1 / aie2, fw 1.5.5) 2026-07-18:**
all 8 drafter projections int32 bit-exact + clean **+40 dB W8A8** (nix1 numpy 1.26
computes the float SNR correctly, unlike halo's numpy 2.x). aie2 supports the same
`mmul<4,8,8,int8,int8,32>`. So the int8 projection runs on BOTH nix1 (aie2) and halo
(aie2p) — `oq_gemm_design` auto-detects the arch. Env (nix1): fork toolchain
`~/mlir-aie-312` (PEANO_INSTALL_DIR + PYTHONPATH=~/mlir-aie-312/install/python, run
with `~/mlir-aie-312/venv312/bin/python`).

**OQ8 quant decision + evidence (2026-07-18).** The NPU projection is a TRUE int8
compute path (W8A16 / W8A8), NOT the dequant-to-bf16 that `build_qwen3_oq8_projection`
(`aie::mmul<...,bfloat16,bfloat16>`) does — that image is a quality probe, not the
target. int8 gives 2× MACs + half weight-feed on AIE. FWHT is DROPPED for int8:
weight rotation is offline-once but the matching activation rotation is unavoidably
per-block, and at 8 bits incoherence buys ~nothing.

Numpy sim (`tools/npu/dflash_int8_sim.py`) on the real 9B drafter vs the F16 golden,
non-rotated per-group G256:

| granularity | W8A16 | W8A8 |
|---|---|---|
| G256 (per-group) | 33.2 dB / cos 0.99976 | 29.8 dB / cos 0.99948 |
| G1024 | 31.7 dB | 27.6 dB |
| per-row (K) | 30.9 dB | 26.0 dB |

Findings: (1) clip-search ≡ RTN at 8 bits (no outliers to clip) → OQ8 ≈ OQ8+ ≈
OQ8++ at W8; the "++" (Hessian/LDLQ) buys sub-dB and is a low-priority follow-on
gated on the calibration-Hessian pipeline. (2) A8 activation quant costs ~3.3 dB.
(3) per-group G256 beats per-row by 2.3–3.8 dB → the kernel MUST apply per-group
`w_scale·a_scale` (the `opus_lowbit::dot_offset_fold` structure), not a single
end-of-K scale.

**OQ8 converter — DONE.** `dflash_convert --oq8` emits a non-rotated per-group
symmetric signed-int8 sidecar (`QuantType::Oq8Plain = 45`, block `[f16 scale]
[256 int8]` = 258 B/group, `"rotated": false` in metadata). Round-trip unit tests
pass; real sidecar `~/.hipfire/drafts/Qwen3.5-9B.dflash.npu.oq8.hfq` (1008 MiB)
validates at ~43 dB weight-only SNR per projection, norms kept F32. This is the
int8 weight format both int8 kernels consume.

**Kernel target = aie2p on `halo`.** ALL existing qwen3 NPU kernels are
`aie.device(npu2)`/`--target=aie2p` (Strix Halo); nix1 is aie2/Phoenix and cannot
run aie2p. Build + run int8 kernels on halo (172.16.16.20, verified: Strix Halo
aie2p 6×8, full mlir-aie + aiecc toolchain). aie2p int8 mmul shapes:
W8A16 = `mmul<4,8,8,int16,int8,32>` (mmul_16_8), W8A8 = `mmul<4,8,8,int8,int8,32>`
(mmul_8_8) — both mirror the bf16 kernel's <4,8,8> tiling. int8 CPU reference to
validate against: `opus_lowbit::dot_signed` / `dot_offset_fold`.

### Phase C — The one new primitive: non-causal cross-attention

`build_qwen3_segmented_attention` is causal + self-only. Fork it to
`build_qwen3_dflash_attention`:

1. **Drop the causal mask** (bidirectional within the block).
2. **Cross-attention K/V staging**: Q length = block_size (16); K/V length =
   ctx_len + 16 = concat(projected `target_hidden` context, current block K/V).
   Stage the context K/V once in memtile; every layer reads it.
3. Keep the QK^T / softmax / AV micro-kernels unchanged.
4. Host-side noise-init of the entry block hidden `[16, hidden]` (buffer fill, not
   a kernel change).

**Gate C.** `test_dflash_attention_npu.py` matches the reference attention slice
(non-causal, with cross-context) within tolerance, for ctx_len ∈ {block-only,
short, 512}.

**Phase C progress (2026-07-18) — kernel authored + compiles; Gate C NOT yet met.**
The compute `.cc` (`segmented_attention_bf16.cc`) already takes `causal` as a
RUNTIME flag and does `real_length` window masking, and is arch-neutral
(`aie::mmul<4,8,8,bf16,bf16>` on aie2 + aie2p). So the non-causal cross-attention
is just: `causal=0`, stage KV=[ctx|block] with `real_length=ctx+block`, extract the
block's last-`block_size` output rows. `build_qwen3_dflash_attention.py` reuses
`build_qwen3_segmented_attention.generate_mlir` verbatim, patches `causal=0` and the
device line, and **compiles cleanly for npu2/aie2p** (final.xclbin + insts.bin).

**Gate C — PASSED (2026-07-18, nix1 single-core).** `test_dflash_attention_npu.py
--heads 0` (all 32 q-heads) vs the Phase-A golden l0: **cos = 1.00000 vs a
bf16-input reference on every head** — the kernel computes bf16 non-causal cross-
attention EXACTLY on nix1's NPU. Golden (f16) cos = 0.99999/head; the worst
`max_abs` = 0.204 is the bf16 arithmetic floor (an ideal bf16 kernel scores the
same vs the f16 golden — |attn_out| reaches ~39 and bf16's ~0.4% relative precision
forces ~0.2 abs there), so the gate compares against a bf16-input reference (the
correct tolerance for a bf16 kernel), not a fixed absolute vs the f16 golden.
Three AIE-toolchain miscompiles were found+fixed in `dflash_attention_sc_bf16.cc`:
(1) `aie::broadcast<bfloat16>(runtime_scalar)` miscompiles → broadcast the softmax
weight as `float`; (2) a `noinline` exp helper's surrounding `sum` reduction
collapsed to ~1.0 → inline the exp (IEEE-754 exponent bit-pack); (3) degree-2 exp
poly ~9% off → degree-6. kv_len=16 also passes the strict absolute gate
(max_abs 0.092). kv_len≥128 doesn't fit the single tile (Gate D KV tiling).

**Second path (nix1, single-core) — build/debug history (now PASSED, see above).** Since the 8-col kernel can't run on nix1, added a single-core drafter
attention: `dflash_attention_sc_bf16.cc` (plain bf16 layout, one q-head/dispatch,
non-causal full softmax, GQA on the host) + `build_dflash_attention_sc.py`
(`@iron.jit` + ObjectFifo, KV=[K|V] in one fifo for the 2-input DMA limit,
depth=1 for the 64 KB tile) + `test_dflash_attention_npu.py`.
- ALGORITHM validated vs golden l0 (numpy `--algo-only`): **cos = 1.000000**.
- BUILDS + RUNS on nix1 (RyzenAI-npu1/aie2) — dataflow executes.
- **Numerics WIP (corrected diagnosis):** with `TensorTiler2D` taps added to
  fill/drain, the output IS input-dependent (different q-heads → different output),
  so the kernel processes real data — but computes wrong attention (cos 0.03–0.21).
  Two things to chase: (1) a compute/layout bug in `dflash_attention_sc_bf16.cc`
  (dot / softmax / KV split / bf16 mac semantics), and (2) a suspected `@iron.jit`
  cache-invalidation gap — head-0 output stayed bit-identical after a `.cc` dot
  rewrite + `rm -rf ~/.npu/cache`, suggesting the JIT design hash does NOT track
  the ExternalFunction `.cc` **content**, so `.cc` edits may silently reuse a stale
  kernel. Debug approach for next session: force a fresh `.cc` (rename/bump a
  comment token) or verify the compiled object changes; add a Q→O passthrough
  variant to confirm the layout; compare on-device output to numpy per-op. The
  attention ALGORITHM is golden-validated (cos=1.0); remaining work is the on-device
  compute/wiring + JIT-cache handling. Committed WIP: ff48d840a, ae5f7f715, + taps.

**Two blockers to Gate C (8-col path):**
1. **nix1 can't run it:** the segmented kernel is **8-column** (aie2p/halo); nix1's
   npu1 is **4-column** (`aie.tile column index 4 must be < 4`). The drafter's tiny
   attention (16 q, 48 kv) actually wants a **≤4-col / single-core** design — a real
   (bounded) redesign reusing the same `.cc` block/init/finish. Or run on halo (npu2).
2. **No host runner exists** for the segmented kernel — the Q/KV/O multi-core bucket
   layout (query-pair + embedded real_length, GQA head map, kv MMUL interleave) must
   be staged by a new `test_dflash_attention_npu.py`, validated vs the Phase-A golden
   l0 tensors (`rust_l0_q_roped`/`k_roped`/`v`/`attn_out`). This is the main remaining
   Phase-C work. The attention MATH is already golden-validated (Phase A, cos=1.0).

### Phase D — Assemble + fuse the block body

Compose the primitives into the 5-layer forward, then fuse to cut the dispatch
floor. Build incrementally, validating at each fusion step against the Phase-A body
golden.

1. **Unfused body**: chain all ops op-by-op (each its own dispatch); validate full
   `[16, hidden]` output parity. This is correctness-first; expect ~40 dispatches.
   **DONE (Gate D step 1, nix1 npu1).** `tools/npu/dflash_body_npu.py` composes the
   Gate-B primitives + Gate-C attention into the full 5-layer body in one process
   via the shared `CachedXRTRuntime` (unified LRU so the pre-built xclbins and the
   `@iron.jit` int8 projection/attention share npu1's context budget — a separate
   `XRTHostRuntime` blew the hw-context limit, `CREATE_HWCTX err=-22`). Op-by-op
   layer-0 hand-offs all pass (cos > 0.9999 vs each golden slice). Full unfused
   body: **cos = 0.99902 vs the f16 golden `rust_final_block_hidden`** and **cos =
   0.99915 vs a bf16/int8-precision numpy reference** (per-layer `l{0..4}_out` all
   cos > 0.999); the precision reference itself sits at cos 0.99943 vs the golden,
   so the on-device body is at the bf16/int8 floor. 88 logical op-dispatches (2048
   raw, since per-group G256 projection = one matmul/group and the norm/rope/swiglu
   xclbins are per-row), ~23 s wall (unfused, correctness-first). Gate on cos, not a
   fixed abs tol (Gate-C precedent). Run:
   `dflash_body_npu.py --golden-dir <OUT>/rust --weights <safetensors> [--op-by-op]`.
   **DONE (Gate D step 2, nix1 npu1)** — dispatch reduction, levers 1/2a/4 of
   `docs/npu/dflash-phase-d-fusion-plan.md`. **2048 → 243 raw dispatches (8.4×)**,
   88 logical, with parity held:

   | config | raw | raw/layer | cos vs f16 golden | cos vs int8/bf16 ref |
   |---|---|---|---|---|
   | baseline (step 1) | 2048 | 384 | 0.999019 | 0.999153 |
   | + lever 1 (batched norm/rope/swiglu) | 1157 | 231 | 0.999019 | 0.999153 |
   | + lever 2a (full-K per-row int8 proj) | 243 | 48.6 | 0.998092 | 0.998147 |

   - **Lever 1 is numerically exact** (bit-identical cos): the `_transform_gen`
     primitives bake `N_div_n` into the instr stream, so batching is a rebuild,
     not a math change. `build_dflash_body_batched.py` emits the `-b<rows>`
     xclbins. RoPE needed a new kernel (`dflash_rope_batched_bf16.cc`): its `cs`
     was a once-acquired *param*, so one dispatch could only carry one position;
     making `cs` a second *tiled input* lets each head-tile carry its own
     position's cs.
   - **Lever 2a held the gate** (cos 0.99902 → 0.99809, still ≫ 0.99), so the
     in-core per-group epilogue (2b) was NOT needed. One `matmul_npu` over the
     whole K per projection; rank-1 rescale `C·(a_scale[m]⊗w_scale[n])`.
   - **Lever 4**: int8 weights quantized ONCE up front (8.9 s setup, offline in a
     real deployment) and uploaded to the NPU ONCE (`--resident-weights`, 0.7 s),
     instead of ~1 GB re-uploaded per pass — warm wall 2268 → 1307 ms.

2. **Fuse stage 1** per layer: rmsnorm → qkv → q/k-norm → RoPE in one dispatch.
3. **Fuse stage 3** per layer: o-proj → residual → rmsnorm → gate/up → SiLU → down
   → residual in one dispatch. (Attention stays its own dispatch — different
   dataflow; confirm this boundary is forced.)
4. Measure actual dispatch count and wall-time; compare to `design.py drafter
   --block-size 16` fused-per-layer.

**Gate D.** Full-body parity holds after fusion; dispatch count ≤ ~3/layer;
measured block wall-time < the 9B verify budget (57 ms) with margin.

**Gate D status: parity MET, dispatch/wall targets NOT yet met.** Parity holds
(cos 0.99809 vs golden, 0.99815 vs the int8/bf16 reference — both ≫ 0.99, gate
unchanged, no loosened tolerance). Remaining gap:

- **48.6 raw dispatches/layer vs the ≤3 target.** 32 of those 48 are *attention*:
  `dflash_attn_head` is one q-head per dispatch (Gate C), so the per-layer floor
  is 32 until attention batches heads in-core (GQA: acquire each kv-head's K/V
  once, loop its 4 q-heads). That — not stage fusion — is now the dominant
  structural term. Non-attention is already down to ~16.6/layer (9 projections +
  7 batched norm/rope/swiglu), which steps 2–3 would fold to ~2.
- **Warm block wall 1307 ms vs the 57 ms budget — but the NPU is not the
  bottleneck.** Measured on-device time for the whole 5-layer block is
  **23 ms (2 % of wall)**; the other **1284 ms is host/XRT round-trip at
  ~5.3 ms per dispatch** from the Python driver. So the block already fits the
  57 ms budget *in NPU compute*; hitting it end-to-end needs a lean (native,
  non-Python) driver — at a native ~100 µs/dispatch even today's 243 dispatches
  would land ≈ 47 ms. Dispatch count and driver overhead are independent levers;
  reducing dispatches alone will not reach 57 ms under the Python host.

### Phase E — DSpark heads (dflash + dspark)

Add the DSpark epilogue on the final block hidden.

1. **markov heads** — small GEMMs → per-position draft logits/features (reuse the
   projection kernel at head dims).
2. **confidence head** — projection + reduce/threshold implementing the
   `mean_confidence < conf_threshold` early-truncation gate. Small GEMM + a reduce;
   start host-side, move on-NPU only if it shows up in the dispatch budget.
3. Wire early truncation as a shortened effective block_size.

**Gate E.** DSpark head outputs match the reference (`dspark_core`) confidence +
markov values; truncation fires at the same positions as the reference.

### Phase F — End-to-end + report

1. Swap the NPU draft into a spec-decode acceptance harness (offline, not the
   daemon) and confirm acceptance rate / τ unchanged vs the GPU drafter on a fixed
   prompt set.
2. Energy: E1 package-delta/block, matched-rate + null subtraction.
3. Write results (latency, dispatch count, parity, energy, acceptance) into the
   design guide and a benchmarks/results entry; update `design.py` calibration if
   measured block time diverges from the model.

**Gate F.** Measured block hides under measured verify on ≥1 real target; results
committed.

**Phase F is now the DECIDING experiment for the weight format, not a reporting
step.** See the quant results below.

## 2a-ROOT. The verify forward is NONDETERMINISTIC (2026-07-19) — read this first

**This supersedes the framing in §2a below.** §2a described two target-path defects
(#16, #17). #17 was real and is fixed. **#16's premise was wrong**, and the actual
root cause is worse:

**The DFlash verify forward produces different outputs from bit-identical inputs.**

```
tape-on   run1 b41c5e03280f   run2 ad7656b40d07   run3 b41c5e03280f
no-tape   run1 02e621bd56b5   run2 02e621bd56b5
ar             02e621bd56b5
```

Tape-on is not a fixed wrong answer — it is a **coin flip**. Consequently **any
bisection comparing single-run md5s is measuring noise**, including several
"eliminations" recorded in §2a and on task #16. Treat them as suggestive, not
established; re-establish with repeats anything a fix depends on. (The
`coherence-gate-dflash.sh` gate also compares single runs and is not a valid
signal here.)

**Evidence.** Per-layer hidden-state hashes inside the verify forward, with
`device_synchronize` before each read: `dn_state` entering a cycle is
**bit-identical** across runs, and the prompt KV region is **bit-identical** — yet
verify-written KV rows and per-layer hidden state **differ every run**. The first
divergent layer varies (3, 11, 23) and is **always a FullAttention layer**; layers
0–2 (LinearAttention) are always stable. Present under `--no-tape` too.

**Localized to an intra-kernel race in `dispatch_attend`** (batched FullAttention).
At one cycle, KV rows for layers 3 and 7 were byte-stable across runs while layer
3's *hidden output* still diverged — the kernel returns different results from
identical stored K/V. Survives flash and non-flash tiers, so the shared suspects
are per-position causal masking via `pbs.positions` / `s.pos_buf` (async-upload vs
kernel ordering) or the `flash_partials` scratch.

Ruled out by direct experiment: uninitialized memory (`HIPFIRE_ZERO_ALLOC=1`),
graph capture, flash split-K, KV-store→attend ordering (explicit
`device_synchronize` between them), dn_state/rollback replay, and allocation
layout (`GdnTape::new_for_config` is called unconditionally even under
`--no-tape`).

### The consequence that matters

**`--no-tape` losslessness is ACCIDENTAL, not structural.** With no tape, rollback
takes `force_serial_rollback` → serial `forward_scratch`, which deterministically
rewrites `dn_state` and the committed KV rows each cycle, **scrubbing** the racy
values before they can be read. The tape path leaves them in place, so the
nondeterminism reaches the next cycle's logits.

So we do not have a lossless verify that one bug breaks. We have a **racy verify
that one code path happens to mask**. #17's fix is principled and correct, but its
validation was measured on the masked path.

**Next step:** minimal isolated repro — batched attention, `n>1`, fixed synthetic
K/V/Q, hash across runs. No model, no tape, no rollback. Tracked as **#18**;
**#16 is superseded** and most likely closes as a duplicate. A separate hazard,
**#19**, was found alongside: `prefill_chunk.rs:4716` and `:6985` hardcode
`is_boundary: false` in `KvTierInputs` while serial decode honours
`kv_cache.layer_is_boundary[layer_idx]` — harmless on this all-Q8 model
(`boundary_layers=0`), but a batched-vs-serial mismatch on any model with
boundary layers.

## 2a. CORRECTNESS: two target-path defects make output drafter-dependent (2026-07-19)

Audited against the reference (`/srv/hipfire/references/dflash/dflash/model.py::dflash_generate`)
after Phase F saw committed text change with the drafter at temperature 0. Spec
decode is lossless by construction, so that must not happen.

**Our DFlash speculative logic is CORRECT — all three of the reference's
bleed-prevention points are implemented, and the accept predicate matches.**

| reference guard | our implementation |
|---|---|
| `past_key_values_draft.crop(start)` | N/A — our draft is stateless across cycles (`speculative.rs:6660`) |
| `past_key_values_target.crop(start)` | positional overwrite; attention masks on absolute position (`speculative.rs:7541-7546`), DeltaNet rewound via `target_snap.restore_to()` (`:7522`) |
| `target_hidden[..., :acceptance_length+1, :]` | exact — `rows_to_keep = accept_len + 1` (`speculative.rs:7474`), scatter copies only those rows (`:6541`, `:7488`) |
| exact-match accept + target-sourced bonus | exact — `argmax_per_pos[i] == block[i+1]`, break on mismatch (`speculative.rs:7345`) |

Also ruled out: HIP graph capture, batched vs serial prefill, rollback branch
selection, and samplers (`repeat_penalty` defaults 1.0, `ngram` false). `cactus_delta`
defaults 0.0 and only affects temp>0.

**The bleed is in the TARGET's numerics path, in two places.**

**#16 — `GdnTape` attachment corrupts verify logits (CONFIRMED).** Same position,
same seed, `accepted=0` in both, so no speculative logic is involved:

```
TAPE ON : pos=19 seed=248068 bonus=271 accepted=0   tau 0.000
TAPE OFF: pos=19 seed=248068 bonus=198 accepted=0   tau 4.556
```

AR baseline gives 198 → tape-on is wrong, and DFlash diverges from `--ar-baseline`
at token index 1. Suspected site `qwen35/prefill_chunk.rs:2432` (tape presence swaps
the QKVZA kernel), though for an `oq4` target `is_oq4` (`:2105`) should short-circuit
that arm — **which branch actually fires needs one instrumented run.**

**#17 — GDN stochastic rounding is seeded by a GLOBAL DISPATCH COUNTER (structural).**
`GDN_REQUANT_FRAME` is a process-global `AtomicU32` (`hipfire-rdna/src/dispatch/mod.rs:115`),
incremented per GDN dispatch (`dispatch/gated.rs:298,361,452`), feeding the rounding
noise (`kernels/src/gated_delta_net_q8_routed_batch_seq.hip:118`). **Dispatch count
depends on `accept_len`**, so a different drafter yields different rounding noise →
different DeltaNet state → different target logits. The drafter reaches into the
target through a global counter. Live by default (`kv_mode=q8`); a bf16 KV/GDN target
is unaffected. Explains the residual `--no-tape` divergence at index 44.

Proposed fix for #17: seed from **absolute token position** (plus layer/head/row)
rather than a dispatch counter — deterministic, reproducible, and independent of
execution history, while keeping the decorrelation stochastic rounding provides.

**⚠ CONSEQUENCE: every acceptance/quality measurement below and in Phase F was taken
on a confounded path.** Re-measure after #16 and #17 land. Verification target: DFlash
at temp 0 must be byte-identical to `--ar-baseline` over a long generation, across
multiple drafters.

## 2b. Weight-format results (2026-07-18) — int4-class FAILS on SNR, and that may not matter

Body-level quality vs the f16 golden, real z-lab 9B drafter, via
`tools/npu/dflash_int8_sim.py`:

| mode | cos | SNR |
|---|---|---|
| w8a16 (int8 W, f32 act) | 0.999759 | **33.18 dB** |
| w8a8 (int8 W + int8 act) | 0.999489 | 29.91 dB |
| mixed4a16 (oq4.25, f32 act) | 0.960586 | **11.04 dB** |
| mixed4a8 (oq4.25 + int8 act) | 0.960390 | 11.01 dB |

**Two candidate rescues were tested. BOTH are second-order. Do not retry them.**

*Overlay count* (`n_out` sweep, storage bits = 4.0625 + n_out/16):

| n_out | bits | cos | SNR |
|---|---|---|---|
| 3 | 4.25 | 0.960586 | 11.04 |
| 7 | 4.50 | 0.964238 | 11.46 |
| 15 | 5.00 | 0.964375 | 11.47 |
| 31 | 6.00 | 0.966222 | 11.71 |
| 63 | **8.00** | 0.968764 | **12.05** |

The curve is FLAT: doubling storage (4.25→8.0 b/w) buys **1 dB**. The endpoint is
damning — at n_out=63 the format costs **exactly int8's 8.0 b/w** and delivers
12.05 dB against int8's 33.18. Same bytes, 21 dB worse. So the error was never in
the outliers the overlay targets; it is **int4 bulk resolution** on the other ~200
values per group, which sparse int8 promotion cannot touch.

*FWHT rotation* (incoherence), tested because the GEMM is weight-bandwidth-bound
(cores ~12% utilized) so an activation FWHT is nearly free in latency:

| mode | cos | SNR |
|---|---|---|
| mixed4a16 | 0.960586 | 11.04 |
| rmixed4a16 | 0.962575 | 11.25 (**+0.21**) |
| rmixed4a8 | 0.962254 | 11.21 (**+0.20**) |

**+0.2 dB.** Free in latency, also free in benefit.

**NONE OF THIS IS ANOMALOUS — there is no bug to hunt.** Quantization theory gives
~6.02 dB/bit. Measured: 33.18 → 11.04 across 4 bits = **5.5 dB/bit**. The codec is
correct; 4-bit weights are simply ~22 dB worse than 8-bit, and second-order tricks
move fractions of a dB, not bits.

### The gate itself was wrong for this artifact

`cos > 0.99` is the right gate for a KERNEL (a mismatch there means a bug). It is
the WRONG gate for a drafter's weight format: **speculative decoding is lossless
regardless of draft quality** — the target verifies every token, so degradation
costs *acceptance rate*, not correctness. A drafter at cos 0.96 is not wrong; it
proposes worse candidates, some of which get rejected.

So the deciding question is: **does the acceptance-rate drop exceed the ~2×
weight-bandwidth saving?** The economics favour testing it — the GEMM is
weight-bandwidth-bound, so halving weight bytes nearly halves the dominant term,
while draft tokens are close to free (block time is independent of activation
rows). Run Phase F across f16 / oq8 / oq4.25 drafters on a fixed prompt set before
spending anything further on codec variants.

### Caveat on a passing unit test

`oq4_25_overlay_rescues_outlier_groups` (in `dflash_convert`) PASSES — the overlay
buys >6 dB — but on synthetic data with one injected 0.9 outlier per 256-group.
Real drafter weights lack that structure, so the test validates the codec's
*mechanics*, not its *premise*. A synthetic fixture that confirms a hypothesis
about a distribution's shape is weak evidence.

## 3. Dependencies / ordering

```
A (reference) ──► B (primitive parity) ──► C (new attention) ──► D (fuse body) ──► E (dspark) ──► F (e2e)
                         └───────────────────────────────────────┘
                         B and C are independent; can run in parallel
```

A blocks everything (no golden = no validation). B and C are independent of each
other. D needs B+C. E needs D. F needs D (DFlash-only e2e possible before E).

## 4. Environment (nix1) — pinned

NPU builds/runs use the custom fork toolchain (`~/mlir-aie-312`), NOT `~/.venv`:
```
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PEANO_INSTALL_DIR=$HOME/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$HOME/mlir-aie-312/install/python:/opt/xilinx/xrt/python:$PYTHONPATH
~/mlir-aie-312/venv312/bin/python tools/npu/test_<kernel>_npu.py ...
```
NPU runs take the resource lease: coordinate with `hipfire lock {acquire,release}`
(the NPU and GPU share the box; the reference dump uses the GPU).

## 5. Risks and mitigations

- **Attention won't fuse into the proj dispatches** (different tiling). Mitigation:
  the plan already budgets attention as its own dispatch; ~3/layer holds regardless.
- **Cross-attention context size**: full ctx_len K/V staged in memtile may exceed
  512 KB/col for long contexts. Mitigation: the drafter context is a few target
  layers over a short window; if it overflows, tile the context read (the model's
  block_attn is already the context KV read, so cost is tracked).
- **bf16 vs f32 accumulation drift** across 5 fused layers. Mitigation: per-layer
  parity gate (Phase D step-by-step), not just end-to-end; keep norms in F32 (the
  converter already does).
- **LDS hazard is GPU-only** (AGENTS.local) — does not affect AIE kernels; only the
  GPU-side reference/verify. Keep reference on the known-good qwen35 path.
- **Sidecar layout mismatch** (z-lab tensor names vs loader expectations). Mitigation:
  Phase A converts + loads through the existing `DflashConfig::from_hfq` path first;
  fix naming in the converter, not the kernels.

## 6. Deliverables

- `tools/npu/build_qwen3_dflash_attention.py` + `.cc` + `test_dflash_attention_npu.py`
- `tools/npu/build_dflash_block_body.py` (assembly + fusion driver) + parity test
- `tools/npu/build_dspark_heads.py` + test
- `crates/hipfire-runtime/examples/dflash_ref_dump.rs` (golden reference dumper)
- Results in `docs/npu/npu-kernel-design-guide.md` + a `benchmarks/results/` entry
- Updated `design.py` calibration if measured block time diverges from prediction

## 7. First concrete step

Phase A.1: convert the z-lab 9B sidecar to `.dflash.hfq` and confirm it loads via
`DflashConfig::from_hfq`. Everything else validates against the reference it unlocks.
