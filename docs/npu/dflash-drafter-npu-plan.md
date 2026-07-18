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

### Phase B — Per-primitive parity at block granularity (M=16)

Confirm every existing primitive matches the reference at the drafter's dims. Most
already have `test_*_npu.py`; re-point them at drafter shapes.

| primitive | kernel | check |
|---|---|---|
| projection GEMM | `build_qwen3_oq8_projection` (m,k,n) | ✅ block-16 bit-exact (done) |
| rmsnorm | `build_qwen35_rmsnorm` | parity on `[16, 4096]` |
| head norm (q/k) | `build_qwen35_headnorm` | parity on `[16, 32/8, 128]` |
| RoPE | `build_qwen35_rope` | parity, rope_theta 1e7 |
| SiLU/SwiGLU | `build_qwen35_swiglu` | parity on `[16, 12288]` |
| softmax | `build_qwen35_softmax` | parity (used by attn) |

**Gate B.** Each primitive passes parity vs its Phase-A golden slice at M=16 on
`accel0`, at rope_theta and eps from the sidecar config.

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

### Phase D — Assemble + fuse the block body

Compose the primitives into the 5-layer forward, then fuse to cut the dispatch
floor. Build incrementally, validating at each fusion step against the Phase-A body
golden.

1. **Unfused body**: chain all ops op-by-op (each its own dispatch); validate full
   `[16, hidden]` output parity. This is correctness-first; expect ~40 dispatches.
2. **Fuse stage 1** per layer: rmsnorm → qkv → q/k-norm → RoPE in one dispatch.
3. **Fuse stage 3** per layer: o-proj → residual → rmsnorm → gate/up → SiLU → down
   → residual in one dispatch. (Attention stays its own dispatch — different
   dataflow; confirm this boundary is forced.)
4. Measure actual dispatch count and wall-time; compare to `design.py drafter
   --block-size 16` fused-per-layer.

**Gate D.** Full-body parity holds after fusion; dispatch count ≤ ~3/layer;
measured block wall-time < the 9B verify budget (57 ms) with margin.

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
