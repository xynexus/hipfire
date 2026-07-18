# DFlash native NPU driver — close the wall-clock gap

Run the DFlash block body's dispatch sequence from **native Rust** instead of Python,
and measure the real block wall. This is the last lever for the plan's latency goal.

## Why (measured, not assumed)

| term | measured |
|---|---|
| per-dispatch overhead, Python/XRT | ~5300 µs |
| **per-dispatch overhead, native DRM** | **169 µs** (31× less) |
| NPU compute for the whole body | 23 ms |
| body dispatches (current) | 42 |

Projection: 23 ms compute + 42 × 169 µs ≈ **30 ms/block vs the 57 ms verify budget**
(1.9× under). So the latency goal is reachable **without further fusion** — the
remaining ≤3-dispatch/layer target was a proxy for latency, and the driver addresses
latency directly. The 30 ms figure is a projection from two separately-measured terms
(it slightly double-counts compute inside the dispatch number); **this task replaces
it with a real end-to-end measurement.**

## Foundation (already exists — do NOT write a new runtime)

`crates/hipfire-xdna`:
- `submit.rs` — direct amdxdna DRM ioctls (`EXEC_CMD` + syncobj wait on
  `/dev/accel/accel0`). No XRT, no Python.
- `kernel.rs` — `NpuKernel::load(xclbin_bytes, insts_bytes)`, `alloc_arg(size)`,
  `dispatch(&[&buf, ...])`.
- `xclbin.rs`, plus `gemm*` / `segmented_attention` / `qwen3_projection` modules.
- Examples to copy from: `npu_cascade_time.rs` (prep-once + timed dispatch loop),
  `hwctx_smoke.rs`, `npu_busy.rs`.

`crates/hipfire-npu` — probe/admission/inventory.

## The artifact-plumbing problem (the non-obvious part)

The body uses kernels from TWO sources with different layouts:

1. **Primitive xclbins** — `target/npu/{name}.xclbin` + `{name}-instr.bin`
   (rmsnorm, hnrope q/k, swiglu, and the `-b<rows>` batched variants). Direct to load.
2. **`@iron.jit` kernels** (int8 projection `int_matmul`, attention
   `dflash_attn_head` / `dflash_attn_all`) — these live in the JIT cache at
   `~/.npu/cache/<hash>/{final.xclbin,insts.bin}`, keyed by a hash of the design +
   CompileTime args. The native driver cannot compute that hash.

**First step: make the Python harness emit an artifact manifest.** Add a dump mode to
`tools/npu/dflash_body_npu.py` that records, for every dispatch it issues: the op
name, the resolved xclbin+insts paths, the buffer sizes and their order, and the
CompileTime shape args. That manifest is the contract the native driver consumes —
it removes all guessing about which cache dir is which kernel and what arg order each
expects. (`aie.utils` exposes the resolved artifact path for a jitted design; if not,
snapshot `~/.npu/cache` before/after a run and diff.)

## Build order

1. **Manifest dump** from `dflash_body_npu.py` (above). Verify each listed xclbin
   loads via `NpuKernel::load`.
2. **One op native** — pick the int8 projection. Feed it the SAME quantized inputs
   the Python path used (dump them to `.npy`), dispatch natively, and compare the
   int32 output **bit-for-bit** against the Python result. This proves buffer layout
   and arg order before anything is chained.
3. **Full body sequence** — chain all 42 dispatches natively, weights uploaded ONCE
   and kept resident (`alloc_arg` + fill once, reuse across dispatches and across
   blocks). Host-side glue (per-row int8 quant, rescale, residual adds, softmax
   pieces that currently live in numpy) must be ported or staged — keep it in Rust,
   f32, mirroring `dflash_body_npu.py` exactly.
4. **Measure + validate** — report cold and warm block wall, per-dispatch mean, and
   NPU-busy; validate the final `block_hidden` against the Phase-A golden with the
   same cosine gate (> 0.99 vs golden AND vs the int8/bf16 precision reference).

## Validation (non-negotiable)

Same honest gate as Phases C–E: cosine vs the f16 golden AND vs the int8/bf16
precision reference. Bit-exactness is expected for the integer GEMM steps — if the
native path differs from the Python path on the same inputs, that is a bug, not
precision. Do not loosen tolerances.

## Guardrails

- Hold the GPU/NPU lock while measuring: `./target/release/hipfire lock acquire
  dflash-native` / `... lock release`.
- The markov head does NOT belong here — it runs on one CPU thread via the top-k
  shortlist (8.7 ms/block, exact). Do not port it to the NPU.
- Keep the Python harness working; the native driver is an addition, and Python
  stays the reference for parity.
- graphify before grepping repo source.
- Report cold vs warm separately, and state plainly if the measured wall misses the
  57 ms budget and why (per-dispatch floor vs compute vs host glue).
