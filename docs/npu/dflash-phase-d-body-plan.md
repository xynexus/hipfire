# DFlash Phase D — assemble the 5-layer block body on the NPU

Build an NPU orchestrator that runs the DFlash drafter's 5-layer block forward by
composing the ALREADY-VALIDATED per-op NPU kernels, and validate full-body parity
against the Phase-A golden. Correctness first (unfused, ~40 dispatches), then fuse.

All primitives already pass on nix1's NPU individually (Gate B) and attention
passes (Gate C). Phase D is INTEGRATION + fusion, not new kernels.

## Architecture (decided, feasibility-checked)

**One process, fork env (`~/mlir-aie-312/venv312`).** Verified: the fork env can
load stock-toolchain xclbins via `XRTHostRuntime` (cross-load works). So a single
orchestrator runs BOTH:
- pre-built primitive xclbins (rmsnorm / headnorm / rope / swiglu) via
  `aie.utils.npukernel.NPUKernel` + `aie.utils.hostruntime...XRTHostRuntime`
  (the pattern in `test_rmsnorm_npu.py` etc.), and
- `@iron.jit` designs for the int8 projection (`oq_gemm_design.matmul_npu`) and
  attention (`build_dflash_attention_sc.run_attn_head`).

Env (every run):
```
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PEANO_INSTALL_DIR=$HOME/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=$HOME/mlir-aie-312/install/python:/opt/xilinx/xrt/python:$PYTHONPATH
PY=$HOME/mlir-aie-312/venv312/bin/python
```

## The body (per `crates/hipfire-runtime/src/dflash.rs::draft_forward_opts` and
`/srv/hipfire/references/dflash/dflash/model.py`)

Config: 5 layers, hidden H=4096, intermediate I=12288, 32 q-heads / 8 kv-heads,
head_dim=128, block B=16, ctx L=32 (tot=48), rope_theta 1e7, eps 1e-6, GQA groups=4.

One-time (before layers): `thp = hidden_norm(fc(target_hidden))`  — fc is
[H, 5*H]=[4096,20480], input target_hidden [L, 5*H], output [L, H]; then rmsnorm.

Per layer li (hidden `x` [B,H] starts = noise_embedding):
1. `xn = input_layernorm(x)`                                     rmsnorm [B,H]
2. `q = wq·xn` [B, 32*128];  `k_noise = wk·xn`, `v_noise = wv·xn` [B, 8*128]
   `k_ctx = wk·thp`, `v_ctx = wv·thp` [L, 8*128]                 int8 projection
3. `q = q_norm(q)` per head [B,32,128];  `k = k_norm([k_ctx|k_noise])` per head
   [tot,8,128]                                                    headnorm
4. rope(q) with positions [L..L+B); rope(k) with [0..L+B)         full-neox rope
5. attention: per q-head h, kv-head h//4, non-causal over tot=48  attention kernel
   → attn_out [B, 32*128]
6. `ap = wo·attn_out` [B,H];  `x = x + ap`                        int8 proj + add
7. `xn2 = post_attention_layernorm(x)`                           rmsnorm
8. `g = wgate·xn2`, `u = wup·xn2` [B,I];  `s = silu(g)*u`;
   `d = wdown·s` [B,H];  `x = x + d`                             int8 proj + swiglu
Final: `block_hidden = norm(x)` [B,H].                            rmsnorm

## Kernel run interfaces (read the test_*/build_* modules for exact I/O layout)

- **int8 projection**: `oq_gemm_design.matmul_npu(A_i8[M,K], B_i8[N,K]) -> C_i32[M,N]`.
  Per-group G256: loop K in 256 chunks, rescale each by `sw[n,g]*sx[m,g]`, sum (see
  `test_oq_gemm_npu.py` / `test_dflash_projection_npu.py` for the exact per-group
  quant + rescale). Weights: quantize each projection weight per-group int8 offline
  once. Activations: quantize per-group int8 per call.
- **rmsnorm**: `target/npu/qwen35-rmsnorm-4096.xclbin` + `-instr.bin`, run via
  XRTHostRuntime (see `test_rmsnorm_npu.py::run_test` for buffer order + dtype).
  Check whether it does one [H] row or a [B,H] batch — if per-row, loop B rows.
- **headnorm**: `qwen35-headnorm-{q,k}-{32,8}h128d.xclbin`, `test_headnorm_npu.py::run_one`.
- **rope (full neox)**: `dflash-rope-{q,k}-{32,8}h128d.xclbin`, `test_rope_npu.py::run_one`.
  Needs a cs buffer = `make_cs_buf(n_rot=128, pos, freq_base=1e7)` PER position;
  q positions [L..L+B), k positions [0..L+B). May need per-position dispatch.
- **swiglu**: `qwen35-swiglu-12288.xclbin`, `test_swiglu_npu.py::run_test`
  (silu(gate)*up over I=12288).
- **attention**: `build_dflash_attention_sc.run_attn_head(Qh[B,128], Kh[tot,128],
  Vh[tot,128], q_len=B, kv_len=tot) -> [B,128]`. Loop 32 heads (GQA). Gate C proven.

Write thin `run_*` wrappers (input np array -> output np array) reusing each module's
host machinery; keep bf16 in/out to match the kernels.

## Data sources

- Weights: read the z-lab safetensors (`/srv/huggingface/models--z-lab--Qwen3.5-9B-DFlash/
  snapshots/<snap>/model.safetensors`, bf16) — names in `tools/npu/dflash_ref.py`
  (`load_safetensors_f32`). Or the F16 sidecar `~/.hipfire/drafts/Qwen3.5-9B.dflash.f16.hfq`.
- Inputs + golden: regenerate with `dflash_ref_dump` (needs GPU + lock) —
  `HIPFIRE_DFLASH_GOLDEN_DIR=<OUT>/rust ./target/release/examples/dflash_ref_dump
  ~/.hipfire/drafts/Qwen3.5-9B.dflash.f16.hfq --block 16 --ctx 32 --out <OUT>`
  (build once: `cargo build --release -p hipfire-runtime --features deltanet
  --example dflash_ref_dump`). Gives noise_embedding.npy, target_hidden.npy,
  positions_{q,k}.npy, block_hidden.npy, and `rust/rust_l0_*.npy` per-op goldens
  (rust_l0_input_norm, l0_q_roped [16,32*128], l0_k_roped/v [48,8*128], l0_attn_out,
  l0_out [16,4096], ..., rust_l{0..4}_out, rust_target_hidden_proj, rust_final_block_hidden).

## Validation strategy (bf16-aware, like Gate C)

The kernels are bf16 (+ int8 projections), so a fixed absolute tolerance vs the f16
golden sits below the precision floor. Gate on **cos** vs the golden per-op, plus a
**bf16/int8-precision numpy reference** (mirror each op's precision: int8 per-group
projections, bf16 norm/rope/attn/swiglu) — the on-device body should match THAT
closely and the f16 golden by cos.

1. **Op-by-op, layer 0**: after each op feed the GOLDEN input for that op (from
   rust_l0_*) and check the op's output vs the next golden — isolates each hand-off.
   Targets: rust_l0_input_norm, l0_q_roped, l0_k_roped, l0_v, l0_attn_out,
   l0_post_attn_residual, rust_l0_out.
2. **Full unfused body**: chain all ops (own dispatch each), feed real inputs, check
   `block_hidden` cos vs `rust_final_block_hidden`. Count dispatches (expect ~40).
   Also check each `rust_l{li}_out` along the way.

**Gate D:** full-body cos > 0.99 vs golden (and high cos vs the int8/bf16-precision
reference); then fuse toward ~3 dispatches/layer and re-validate parity + wall-time.

## Deliverables

- `tools/npu/dflash_body_npu.py` — the orchestrator (run_* wrappers + body driver +
  op-by-op and full-body validation modes).
- Fusion (after unfused passes): fuse stage-1 (rmsnorm→qkv→qk-norm→rope) and stage-3
  (o-proj→resid→rmsnorm→gate/up→silu→down→resid) per layer; attention stays its own
  dispatch. Measure dispatch count + block wall-time (< 57 ms target).

## Notes / traps

- GPU/NPU share the box: `./target/release/hipfire lock acquire dflash-body` for the
  golden dump (GPU); release after.
- graphify hook: `graphify query "..."` before grepping source.
- Start with the op-by-op layer-0 checks (cheap, isolates each hand-off) BEFORE the
  full chain — most bugs will be I/O layout mismatches between kernels.
- Attention currently one head/dispatch, kv_len≤~64 (single tile). ctx=32→tot=48 fits.
- Don't loosen tolerances to pass; use the bf16/int8-precision reference as the honest
  gate (see Gate C precedent in `test_dflash_attention_npu.py`).
