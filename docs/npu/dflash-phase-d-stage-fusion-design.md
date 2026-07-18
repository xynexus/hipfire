# DFlash Phase D — multi-core / memtile STAGE FUSION design

Goal: get from **13.6 dispatches/layer** (current, one dispatch per logical op) to
the plan's **~3/layer** by fusing multiple logical ops into single AIE kernels.
Parity must not move: full-body cos > 0.99 vs the f16 golden AND vs the int8/bf16
precision reference (current: 0.998092 / 0.998147 — has not moved across 5 levers).

## Why this needs multi-core / memtile

The block activation `[16, 4096]` bf16 = **128 KB**, but an AIE core tile is
**64 KB**. So a single-core kernel CANNOT hold the stage activation resident — a
fused stage must stage it in **memtile** (512 KB per column, 4 columns on npu1 =
2 MB) and have cores work on slices. The projection weights are far larger still
(qkv concat int8 `[6144,4096]` = **25 MB**) and must STREAM regardless — so the
GEMM is weight-streaming-bound and the fused norms/rope are cheap pre/post passes
on the small activation.

## Current 13.6 dispatches/layer

| # | op | fuse target |
|---|---|---|
| 1 | rmsnorm (input) | **stage 1** |
| 2 | qkv concat proj (GEMM) | stage 1 |
| 3 | k_ctx/v_ctx concat proj (GEMM) | stage 1 |
| 4,5 | headnorm q, headnorm k | stage 1 |
| 6,7 | rope q, rope k | stage 1 |
| 8 | attention | stays its own dispatch |
| 9 | o proj (GEMM) | **stage 3** |
| 10 | rmsnorm (post) + residual | stage 3 |
| 11 | gate/up concat proj (GEMM) | stage 3 |
| 12 | swiglu | stage 3 |
| 13 | down proj (GEMM) + residual | stage 3 |

Target: stage1(1) + attention(1) + stage3(1) = **3/layer**.

## Incremental path (do in this order — each is independently valuable)

### Step 1 — fused headnorm+rope, FULL-NEOX (tractable, do first)
`headnorm_rope_bf16.cc` / `qwen3_headnorm_rope_bf16.cc` already fuse head-norm with
rope in one dispatch, but are hard-coded to **Qwen3.5 partial rotary**
(`n_rot = head_dim/4`). The DFlash draft needs FULL head_dim neox — exactly the
mismatch fixed for the standalone rope in `dflash_rope_bf16.cc`. Make a full-neox
variant (`n_rot = head_dim`, cs = [head_dim] half-split cos/sin, packed weight+cs
tensor param) + a batched build (`-b<rows>`), and use it for BOTH q and k.
**Saves 2 dispatches/layer (4 ops -> 2): 13.6 -> 11.6.** Low risk; validate against
the golden `rust_l0_q_roped` / `rust_l0_k_roped`.

### Step 2 — fuse rmsnorm INTO the projection GEMM (pre-pass)
The GEMM streams 25 MB of weights; the preceding rmsnorm touches only the 128 KB
activation. Fuse by giving the projection kernel a pre-pass that norms the staged
activation (in memtile) before the GEMM streams weights. Requires a CUSTOM int8
GEMM (today it's the stock `kernels.mm` via `oq_gemm_design.int_matmul`) — add the
norm pre-pass + keep the per-row int8 activation quant in-core. Saves the two
rmsnorm dispatches/layer (13.6 -> ~9.6 after step 1's -2).
Note the activation quant currently happens on the HOST; folding it in-core is part
of this step (it also removes a host round-trip).

### Step 3 — full stage-1 / stage-3 fusion (largest)
One kernel per stage, multi-phase on each core, activation resident in memtile
across phases:
- **stage 1**: rmsnorm -> qkv GEMM -> (k_ctx/v_ctx GEMM) -> headnorm -> rope.
- **stage 3**: o-proj GEMM -> residual add -> rmsnorm -> gate/up GEMM -> swiglu ->
  down GEMM -> residual add.
Each stage is a multi-phase AIE program: phases share the memtile-staged activation
and switch objectfifos per phase (weights stream per GEMM phase). This is the big
design; only attempt after steps 1–2 land.

## Guardrails

- Validate after EVERY step: `dflash_body_npu.py --batch-ops --resident-weights
  --proj row --concat-proj --attn-all-kv` plus the op-by-op layer-0 mode; cos must
  stay > 0.99 vs golden AND vs the precision reference. Do NOT loosen tolerances.
- Env: fork `~/mlir-aie-312/venv312`; ALL NPU loads through the shared
  `CachedXRTRuntime` (`aie.utils._get_default_npu_runtime()`) — a private
  XRTHostRuntime exhausts Phoenix hw-contexts (err=-22).
- `@iron.jit` must list the `.cc` in `source_files=[...]` so kernel edits invalidate
  `~/.npu/cache`.
- AIE API traps already hit (see `dflash_attention_sc_bf16.cc` comments): runtime
  `aie::broadcast<bfloat16>` miscompiles (use float broadcast); a `noinline` helper
  around a scalar reduction miscompiles (inline it); degree-2 exp poly is too coarse.
- Dispatch count is the STRUCTURAL target. The wall (<57 ms) is host-bound —
  NPU-busy was already 23 ms at 243 dispatches — so it needs a native driver +
  residency, not more fusion. Report both, honestly.
