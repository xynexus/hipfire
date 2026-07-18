# DFlash Phase D — multi-core / memtile STAGE FUSION design

Goal: get from **13.6 dispatches/layer** (one dispatch per logical op) to the
plan's **~3/layer** by fusing multiple logical ops into single AIE kernels.
Parity must not move: full-body cos > 0.99 vs the f16 golden AND vs the int8/bf16
precision reference.

**Current status: 8.4 dispatches/layer, parity improved, ~3/layer NOT reached.**

| stage | dispatches/layer | cos vs golden | cos vs precision ref |
|---|---|---|---|
| baseline (5 levers) | 13.6 (68) | 0.998092 | 0.998147 |
| + step 1 headnorm+rope fused | 11.6 (58) | 0.998132 | 0.998221 |
| + step 2 rmsnorm absorbed | 9.4 (47) | 0.998200 | 0.998578 |
| + step 3a qkv/ctx row-stack | **8.4 (42)** | **0.998200** | **0.998578** |

Steps 1–3a needed exactly one new AIE kernel
(`dflash_headnorm_rope_batched_bf16.cc`); steps 2 and 3a are pure algebra on the
int8 scales and cost nothing. The remaining 8.4 -> ~3 needs the multi-phase
memtile program described under step 3, which was deliberately NOT attempted —
see the blockers recorded there.

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

### Step 1 — fused headnorm+rope, FULL-NEOX — **LANDED**
Result: **68 -> 58 raw dispatches (13.6 -> 11.6/layer)**, parity did not regress —
it improved slightly (one fewer bf16 round-trip between the two ops):

| | dispatches/layer | cos vs golden | cos vs precision ref |
|---|---|---|---|
| before | 13.6 (68) | 0.998092 | 0.998147 |
| after  | 11.6 (58) | 0.998132 | 0.998221 |

op-by-op layer 0: 8/8 PASS, `q_roped` cos 0.999900, `k_roped` cos 0.999890.

Built as `dflash_headnorm_rope_batched_bf16.cc` +
`build_dflash_body_batched.py --which hnrope_{q16,k48}`, wired behind
`dflash_body_npu.py --fuse-hnrope`.

**Trap found — core-tile DMA fanin (relevant to steps 2/3).** The obvious wiring
(tiled qk + tiled cs + a weight tensor param) needs THREE inbound DMA channels,
but an AIE2 core tile has only **2 input / 2 output**. aiecc rejects it:
> tile (0,3) requires 3 input/1 output DMA channels, but only 2 input/2 output available

Workaround used: widen the tile to `2*head_dim` so each invocation handles a head
PAIR, letting gamma and the row's cs share one stream as `[gamma | cs]`. Valid
because gamma is shared by every head and both heads of a pair sit in the same row
(n_heads even), hence the same position/cs. **Any further fusion that wants more
than two inbound streams must stage through memtile** — this is the concrete
mechanism forcing the memtile design in steps 2/3, not just a capacity argument.

### Step 1 (original plan)
`headnorm_rope_bf16.cc` / `qwen3_headnorm_rope_bf16.cc` already fuse head-norm with
rope in one dispatch, but are hard-coded to **Qwen3.5 partial rotary**
(`n_rot = head_dim/4`). The DFlash draft needs FULL head_dim neox — exactly the
mismatch fixed for the standalone rope in `dflash_rope_bf16.cc`. Make a full-neox
variant (`n_rot = head_dim`, cs = [head_dim] half-split cos/sin, packed weight+cs
tensor param) + a batched build (`-b<rows>`), and use it for BOTH q and k.
**Saves 2 dispatches/layer (4 ops -> 2): 13.6 -> 11.6.** Low risk; validate against
the golden `rust_l0_q_roped` / `rust_l0_k_roped`.

### Step 2 — rmsnorm absorbed into the projection — **LANDED**
Result: **58 -> 47 raw dispatches (11.6 -> 9.4/layer)**, parity improved again:

| | dispatches/layer | cos vs golden | cos vs precision ref |
|---|---|---|---|
| baseline    | 13.6 (68) | 0.998092 | 0.998147 |
| after step 1| 11.6 (58) | 0.998132 | 0.998221 |
| after step 2|  9.4 (47) | 0.998200 | 0.998578 |

op-by-op layer 0: 7/7 PASS (the standalone `input_norm` check disappears — there
is no standalone norm op left to check; `q_roped`/`k_roped` now validate the
norm-fold and the projection together, fed the RAW layer-0 hidden).
Wired behind `dflash_body_npu.py --fold-norm`.

**This did NOT need a custom int8 GEMM.** The plan assumed a norm pre-pass inside
the GEMM kernel; the algebra makes that unnecessary. `rmsnorm(x)[r,k] =
x[r,k]*gamma[k]/rms(x[r])` is a per-INPUT-CHANNEL scale times a per-ROW scale,
and the int8 projection already carries exactly those two degrees of freedom:

* **gamma** multiplies the activation elementwise on the host, alongside the
  per-row int8 quant that already happens there — free, no dispatch.
* **1/rms** never touches the activation at all. Per-row int8 quantization is
  invariant to a per-row rescale, so quantizing `x*gamma` gives the same int8
  codes and the `1/rms` is applied to the resulting per-row activation SCALE.
  Exact.

Absorbed: both per-layer rmsnorms + the one-time `hidden_norm` (`compute_thp` now
returns a bundle `(value, rms, gamma_name)` and defers the norm into the per-layer
k_ctx/v_ctx projection). The final `norm.weight` stays a real dispatch — its
output is the block result, not a GEMM input.

**Measured dead end — do NOT fold gamma into the WEIGHT.** The obvious "fuse into
the GEMM" move, `W'[n,k] = W[n,k]*gamma[k]` folded offline before quantization, is
equally exact in real arithmetic but was tried and measurably regresses parity:
full body **0.998132 -> 0.994632** vs golden, with the precision reference
degrading identically (0.998592 -> 0.994792). The cause is structural, not a
kernel bug: a per-INPUT-channel scale is exactly what per-OUTPUT-channel int8
quantization cannot absorb. Both variants save the same dispatches; the
activation-side fold is strictly better.

### Step 2 (original plan — superseded by the above)
The GEMM streams 25 MB of weights; the preceding rmsnorm touches only the 128 KB
activation. Fuse by giving the projection kernel a pre-pass that norms the staged
activation (in memtile) before the GEMM streams weights. Requires a CUSTOM int8
GEMM (today it's the stock `kernels.mm` via `oq_gemm_design.int_matmul`) — add the
norm pre-pass + keep the per-row int8 activation quant in-core. Saves the two
rmsnorm dispatches/layer (13.6 -> ~9.6 after step 1's -2).
Note the activation quant currently happens on the HOST; folding it in-core is part
of this step (it also removes a host round-trip).

### Step 3a — q/k/v and k_ctx/v_ctx GEMMs merged by ROW stacking — **LANDED**
Result: **47 -> 42 raw dispatches (9.4 -> 8.4/layer)**, parity **bit-identical**
to step 2 (0.998200 / 0.998578) — this fusion is exact, so the numbers do not
move at all. Cold wall 14.06 s -> 12.42 s. Wired behind `--stack-ctx`.

The two projections use the SAME per-layer `[q|k|v]` weight and differ only in
their activation: block hidden (16 rows) vs the context projection `thp` (32
rows). Per-row int8 quantization is independent per row, so vstacking the two
activations into one `[48, 4096]` GEMM is bit-identical for every kept output.

Only expressible because step 2 moved gamma to the ACTIVATION side: the two
halves carry different norms (`input_layernorm` vs `hidden_norm`), which a
shared weight could never have absorbed but which per-row activation scaling
handles for free. Rows are stacked `[thp ; hidden]` so k/v emerge already in the
ctx-then-noise order attention wants.

Costs ~2.1x the MACs in that GEMM (the q outputs of the 32 thp rows are computed
and discarded), but the 25 MB of int8 weight streams ONCE instead of twice —
which is the actual bound, hence the wall went down, not up.

### Step 3 — full stage-1 / stage-3 fusion (largest) — NOT ATTEMPTED
One kernel per stage, multi-phase on each core, activation resident in memtile
across phases:
- **stage 1**: rmsnorm -> qkv GEMM -> (k_ctx/v_ctx GEMM) -> headnorm -> rope.
- **stage 3**: o-proj GEMM -> residual add -> rmsnorm -> gate/up GEMM -> swiglu ->
  down GEMM -> residual add.
Each stage is a multi-phase AIE program: phases share the memtile-staged activation
and switch objectfifos per phase (weights stream per GEMM phase). This is the big
design; only attempt after steps 1–2 land.

**Status: deliberately not attempted.** Steps 1, 2 and 3a took the body from
13.6 to **8.4 dispatches/layer** without a single custom GEMM and without moving
parity (it improved). The remaining 8.4 -> ~3 is exactly this multi-phase memtile
work, and the current per-layer residue is:

| # | op | fusable only by |
|---|---|---|
| 1 | q/k/v/k_ctx/v_ctx stacked GEMM | — |
| 2,3 | hnrope q, hnrope k | stage-1 memtile program |
| 4 | attention | stays its own dispatch (by design) |
| 5 | o proj | stage-3 memtile program |
| 6 | gate/up GEMM | stage-3 |
| 7 | swiglu | stage-3 (nonlinear — cannot fold into scales) |
| 8 | down proj | stage-3 |

Every remaining merge requires the GEMM kernel to absorb a neighbour, which means
replacing the stock `kernels.mm` / `oq_gemm_design.int_matmul` single-core design
with a multi-phase program. Three concrete blockers, in order of severity:

1. **DMA fanin is the hard wall, confirmed empirically in step 1.** A core tile
   has 2 inbound / 2 outbound DMA channels, and ObjectFifo endpoints are
   allocated statically for the WHOLE program, not per phase. A fused stage 1
   needs activation + weight + cs + gamma inbound. So the operands must be
   multiplexed through memtile — a dataflow redesign, not a kernel edit.
2. **The resident-weight path must survive.** `design.upload_int8` /
   `matmul_npu_resident` keep ~1 GB of int8 weights on-device and the whole body
   depends on it; a rewritten program has to preserve that or lose far more than
   fusion gains.
3. **New unvalidated numerics.** Keeping the activation on-device across stage 1
   moves the GEMM output rescale (per-row act scale ⊗ per-channel weight scale),
   the headnorm, the rope, AND the int8 requant for the next GEMM into in-core
   bf16 code. Today the rescale and requant are exact f32 host arithmetic; all of
   that would become new device code needing its own parity campaign.

Next concrete move if this is picked up: do NOT start with the full stage. Start
by proving ONE memtile-staged activation feeding the existing `int_matmul` — i.e.
a GEMM whose B operand is read from memtile rather than host DDR, with numerics
unchanged. That single step exercises blockers 1 and 2 in isolation and is
independently verifiable against the current parity numbers before any op is
actually fused into it.

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
