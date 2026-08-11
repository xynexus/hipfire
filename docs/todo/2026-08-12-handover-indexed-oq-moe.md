# Handover — indexed OQ MoE decode, and the 122B serving path

Written 2026-08-12. Updated the same day after steps 5 and 6 landed. Read this
first, then `docs/todo/2026-08-11-122b-paged-serving.md` for the full
investigation trail (including the dead ends, which are worth not re-walking).

## One-paragraph state

The `HIPFIRE_QWEN35_MOE_OQ_INDEXED` finite-KLD failure is **root-caused and
fixed**: routed experts carry DIFFERENT per-channel AWQ scales, but the indexed
MoE path rotated the activation once per layer with one representative's scale
and handed that single buffer to all of them — on BOTH the gate_up side and the
down side. Per-expert rotation is now wired through every dispatch site, and
verified end-to-end: the layer-0 residual cosine against the flag-off oracle went
from **0.244 to 0.999999**, with no layer below 0.9994 across all 40, and KLD
went from **5.108296 to 0.031515** (ppl 1171.67 -> 7.46) — 162x better, and
within 3.8% of the CPU fallback it has to match. **The flag is now ON by
default** — `HIPFIRE_QWEN35_MOE_OQ_INDEXED=0` (or `off`) falls back.

## The bug, in one paragraph

`--ldlq` pre-scales routed-expert weights at quantize time: the artifact stores
`W·s`, so the forward must compute `(W·s)·(x/s) = W·x`. The divide must precede
the FWHT (the FWHT mixes channels, so `FWHT(x/s) != FWHT(x)/s`), which means it
cannot be folded into the GEMV — every expert needs its own rotated buffer. The
indexed path instead built ONE rotation per layer and fed it to every routed
expert. Measured on the 35B-A3B oq4.25++: expert0/expert1/expert7/shared AWQ
payloads all differ (control: the same tensor extracted twice hashes
identically). Routing is exactly why they differ — each expert sees a different
token subset, hence a different imatrix, hence a different `s`.

The same argument applies to `down_proj` and was missed for longer, because the
comment there claimed all experts "share the same input residual basis". They do
not: down_proj's input is each expert's OWN hidden state.

## What is landed

Steps 1–4 (commit `35893d958`), unchanged:

1. **`kernels/src/rotate_x_mq_awq_indexed_batched.hip`** + wrapper in
   `dispatch/rope.rs`. Reads `topk_indices` on-device, selects each slot's AWQ
   vector from a pointer table, expands `[N x K] -> [N x K_TOP x K]`.
2. **Per-expert AWQ pointer tables** — `expert_gate_up_awq_ptrs` /
   `expert_down_awq_ptrs` on `MoeFfnWeights`, built by
   `build_expert_awq_ptr_tables`. `None` when no expert has a sidecar.
3. All four indexed OQ gate_up kernels can index x per slot — **now via an
   explicit `x_per_slot` parameter**, see the correction below.
4. **`hfq extract` offset bug fixed** (`bin/hfq.rs`).

Steps 5–6, this session:

5. **Per-slot x wired through every dispatch site**, with a `[N x K_TOP x dim]`
   f32 scratch (`moe_x_rot_expanded{,_batch}`) alongside the existing
   `moe_down_expanded` — same shape, same alloc/free path, so it is the
   input-side mirror of a buffer that already existed.
6. **New down-side kernel** `fused_silu_mul_mq_rotate_awq_indexed.hip` + wrapper
   `Gpu::fused_silu_mul_rotate_mq_awq_indexed`. `rot_batch` is already one row
   per (token, krank), so only the scale lookup changes — the kernel is the
   existing AWQ silu_mul+rotate with a per-slot pointer select.
7. **Null-table support.** Both indexed rotate kernels accept a null pointer
   TABLE (`Option<&GpuTensor>` in Rust), meaning no expert at this layer has a
   sidecar. Every slot then takes the plain rotation. This matters because the
   expansion is required *regardless of AWQ* — the GEMVs read x per slot either
   way — so a null table lets one code path serve AWQ and non-AWQ artifacts
   without a branch at the call site.
8. **One parse of the env flag.** `hipfire_dispatch::families::moe::
   oq_indexed_decode_enabled()` is now the only reader. See "Corrections".

### Verification

- `parity_rotate_x_mq_awq_indexed` — **bit-exact (0.00000000)** vs
  `rotate_x_mq_awq_batched` driven one slot at a time, plus a null-TABLE arm
  asserted **exactly** equal to `rotate_x_mq_batched`.
- `parity_silu_mul_rotate_awq_indexed` (new) — the down-side mirror, same
  structure, also **bit-exact** with both null arms covered.
- `parity_gemv_oq8_moe_indexed` — passes under the per-slot contract.
- **Per-layer steer trace, flag off vs flag on, 35B-A3B oq4.25++**: worst cosine
  across 40 layers **0.999423** (at L37, ordinary quant drift). Layer 0 is
  **0.999999**, previously 0.244. That is the decisive result: the layer-0 scale
  error is gone.
- **End-to-end KLD**, `moe-a-nointra.hfq` vs `moe-bf16.kldref.hfq`, 8 chunks of
  `benchmarks/calib/calib-1m.txt` at n_ctx 2048. The flag-off arm reproduces
  `0.030367` / ppl `7.462` to every printed digit, which is what confirms this
  is the same corpus the original numbers came from:

  | | mean_kld | p99_kld | ppl |
  |---|---|---|---|
  | flag OFF (CPU fallback) | 0.030367 | 0.038574 | 7.4622 |
  | flag ON (indexed OQ), before | 5.108296 | — | 1171.67 |
  | **flag ON (indexed OQ), after** | **0.031515** | 0.039192 | **7.4643** |

  **162x better**, and within 3.8% KLD / 0.03% ppl of the fallback. The residual
  3.8% is expected, not a leftover bug: the indexed path is not bit-identical to
  the CPU fallback (different accumulation order and expert-GEMV shape), so a
  small delta is the floor. A scale error cannot hide at that magnitude — the
  one this fixes showed up as 168x.

  Caveat on the absolute number: `calib-1m.txt` is a calibration corpus, so this
  is train-on-test and NOT a quality claim. It is a valid OFF-vs-ON A/B, which is
  the only question being asked here, because the bias applies to both arms
  equally. Do not quote 0.0315 as this artifact's quality.
- `./tests/no-gpu-ci.sh` rc=0.

## Corrections to the previous handover

Two things the earlier version of this document got wrong. Both mattered.

**1. The blast radius was five dispatch sites across three archs, not two.**
The previous handover said the remaining work was "mechanical scratch-buffer
plumbing in two forward paths" (`prefill_chunk.rs`, `moe_decode.rs`). In fact the
four OQ gate_up kernels are called from **18 sites in 5 files**:
`qwen35/moe_decode.rs`, `qwen35/prefill_chunk.rs`, **`hipfire-dispatch/src/
pipeline/mod.rs`** (the LIVE qwen35 serving path — `moe_family().run()`, which
both qwen35 paths delegate to by default), **`arch-minimax/src/forward.rs`** (6
sites), and **`arch-lfm2moe/src/forward.rs`** (2 sites).

Step 3 changed those kernels' x contract *unconditionally* while updating none of
those call sites. minimax and lfm2moe are **not gated by
`HIPFIRE_QWEN35_MOE_OQ_INDEXED`** — so the branch shipped a live out-of-bounds
read (`x + krank*K` on a `[K]` buffer) on both archs, not the dormant
flag-gated landmine the handover described.

**2. The fix for that is an explicit layout parameter, not a silent contract
change.** The four kernels now take `int x_per_slot`: `0` = shared `[N x K]`
(what minimax/lfm2moe pass — byte-identical to their pre-`35893d958`
behavior), `1` = per-slot `[N x K_TOP x K]`. Making the correct layout opt-in
rather than mandatory is what let the two archs I cannot test here be restored
exactly instead of being "fixed" speculatively.

## Known gaps

- **minimax and lfm2moe still share one rotation across routed experts.** That
  is their status quo, and it is only wrong if their artifacts carry per-expert
  AWQ sidecars — neither loader builds a per-expert AWQ table today, so there is
  nothing to select. If either grows AWQ-scaled routed experts, they need the
  same treatment: build the pointer tables, allocate the per-slot scratch, pass
  `x_per_slot = true`. The call sites carry a comment saying so.
- **Paged mode passes a null AWQ table and warns** if the artifact has expert
  sidecars. The 122B has none today.

## Tools (use these, they save hours)

**Per-layer divergence trace** — the oracle is the same model with the flag off;
no external reference needed. `steer_capture` is PREFILL-only and needs `pp==1`.

```
{"type":"load","model":"<model>.hfq","params":{"max_seq":2048,"dflash_mode":"off"}}
{"type":"steer_begin_capture","num_layers":40,"hidden":2048}
{"type":"steer_capture","system":"","user":"<prompt>"}
{"type":"steer_finish_capture"}
{"type":"unload"}
```
Emits `{"type":"steer_captured","means":[[...]]}` — per-block last-prompt-token
residual. Run twice (flag off = oracle, flag on = candidate), cosine-diff per
layer, and the first divergent layer localises the fault in one shot.

**Parity harnesses** (`cargo run --release -p hipfire-rdna --example ...`):
- `parity_rotate_x_mq_awq_indexed`
- `parity_silu_mul_rotate_awq_indexed`
- `parity_gemv_oq8_moe_indexed`
- `parity_moe_down_combine_oq8_indexed`

All three-way where possible (CPU oracle + a known-good production kernel +
the kernel under test), because oracle-vs-candidate alone cannot distinguish
"kernel wrong" from "oracle wrong".

## Traps that cost time here

- **The daemon SELF-LOCKS.** Never wrap `hipfire-daemon` in `hipfire lock run` —
  it deadlocks and the error names your own wrapper label as the blocker. Same
  trap as `hipfire eval` and `coexistence calibrate`. Run it directly.
- **`cargo fmt -- --check <files>` checks whole crates**, not the files listed,
  so it reports pre-existing diffs in code you never touched. Use `rustfmt` on
  the specific files.
- **Env-doc staleness fires on pure line-number churn.** Any edit that shifts a
  line containing an env read makes `no-gpu-ci` fail. Fix:
  `cargo run -q -p hipfire-cli -- gen-env-docs`. `docs/env-vars.md` is gitignored
  but `env_docs.rs` is tracked, so the regeneration must be committed.
- **Test coverage that isn't.** A parity test picked its expert index with a
  stride sharing a factor with `n_exp`, so it only ever visited two experts and
  never exercised the null-sidecar arm — while printing PASS. It now picks a
  coprime stride and asserts full coverage. Check your generators.
- **Four wrong root causes preceded the right one** (stale binary, wrong packer,
  BF16 experts, `from_ffn` paged dtypes). Every one was disproved by measuring,
  not by reading. The kernels were exact the whole time; the bug was in what fed
  them.

## Separately: the 397B decision is still open

The algorithm investigation (ADMM, Kronecker, intra-block) closed NEGATIVE — see
`docs/plans/2026-08-10-admm-quant-and-qat.md`, especially the AMDAHL section at
the top: the Cholesky is only ~21% of quantize time, so every lever in that plan
targeted the minority cost and was capped at 1.27x before it began. The 397B
quantize is 33-42 h with no speedup available. Source is staged byte-exact at
`~/.hipfire/models/models--Qwen--Qwen3.5-397B-A17B.hfa`.

The blocker named here — "the 122B artifact cannot be served on this box because
decode needs the indexed path" — is now **removed on the qwen35 side**: the
indexed path is correct. What remains before a 122B serving claim is running it,
not fixing it.
