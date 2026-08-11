# Handover — indexed OQ MoE decode, and the 122B serving path

Written 2026-08-12 at the end of a long session. Read this first, then
`docs/todo/2026-08-11-122b-paged-serving.md` for the full investigation trail
(including the dead ends, which are worth not re-walking).

## One-paragraph state

The `HIPFIRE_QWEN35_MOE_OQ_INDEXED` finite-KLD failure is **root-caused**:
routed experts carry DIFFERENT per-channel AWQ scales, but the indexed MoE path
rotates the activation once per layer with the SHARED expert's scale and hands
that one buffer to all of them. The fix (per-expert rotation) is **60% landed
and each landed piece is verified**; what remains is mechanical scratch-buffer
plumbing in two forward paths. The flag stays OFF, and there is a landmine if
anyone turns it on before that plumbing lands — see "Landmine" below.

## On origin/master

- `0dd015f24 fix(qwen35): carry the AWQ sidecar through the MoE-block repack` —
  necessary but NOT sufficient; the commit message says so explicitly.
- `0d1ea653d docs(quant-formats): reimplementation-grade KVarN spec`

Everything else from this session is on the WIP branch
**`origin/wip/indexed-oq-per-expert-awq`** (branched from
`feat/hfa-calibrate-source`, NOT from master — `ldlq.rs` and friends differ
between them, so a master-based branch would conflict):

- `35893d958 wip(qwen35): per-expert AWQ rotation for indexed OQ MoE — 3 of 5 steps`
- `3a69a20c2 wip(quantize): close the ADMM/Kronecker investigation — all levers negative`

Neither is proposed for master; see "Landmine" for why the serving one must not
land as-is. Start from that branch, not from a dirty tree.

Still uncommitted and deliberately so: `.agents/scheduled_tasks.lock` (session
state), `P1-correction.md` and `opus-improvements-conversation-extract.md`
(pre-existing, not from this session), and `docs/quant-formats/kvarn.md` — its
content is already on master via `0d1ea653d`, so re-adding it would only create a
merge conflict.

## The bug, in one paragraph

`--ldlq` pre-scales routed-expert weights at quantize time: the artifact stores
`W·s`, so the forward must compute `(W·s)·(x/s) = W·x`. The divide must precede
the FWHT (the FWHT mixes channels, so `FWHT(x/s) != FWHT(x)/s`), which means it
cannot be folded into the GEMV — every expert needs its own rotated buffer. The
indexed path instead builds ONE `pbs.x_rot_batch` per layer
(`prefill_chunk.rs:148`) using the shared expert's scale, and feeds it to every
routed expert. Measured on the 35B-A3B oq4.25++: expert0/expert1/expert7/shared
AWQ payloads all differ (control: the same tensor extracted twice hashes
identically). Routing is exactly why they differ — each expert sees a different
token subset, hence a different imatrix, hence a different `s`.

Symptoms this explains: KLD 0.030367 -> 5.108296 (ppl 7.46 -> 1171.67) with the
flag on; per-block residual divergence at **layer 0** (cosine 0.244, norm 16x —
a scale error from the first layer, not accumulated drift); decode output
degenerating to `"The capital of France 是斯"`.

## What is landed and verified (on `wip/indexed-oq-per-expert-awq`, commit `35893d958`)

1. **`kernels/src/rotate_x_mq_awq_indexed_batched.hip`** + wrapper in
   `dispatch/rope.rs` + `kernels.rs` const. Reads `topk_indices` on-device,
   selects each slot's AWQ vector from a pointer table, expands
   `[N x K] -> [N x K_TOP x K]`. A null pointer falls back to the plain rotation
   so mixed artifacts stay correct.
   Verified by `parity_rotate_x_mq_awq_indexed`: **bit-exact (0.00000000)** vs
   the trusted `rotate_x_mq_awq_batched` driven one slot at a time, three
   configs, null arm asserted to be exercised.

2. **Per-expert AWQ pointer tables** — `expert_gate_up_awq_ptrs` /
   `expert_down_awq_ptrs` on `MoeFfnWeights` (layout.rs), built by
   `build_expert_awq_ptr_tables` in loading.rs exactly like
   `expert_gate_up_ptrs`. All four construction sites updated. `None` when no
   expert has a sidecar, so non-AWQ artifacts keep the cheaper path
   byte-identically. Paged mode passes `None` and WARNS if the artifact has
   expert sidecars — it does not degrade silently.

3. **All four indexed gate_up kernels index x PER SLOT** (oq4/oq8 x
   batched/non-batched): `x` is now `[K_TOP x K]` / `[N x K_TOP x K]`.
   `parity_gemv_oq8_moe_indexed` was rewritten to feed a DIFFERENT x per slot —
   a shared-x kernel can no longer pass it — and passes on three shapes.

4. **`hfq extract` offset bug fixed** (`bin/hfq.rs`): it copied `hfqm_modules`,
   a table of absolute byte ranges into the SOURCE file, so every extract was
   unreadable (`invalid range ... for file_len`). Now dropped, with a log line.
   This is what made an earlier AWQ comparison look invalid.

## What remains

5. **Scratch buffer + rotation call, in BOTH forward paths.** Allocate
   `[N x K_TOP x dim]` f32, call `rotate_x_mq_awq_indexed_batched`, pass it to
   the indexed gate_up instead of `pbs.x_rot_batch`.
   - `prefill_chunk.rs` — the indexed gate_up dispatch is around line 1246
     (`DType::Oq8G256 => gpu.gemv_oq8g256_moe_gate_up_k8_indexed_batched(...)`),
     currently passing `&pbs.x_rot_batch`.
   - `moe_decode.rs` — around line 1160, `let xr = x_rot_local.expect(...)`.
   - The scratch lives in three structs, each with its own alloc and free path:
     `qwen35/mod.rs:525` (decode), `prefill_batch.rs:93` (prefill),
     `mtp_head.rs:350` (MTP). This is the mechanical-but-wide part.
   - The DOWN side already has the right buffer shape (`rot_batch` is
     `[N x K_TOP x mi]`); it needs the AWQ-aware rotation, not a new buffer.

6. **Verify.** Re-run the steer trace (below) and confirm layer-0 cosine goes to
   ~1.0, then re-run the KLD and confirm it returns to ~0.0304.

## Landmine

The kernels now expect a per-slot `x`, but the call sites still pass the shared
`pbs.x_rot_batch`. That path is unreachable today because the flag gates it off.
**Anyone setting `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1` before step 5 lands gets
garbage from the NEW contract.** Finish step 5, or revert item 3, before
enabling. (Related, and worth fixing whenever: `loading.rs` gates the MoE-block
repack on `env == "1"` while `qwen35_moe_oq_indexed_decode_enabled()` accepts
`"1" | "on"`, so `=on` enables dispatch without the repack — guaranteed garbage.)

## Tools built this session (use these, they save hours)

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
- `parity_gemv_oq8_moe_indexed`
- `parity_moe_down_combine_oq8_indexed`

All three-way where possible (CPU oracle + a known-good production kernel +
the kernel under test), because oracle-vs-candidate alone cannot distinguish
"kernel wrong" from "oracle wrong".

## Traps that cost time here

- **`hfq list` panics on `hfq extract` output** — fixed, but if you see
  `invalid range ... for file_len`, that is this.
- **Adding/renaming any `ADMM_PROBE_*` or other env var makes `no-gpu-ci` fail**
  on stale env docs. Fix: `cargo run -q -p hipfire-cli -- gen-env-docs`.
  Note `docs/env-vars.md` is gitignored but `env_docs.rs` is tracked; committing
  from a clean worktree off master will NOT carry your local regeneration.
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

Worth settling first: **the 122B artifact still cannot be served on this box**
(paged experts load it in 9.32 GB, but decode needs the indexed path this
handover is about). Producing a 397B the same way would produce another artifact
this machine cannot run.
