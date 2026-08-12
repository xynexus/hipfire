# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

## [FIXED] down_proj gets no Hessian/imatrix on bf16 models — `gemv_bf16_xf32` never tapped
- Category: Correctness / Calibration
- Location: `crates/hipfire-rdna/src/dispatch/gemv.rs` `gemv_bf16_xf32`
  (L4863); gate is `capture_at_weight_gemv_wrapper`
  (`crates/hipfire-runtime/src/weights.rs` L384).
- Root cause: `weight_gemv` deliberately SKIPS its tap for BF16 and F16, because
  those "terminate in capture-aware RDNA entrypoints" and tapping both would
  double-count. That premise held for F16 (`gemv_f16_xf32` taps at L5648) and
  for batched BF16 (`gemm_bf16_x_bf16_wmma_labeled` taps), but NOT for BF16 at
  batch 1: `KernelKey::GemvBf16` routes to `gemv_bf16_xf32`, which tapped
  nowhere. So the wrapper deferred to a chokepoint that did not exist and the
  activation was captured by neither.
  Only down_proj showed the loss because it is the only qwen35 linear that
  reaches `weight_gemv` at batch 1 in bf16 — via `weight_gemv_swiglu_residual`
  -> generic `_ =>` tail -> `weight_gemv_residual` -> generic `_ =>` tail. qkv
  and gate/up are captured by the fused kernels
  (`dispatch/fused.rs` L2982-2985, L3076-3077); everything else goes batched.
  Quantized weights were never affected — they keep the wrapper tap.
- Fix (2026-08-07): one `maybe_capture_activation` in `gemv_bf16_xf32`, matching
  what `gemv_f16_xf32` has always done. Verified: the same qwen3.5-0.8b calib
  goes 162 hessians / 11 kinds -> 186 / 12, with `mlp.down_proj` x24 present —
  +24 is exactly one per layer, and 186/12 matches the known-good 2026-08-06
  artefact.
- Evidence that led here: fresh qwen3.5-0.8b calib
  (`collect_artifacts --max-tokens 512`) yielded 162 / 11 — linear_attn x5,
  mlp.gate_proj, mlp.up_proj, self_attn x4, `mlp.down_proj` absent. NOT caused
  by batched prefill: `HIPFIRE_PREFILL_BATCHED=0` reproduced the identical
  162 / 11, which is what pointed at a dtype-gated tap rather than a path.
- Impact: silent where it bites. `--ldlq` does not fail on a missing Hessian,
  it logs `ldlq: skip <t>` and falls back to RTN, so an `oq*++` built from an
  affected calib quietly RTN-quantizes its down_proj — the widest FFN matrix
  and the one the outlier-budget study found most sensitive — while reporting
  success.
  BUT the blast radius is narrow, and an earlier revision of this entry
  overstated it as "any bf16-sourced calib is suspect". A full audit of all 26
  retained calib artefacts (local + `/srv/hipfire/calib`, via
  `hipfire-coexistence artifact inspect`) found ZERO with the missing-down_proj
  signature. The reason: a calib built from an HF **safetensors directory**
  loads F16, and `gemv_f16_xf32` has always tapped; a calib built from a
  **quantized** artefact keeps the `weight_gemv` wrapper tap. Only a calib
  sourced from a **bf16 `.hfq`** hits the gap, and that workflow only started
  being used on 2026-08-07. Check provenance with `artifact inspect` —
  `metadata.source_model` — before assuming an artefact is affected.
- Still open: the collector has no coverage assertion. A dense arch should
  produce one Hessian per admitted projection per layer, and a shortfall should
  fail rather than write a partial artefact — that would have caught this at
  the point of writing instead of at quantize time.
- Related (also FIXED 2026-08-07): `qwen3.5-{0.8b,2b,4b}.calib.hfq` were a
  SEPARATE and older defect — built 2026-06-27 from `*.q8f16ref.hfq` sources at
  128 tokens, they carried `kinds=1`, down_proj ONLY, the mirror image of this
  bug. All three have been rebuilt from bf16 sources at 512 tokens and now
  carry the full 12 kinds (186/186/248 hessians), local and `/srv` copies
  md5-identical. The root cause of THAT defect was never diagnosed — the
  artefacts are replaced, but if a `q8f16ref` source is ever used for
  calibration again, audit the result.
  `/srv/hipfire/calib/FLUX.2-klein-base-4B.calib.hfq` is empty (`n_hessian`
  absent, 0 kinds, 6 MB) and is a third, separate issue. The remaining 22
  artefacts audit clean.
- Scope: Calibration / quantization quality
- Confidence: Confirmed by rebuild (an earlier revision of this entry blamed
  `weight_gemv_swiglu_residual` for having no tap; that was wrong — its generic
  tail does reach `weight_gemv`, which is where the dtype gate then dropped it)

## [RESOLVED 2026-08-11] Fused DENSE batch path was unreachable — a LUT3 lm_head, not the VL wrapper

**The VL-wrapper diagnosis below was WRONG.** `is_qwen35_dense_arch_id` was true
all along — there is no separate Qwen3.5-VL arch id, so a VL text tower is still
arch 5. The real gate, obtained by logging the refusal instead of guessing:

    fused dense declined: unsupported weights: does not support lm_head
                          (unsupported weight dtype; dtype Bf16L3)

The lm_head is emitted by the bf16 codec as LUT3 whenever it is gather-shaped and
wins on size — the DEFAULT for a tied-embedding model — and `Bf16L3` was not in
the fused body's accepted dtype list. So "fused dense" was unreachable for most
dense artifacts, independent of KV mode, AWQ, or VL.

Fixed by ADDING SUPPORT rather than disabling the codec: `gemm_bf16l3_xf32` is
the batched sibling of the `gemv_bf16l3_xf32` the serial path already uses, so the
weights stay packed and no rotation is needed (LUT3 is a lossless bf16 recoding,
not a quantization — no awq_scale or FWHT basis to undo). Verified: the fused
dense path now runs with a packed LUT3 lm_head and no env override, 1.11x launches
at width 4 versus 4.00x before.

Three wrong hypotheses preceded this (AWQ pre-scaling, weight dtype of the BODY
weights, the VL wrapper), each killed by measurement. The lesson is in the fix:
`HIPFIRE_DECODE_BACKEND_TRACE=1` now prints the named refusal, because the only
other symptom is per-row launch counts and narrowing that means guessing at one
predicate at a time.

## [Medium] DeltaNet multi-step error attributed: the KV dot product dominates, storage is least

Measured 2026-08-12 with `deltanet_error_ablation`, now that both FP64 oracles
are validated. Each precision term is switched independently while everything
else runs in f64, so a configuration's error is attributable to what is left in
f32. Relative L2 error of the STATE against an all-f64 run:

| configuration | 24 tokens | 96 tokens |
|---|---|---|
| all f32 (models the kernel) | 3.140e-7 | 1.727e-6 |
| **only KV dot + reduction f32** | **2.530e-7** | **1.298e-6** |
| only UPDATE f32 (subsumes tile) | 2.033e-7 | 1.062e-6 |
| only TILE f32 (storage alone) | 1.380e-7 | 5.941e-7 |
| only OUT dot f32 | 0 | 0 |

The model is faithful: "all f32" lands at 3.140e-7 against the GPU f32 kernel's
measured 2.997e-7 on the same shape, ~5%. It runs on the CPU but reproduces the
GPU REDUCTION ORDER (4 values per lane, then a 5-level 32-lane halving tree),
which a serial sum would not.

**`kv = <S[r,:], k>` and its reduction tree is the largest single term** — 81% of
the total at 24 tokens, 75% at 96 — and the LDS tile's f32 storage is the
smallest of the three, at 44% / 34%. That inverts where the FP16-vs-FP32 debate
has been aimed: the argument has been about STORAGE width while the dominant loss
is a 128-term dot product summed in f32.

Two rows are structural and the table must not be read as a clean decomposition:
- `only OUT dot f32` is exactly 0 because `out_v` is written to the output and
  never fed back into S, so it cannot move the state at any token count. It does
  move the logits, which this experiment does not measure — a separate question.
- `all f32 EXCEPT tile` equals `all f32` because an f32 update already yields an
  f32-valued result, making the tile's rounding a no-op. **UPD subsumes TILE, so
  the terms are not orthogonal and do not sum** (81+65+44 > 100). The isolated
  storage cost is the `only TILE f32` row, where the update runs in f64 and only
  the store rounds.

Every term grows ~5.2-5.5x for 4x the tokens, i.e. slightly superlinear, matching
the compounding seen end to end.

Actionable: compensated (Kahan/Neumaier) summation on the KV dot product and its
reduction tree targets the largest term and costs no fp64 rate penalty. That is a
better first move than any storage-format change, and this table is the argument
for it.

## [High] The FP32 DeltaNet reference drifts ~7x MORE than FP16 drifts from it

Measured 2026-08-12 with a new FP64-accumulate oracle
(`kernels/src/gated_delta_net_f64acc{,_routed_batch_seq}.hip`,
`HIPFIRE_DN_STATE_F64_ORACLE=1`). FP32 storage, `double` tile and arithmetic,
identical routing and lane mapping — so it isolates the error the f32 kernel
accrues inside its own tile from any storage round-trip.

L2 relative divergence of the DeltaNet state, 35B-A3B, 120 decode steps (pos 144):

| comparison | divergence |
|---|---|
| FP16 storage vs the FP32 kernel | 5.05e-03 |
| **the FP32 kernel vs the FP64 oracle** | **3.51e-02** |

**The reference is ~7x further from fp64 than FP16 is from the reference.** Every
FP16-vs-FP32 KLD figure quoted for this subsystem — including the 2.57e-03 that
kept FP16 opt-in — measures divergence from an accumulator that is itself drifting
harder than the thing being measured.

Why: `gated_delta_net.hip` is float throughout. The per-token update does a
`HD`-term dot product, a cross-lane `__shfl_down` tree, and a multiply-accumulate
into the state, all in f32, and the result is fed back in. Storage format is a
side issue next to that.

What this reframes:
- **FP16 state storage is a second-order concern.** Arguing about 10 vs 24
  mantissa bits of STORAGE while the ACCUMULATION loses more than that is the
  wrong axis.
- Compensated (Kahan/Neumaier) summation in f32 is the obvious lever and costs no
  fp64 rate penalty — the dot product and the `__shfl_down` reduction are both
  ordinary summations. Worth trying before any further storage-format work.
- The oracle is not a serving path: fp64 on consumer RDNA3 runs at a small
  fraction of fp32. It is a correctness reference, measured offline.

Caveat on reading these numbers: a recurrence amplifies any perturbation, so
neither figure is "the error" in an absolute sense — both are trajectory
divergence after 144 steps. The comparison is still apples-to-apples: swapping
f32->f64 accumulation moves the state 7x more than swapping f16->f32 storage does.

**Status 2026-08-12: the ORACLE ITSELF IS NOW VALIDATED, but only the PLAIN
kernel — the 3.5% figure came from the ROUTED one and stays provisional.**

`parity_gated_delta_net_f64acc` checks both GPU kernels against an independent f64
CPU implementation of the recurrence:

| kernel | rel L2 err vs f64 CPU reference |
|---|---|
| `gated_delta_net_f32` | 2.997e-7 |
| `gated_delta_net_f64acc` | **2.497e-8** |

The oracle sits at the FP32 STORAGE floor (~6e-8), which is its design point — it
accumulates in double but still stores f32, so one narrowing at the end is
unavoidable. It is 12x closer to truth than the f32 kernel, and that gap is the
term it exists to isolate.

Two things this caught, both mine:
- The first oracle used `TILE_ROWS 8` where the kernel defines **4**, inferred
  from a stale comment ("TILE_ROWS x 128 floats = 4KB"; 4x128x4 is 2KB) instead
  of read from the `#define`. The dispatcher launches `128/TILE_ROWS` blocks, so
  the blocks overran the row range: **relative error ~1.0, i.e. output unrelated
  to the reference** — while still producing plausible aggregate numbers in a
  serving run.
- The acceptance bound was first set to 1e-15, which failed a CORRECT kernel.
  The bound was wrong, not the kernel.

**The ROUTED oracle is now validated too, so the 3.5% is ESTABLISHED.**
`parity_gated_delta_net_f64acc_routed` drives the routed kernels through
session-major pointer tables against an independent f64 CPU reference, with the
three sessions' rows INTERLEAVED in the batch — a reference that processed each
session's rows contiguously would agree with a kernel that ignored routing
altogether, so the interleaving is what makes the check mean something:

| routed kernel | rel L2 err vs f64 CPU reference |
|---|---|
| `gated_delta_net_f32_routed_batch_seq` | 1.570e-7 |
| `gated_delta_net_f64acc_routed_batch_seq` | **2.585e-8** |

Same shape as the plain pair: the oracle sits at the FP32 storage floor and the
f32 kernel is ~6x worse. Both oracles are now checked against an independent
implementation, so the comparison at the top of this entry rests on measured
kernels rather than on assumption.

## [Medium] FP16 DeltaNet state error COMPOUNDS with sequence length — no bug, but the framing understates it

Investigated 2026-08-12 on the suspicion that a 45x KLD gap between the 2B and
the 35B (5.65e-05 vs 2.57e-03, from `pr/deltanet-fp16-state`) was too dramatic for
a storage-precision change. It is not a bug. Three candidate bugs were ruled out
by measurement:

- **Overflow.** `gated_delta_net_f16.hip:115` is a bare `(_Float16)` cast with no
  scale and no clamp, so FP16's 65504 ceiling applies directly. Measured max|S| is
  **16.2** on the 35B and **13.8** on the 0.8B — an order of magnitude of headroom.
  `over_fp16_max=0`, `nonfinite=0` on both.
- **Arithmetic silently done in FP16.** It is not: the kernel keeps
  `__shared__ float S_tile`, widens on load (`(float)S_global[i]`), does every
  update in f32, and narrows once on store. The "storage only, arithmetic stays
  FP32" claim holds.
- **A dtype/plumbing error.** None found.

What IS true, and what the "storage only" framing understates: **the state is a
recurrent accumulator that gets re-rounded to FP16 on every kernel invocation**,
with round-to-nearest and no error feedback. That bias compounds. Measured on the
35B, FP16-vs-FP32 relative divergence of the state's L2 norm:

| decode steps | seq pos | L2 relative divergence |
|---|---|---|
| 2 | 26 | 2.49e-06 |
| 40 | 64 | **3.22e-05** |

**13x more error for 2.5x more tokens** — superlinear, not a fixed storage cost.
So a KLD figure measured at one context length understates longer ones, and a
model with more recurrent layers accumulates more of it. That is the mechanism
behind "worse on the bigger model", together with the 35B carrying 2.6x more of
its state in FP16's low-precision region (31.3% of elements subnormal in FP16 vs
12.2% on the 0.8B; min |S| 3.1e-14 vs 3.9e-12, and FP16 flushes everything below
~6e-8 — the FP16 runs bottom out at exactly 2.98e-8).

**Dithered (stochastic) rounding was tried and is WORSE — negative result, do not
retry.** `HIPFIRE_DN_STATE_FP16_DITHER=1` narrows with a dither hashed from the
value's own bits and the element index (a pure function of the input, so
spec-decode snapshots still restore exactly what they saved — the property the Q8
path's stochastic rounding broke). It runs in both the single-session and routed
f16 state kernels. Measured, FP16-vs-FP32 L2 divergence of the state:

| decode steps | seq pos | round-to-nearest | dithered |
|---|---|---|---|
| 2 | 26 | 2.49e-06 | 1.83e-05 |
| 40 | 64 | 3.22e-05 | 5.34e-05 |
| **120** | **144** | **5.05e-03** | **2.69e-02** |

Worse at every measured length, and the gap widens with context. The two short
points alone suggested the opposite (growth 12.9x vs 2.9x, fitting to N^2.8 vs
N^1.2) — that fit was an artifact of extrapolating from two points, and the
120-step run falsified it. The lesson is the measurement, not the model.

So the compounding is NOT principally a rounding-BIAS artifact: the dither
removes bias but injects up to 1 ULP of per-step noise, and the recurrence
amplifies that noise faster than it amplifies the bias. The residual error is the
mantissa itself — 10 bits is not enough for this accumulator over hundreds of
steps, whatever the rounding mode.

The flag is left in, defaulting OFF, with these numbers on it: the branch is
uniform and free, and keeping it makes the negative result cheap to re-verify
instead of cheap to re-attempt. What is still untried is error feedback (carry an
FP32 residual, ~1.5x the FP16 size), which cancels rather than randomises the
error — a different mechanism, and the only remaining lever short of keeping the
state in FP32.

## [High] Session release LEAKS its KV and DeltaNet GPU buffers — and blocks sound CoW

Found 2026-08-12 while implementing copy-on-write checkpoints. Sessions are
released with `m.q35_registry.sessions.remove(session_id)`
(`serving-core/src/session.rs:1621`, `:1848`), which drops the
`Qwen35RequestSessionState` — but nothing frees its device memory:

- no `Drop` impl on `KvCache`, `DeltaNetState`, or the session state;
- `GpuTensor` has no `Drop` either (v2-plan risk #1 states this outright);
- the only `free_tensor` in `session.rs` is for `logits` (`:301`). Nothing frees
  `k_gpu`, `v_gpu`, `k_window`, `s_matrices`, `conv_states`.

An `OwnedTensor` RAII wrapper DOES exist (`hipfire-rdna/src/dispatch/mod.rs:377`)
and is simply not used for session state.

Per released session that is ~30 MiB of DeltaNet state (FP16) plus ~6.5 MiB of KV
on a 35B-A3B — and double that for any session that also has a retained Final
checkpoint. It is invisible in the usual way: on 42 GiB of GTT it reads as "the
model got slower" long before it reads as OOM.

**This is the prerequisite for copy-on-write checkpoints, not a detail beside
them.** CoW needs to know when a shared buffer's last referent goes away so the
survivor can free it. On a base where release frees nothing, "sharing" a buffer is
indistinguishable from leaking it twice: the implementation would appear to work —
tests would pass, memory would look fine relative to today — precisely because the
system already never frees. That is a fake CoW, and the failure mode when
ownership is later added is a use-after-free on a buffer some other session is
still reading.

Order of work: give session state real ownership first (`OwnedTensor` or an
explicit release path on the registry remove), prove it with the VRAM slope
sampled in a long multi-session run, and then build CoW on top. The acceptance
test for the CoW step already exists — `HIPFIRE_KVARN_DUMP` compares two sessions'
K state numerically, so a session reading a buffer another session wrote shows up
as a diff rather than as plausible text.

## [Medium] Batch-64 collapse is GTT exhaustion from per-session DeltaNet state, not batching

Profiled 2026-08-12 at widths 16/32/64 on `Qwen3.6-35B-A3B--oq4` (kvarn KV,
paged experts, 8 GiB expert cache, max_seq 512):

| width | sessions ok | tok/s |
|---|---|---|
| 16 | 16/16 | 5.66 |
| 32 | 32/32 | 5.54 |
| 64 | **19/64** | 2.55 |

The full error — truncated in the earlier sweep, which is why this looked like a
batching problem — names the cause exactly:

```
clone qwen35 checkpoint dn.s_matrices[14] alloc:
  hipMalloc(2097152 bytes = 2.00 MiB), free=10.6 MiB of total=43008.0 MiB
```

Ten MiB free of 42 GiB. It is an out-of-memory, and the allocation that fails is a
**checkpoint clone of DeltaNet state**, not anything KV or expert related.

**Per-session cost, from `text_config` (`linear_attention: 30, full_attention: 10`):**

| item | size |
|---|---|
| DeltaNet state — 30 layers x `[524288] F32` | **60 MiB** |
| KVarN KV — 10 layers (records + f32 window + Q8 V) | ~6.5 MiB |
| checkpoint clone of the DN state | **+60 MiB** |

**The recurrent state is ~9x the KV cost per session, and the checkpoint doubles
it.** Concurrency on a hybrid model is therefore bounded by DeltaNet state, not by
the KV cache — which is the opposite of where capacity planning usually looks, and
the opposite of where this plan's own 0.4 analysis looks (expert bytes and KV).

**FIXED for the collapse by `HIPFIRE_DN_STATE_FP16=1`, measured 2026-08-12.**
FP16 state halves the 60 MiB to 30 MiB and that is enough to make 64 sessions fit:

| width | FP32 state | FP16 state |
|---|---|---|
| 16 | 16/16, 5.66 tok/s | 16/16, 5.96 |
| 32 | 32/32, 5.54 | 32/32, 6.25 |
| **64** | **19/64, 2.55** | **64/64, 6.31** |

Zero allocation failures at any width, and achieved decode widths now reach 44
(against ~20 before). Throughput also becomes monotonic in width instead of
collapsing.

FP16 state is opt-in (`hipfire_env::DN_STATE_FP16.flag()`); it was briefly made
default on 2026-08-09 and reverted the same day because surviving Q8 dispatch arms
faulted on half-size state. Those kernels and callers have since been deleted, so
that blocker is gone; what still holds the default is that the supporting evidence
is one prompt on one model. This measurement is a second, independent reason to
want it (capacity, not just accuracy) and should be weighed in that decision.

**Checkpoint lifecycle, traced 2026-08-12 — and CoW is NOT the lever it looks
like.** The allocation that fails is a `Qwen35PrefillCheckpointKind::Final`
checkpoint, created for EVERY session after batch prefill
(`qwen35_prefill.rs:1924`), via
`sequence_state_arena_checkpoint_session_state(source -> dest)`, which keeps BOTH
the live session and the snapshot. The default eviction policy is
`SequenceStateEvictionPolicy::ManualReleaseOnly`, so those snapshots are never
reclaimed automatically. Per session that is ~67 MiB live + ~67 MiB retained.

Note `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS=0` does NOT suppress it — that gates
SemanticBoundary checkpoints only, and the Final one is unconditional. Verified:
the collapse reproduces identically with boundary checkpoints off.

Copy-on-write would help far less than it appears, because the two halves behave
differently once the snapshot exists:

| state | live session's writes | CoW value |
|---|---|---|
| KV (~6.5 MiB) | append-only PAST the checkpoint cursor; `[0, cursor)` is never rewritten | shareable permanently — a real saving, but only ~10% of the session |
| DeltaNet (60 MiB FP32 / 30 FP16) | the recurrent matrix is OVERWRITTEN every step | first decode step materializes the copy — CoW DEFERS it, and since every session decodes, peak memory is unchanged |

So CoW buys the KV tenth and defers the DeltaNet nine-tenths. It does not reduce
the peak that OOMs. The levers that actually move it, in order:
1. **FP16 DeltaNet state** — measured above, takes 64 sessions from 19/64 to 64/64.
2. **Release the Final checkpoints.** They are retained under ManualReleaseOnly
   for the process lifetime; nothing reclaims them.
3. **Do not take a Final checkpoint per session** when no prefix reuse will
   consume it — it is a snapshot for resume, and a batch of one-shot completions
   never reads it back.
- `rocm-smi --showmeminfo vram` is useless here: it reported 80-93 MiB of 256 MiB
  across the whole run, because that is the dedicated carve-out, not the 42 GiB
  GTT pool the allocator actually draws from. The allocator's own
  `free=X of total=Y` message is the only truthful source on this box.

Not yet checked: `PrefillBatchScratch` is sized from `pbs.max_batch`, so per-round
scratch (activations, fa_q/k/v, logits) also grows with width and may be a
co-factor at 64. The DN clone is what actually failed, but it failed with 10 MiB
left, so whatever else grew is complicit.

## [FIXED 2026-08-11] Routed KVarN prefill wrote K in the WRONG BASIS — both arms

Root-caused numerically after text-level A/B ran out of resolution. `HIPFIRE_KVARN_DUMP`
dumps a session's KVarN window + records at the first decode step — a point both
prefill backends reach identically — so fused and serial can be diffed directly.

`prefill_chunk.rs` rotates K and Q in place (FWHT, `rotate_x_mq_batched`) at
`head_dim == 256` before the KVarN write. `prefill_batch.rs` — the routed/fused
path — did not. The cache therefore ended up in a different basis than every
reader expects. Measured on `Qwen3.5-0.8B-Base--oq8`, layer 3, identical prompts:

| config | max abs delta on the K window |
|---|---|
| before the fix | **11.59** — every prefilled token wrong |
| rotation disabled on both sides | 0.081 (noise floor) |
| **after the fix** | **0.070** — at the noise floor |

The slot routing was always correct (exactly slots 0..seq_pos-1 differed); only
the values were wrong, which is the signature of a basis mismatch rather than a
plumbing bug.

**This affected the grouped-MoE arm too**, which had been called verified. Its
text parity matched the Q8 control's divergence positions, which looked like
evidence and was not — both arms shared the same missing rotation, and text
agreement at 4-bit K is too coarse to see it.

Two harness traps found on the way to this, both of which produced confident
wrong answers first:
- the dump initially selected the first layer with `numel() > 0`, which on a
  hybrid model is a LinearAttention **placeholder**. Both dumps came back all
  zeros and compared "IDENTICAL". A silent probe is not evidence.
- the dump takes `envelope.sessions.first()`, which is not deterministic across
  runs. With four distinct prompts the two dumps were of DIFFERENT sessions
  (`seq_pos` 19 vs 20). That confound is what made an earlier
  `HIPFIRE_KVARN_ROTATE=0` test appear to exonerate the rotation. Fixed by
  prefilling four IDENTICAL prompts.

**All three write paths re-verified numerically after the fix** — the earlier
text-based verification is superseded:

| path | max abs delta vs serial | slots affected |
|---|---|---|
| grouped-MoE prefill (35B-A3B--oq4) | **0.0044** | 0..24, all prefilled |
| dense prefill (0.8B-Base--oq8) | **0.070** | 0..18, all prefilled |
| dense DECODE, 19 steps | **0.083** | **19..37 only** — prefill slots 0..18 identical |

The decode row is the tidiest evidence in the set: with both runs sharing one
fused prefill, slots 0..18 come back bit-identical and only the decode-written
slots differ, and then only at the noise floor. Right slots, right values.

Residual: dense text parity is 3/4 diverging vs the Q8 control's 1/4. With K at
the noise floor on every path, this is consistent with 4-bit K sitting closer to
near-ties than 8-bit and flipping greedy decisions more often. Consistent, not
proven — text is the wrong instrument for the question, which is what this whole
investigation demonstrated.

## [Superseded by the above] KVarN dense arm diverges from serial far more than the Q8 control

Found 2026-08-11 once the fused dense path was reachable. Same model
(`Qwen3.5-0.8B-Base--oq8`), same fused dense backend, 48-token greedy, fused vs
serial:

| KV | prompts matching serial |
|---|---|
| q8 (control) | 3/4 |
| **kvarn (ported arm)** | **0/4** |

One kvarn divergence starts at the FIRST token — serial answers "The capital of
France is **Paris**." while fused emits a completely different reasoning-style
response. That is not the late near-tie signature the grouped-MoE arm showed,
where kvarn and Q8 diverged at byte-identical positions.

So the dense KVarN arm is NOT yet at parity with its baseline, unlike the
grouped-MoE arm which is. Do not enable dense KVarN for production on this
evidence.

**Narrowed 2026-08-11 — the PREFILL arm is implicated, and it is not the flush.**

| config | prompts diverging from serial |
|---|---|
| kvarn, fused prefill + fused decode | 4/4, one from the FIRST token |
| kvarn, **serial** prefill + fused decode | 3/4 — the token-0 case is FIXED |
| q8, fused prefill + fused decode (control) | 1/4 |

Three things this rules out:
- **Not cross-session contamination.** The independence test passes: the probe's
  output is byte-identical (755 chars) whether batched with one set of companions
  or a completely different set. State does not leak across rows.
- **Not the block flush.** These are 48-token generations from short prompts, so
  no session reaches position 127 and gather+quantize never fires.
- **Not the shared kernels.** The routed window write, routed attention and flush
  executor are the same code the grouped-MoE arm uses, and that arm matches its
  Q8 baseline at byte-identical divergence positions.

**Ruled out so far (each by measurement, none by argument):**

| hypothesis | how it died |
|---|---|
| cross-session contamination | independence test PASSES — probe byte-identical (755 ch) across different companions |
| the block flush | 48-token generations never reach position 127; gather+quantize never fires |
| the shared kernels | routed write / attention / flush are the SAME code the grouped-MoE arm uses, and that arm matches its Q8 baseline |
| a dense-vs-MoE difference in the layer body | the two `if let Some(kvarn)` arms diff by COMMENTS ONLY — functionally identical |
| a second, unported KV write site in the dense body | both layer functions have exactly one write + one attention per KV mode |
| the KVarN FWHT rotation | `prefill_chunk.rs` rotates K/Q at `head_dim == 256` and `prefill_batch.rs` does not — but disabling it on BOTH sides (`HIPFIRE_KVARN_ROTATE=0`) leaves the divergence at 4/4 |

The rotation asymmetry is real and worth fixing on its own — the routed batch path
has no KVarN rotation while the chunked path does, and both test models are
head_dim 256 (record size 17664 B). It is simply not what causes this.

**Root cause NOT found.** Text-level A/B has run out of resolution: every
remaining hypothesis needs to see the actual K values. The next step is numerical,
not another parity run — dump a session's KVarN window and records after a fused
prefill and after a serial prefill of the same tokens and diff them. That
localizes it to the write, the records, or the read in one experiment instead of
one guess per run.

## [Superseded] No available Qwen3.5 artifact can exercise the fused DENSE batch path — all VL-wrap

Found 2026-08-11 while trying to verify the KVarN dense port. Three non-AWQ dense
artifacts were quantized specifically for this and none reaches the fused dense
backend:

| artifact | source | result |
|---|---|---|
| `Qwen3.5-0.8B--oq4.hfq` | HF snapshot | serial (4.00x launches at width 4) |
| `Qwen3.5-0.8B--oq8.hfq` | HF snapshot | serial |
| `Qwen3.5-0.8B-Base--oq8.hfq` | `.hfa`, quantizer reports `Architecture: qwen3_5 (id=5)` | serial |

Every one logs `qwen3.5-vl text wrapper: mrope_interleaved=true` at load, so the
runtime wraps it as VL regardless of the arch the QUANTIZER reports, and
`is_qwen35_dense_arch_id(m.arch_id)` is false — the first term of the fused dense
decode selection. Note the quantizer and the runtime disagree about the
architecture of the same file, which is worth a look on its own.

**Not a KVarN problem.** The Q8 control on the same model is also 4.00x serial,
i.e. the path is refused in the mode it was built for. Two earlier hypotheses were
tested and killed the same way: AWQ pre-scaling (removed, still serial) and an
unaccepted weight dtype (`oq8` maps to the accepted `Oq8G256`, still serial).

Consequences:
- the KVarN **dense** arm is code-complete but unexercised; the grouped-MoE arm is
  the verified one.
- `docs/plans/2026-08-09-v2-daemon-module-major-multistream.md` names
  `qwen3.5-0.8b--oq4++.hfq` as the first-demonstration interactive model and calls
  it "arch 5 (dense)". It is not — it VL-wraps too. M3's demonstration needs a
  different model, and "dense is correct here: it isolates the scheduling claim
  from the MoE claim" does not hold with this artifact.

To verify either, a genuinely non-VL dense model is needed. Every Qwen3.5 variant
on hand (`0.8B`, `0.8B-Base`) carries the mrope/VL metadata that triggers the
wrapper.

## [Low] Fused grouped-MoE batch diverges from serial decode — systematic, NOT contamination

Found 2026-08-11 while validating the KVarN port, using the shipped Q8 path as a
control. Same model, same KV mode, same greedy decode, same prompts; the only
variable is serial (batch 1) vs fused (batch 4). Longest common prefix of the
outputs, 200-token generations on `Qwen3.6-35B-A3B--oq4`:

| prompt | kvarn | q8 (shipped) |
|---|---|---|
| bicycle derailleur | 580 chars | **580** |
| water cycle | 17 | **17** |
| printing press | 959 | 1143 (exact) |
| refrigerator | 31 | **31** |

**Three of four diverge at the byte-identical position under two different K
formats** (4-bit var-norm records vs Q8). That rules out KV quantization as the
cause and localizes it to the shared fused machinery — the routed batched
attention/MoE kernels or their reduction order — not to either KV path.

Two prompts diverge after ~17 and ~31 characters, i.e. the fused and serial
outputs are essentially different texts. Greedy decoding does amplify a near-tie
into a different continuation, and every output stays coherent, so this is not
obviously corruption. But a 1.7% common prefix is a large effect to attribute to
rounding, and it is worth deciding which it is before leaning harder on the fused
path for throughput.

Not caused by the KVarN port: the control run is on the shipped Q8 path with the
port gated off. Filed separately so it is not mistaken for KVarN fallout.

Next step if pursued: a logit-level comparison rather than text. Note
`HIPFIRE_FORWARD_ORACLE`, which `superop.rs:39` advertises for exactly this
("available for dual-run diffing"), is **not implemented** — the name appears in
that doc comment and nowhere else in the tree.

**DOWNGRADED 2026-08-11 (Medium -> Low): sessions are independent.** The test that
separates "different but valid" from "rows contaminate each other" is whether a
session's output depends on WHO ELSE is in the batch. Same probe prompt, greedy,
one daemon lifetime, batch 4 both times, only the other three rows changed:

| KV | probe alone | probe + companions X | probe + companions Y | X vs Y |
|---|---|---|---|---|
| kvarn | 776 chars | 762 | 762 | **byte-identical (762)** |
| q8 | 784 chars | 769 | 769 | **byte-identical (769)** |

Under both KV modes the probe's output is unchanged by its batch companions. So
the fused path does not leak state across rows, and it is deterministic: the
serial-vs-fused difference is the same 17-character divergence point regardless
of KV format AND regardless of batch content.

That makes this a systematic difference between two implementations of the same
math — reduction order and precision in the routed batched kernels versus the
per-session ones — rather than corruption or nondeterminism. Greedy decoding
turns one near-tie into a different continuation, which is why a benign numeric
difference presents as 1.7% common prefix.

Still worth a logit-level check if the fused path is ever leaned on for
throughput, but it is not a correctness blocker and it is not new.

## [High] `kv_cache = "auto"` bypasses the KV deprecation gate and is the shipping default

Found 2026-08-11 while auditing the KV deprecation. The deprecation added in this
branch refuses `q8`/`asym3`/etc **by name**, but the default path never presents a
name it recognises:

- `default_kv_cache()` returns `"auto"` (`hipfire-config/src/lib.rs:104`), and it
  is the `#[serde(default)]` for the config field (`:310`). `routes/chat.rs:3250`,
  `:3297` and `hipfire-cli/src/commands/chat.rs:333` also send it literally.
- `"auto"` is not in `DEPRECATED_KV_MODES`, so `reject_deprecated_kv_mode`
  (`serving-core/src/load.rs:537`) passes it.
- It then matches `"q8" | "int8" | "auto" | ""` (`load.rs:2745`) and builds a **Q8**
  cache — or `new_gpu_asym3_capped` at `head_dim == 256` (`load.rs:2491`,
  `:3634`, `session.rs:1717`). Five construction sites resolve `auto` to a
  deprecated mode.

So an operator who sets nothing gets exactly the mode the gate exists to refuse,
with no warning. An operator who names it explicitly is refused. Note the empty
string is NOT affected: `load.rs:773` normalises `""` to `fp32` *before* the gate,
so this is specific to the literal `"auto"` — which is the default.

**The fix is coupled to the kvarn port and should not be applied alone.** Pointing
`auto` at kvarn is the obvious correction and matches the stated intent (kvarn is
the default; asym/q8 deprecated), but the fused grouped-MoE prefill and decode
paths **hard-require Q8** (see the entry below). Today `auto -> q8` is the only
reason batched MoE decode is reachable at all; switching `auto` to kvarn would
silently disable fusion for every default deployment — trading a naming
inconsistency for a ~2x throughput regression at width 16. Sequence it after the
kvarn port, or land both together.

## [High] Grouped-MoE fused prefill-session batch requires Q8 KV — blocks batching on kvarn
- Category: Capacity / KV-mode coupling — the first real dependent the KV
  deprecation surfaced
- Surfaced 2026-08-10 once the KV sizing fixes (`21aca50bb`, `f2d59b442`) made
  batched prefill stop being memory-bound. The error is explicit:
  ```
  qwen35 grouped-MoE fused prefill-session batch backend failed:
    "grouped MoE session fused prefix row 0 must use Q8 KV state for the MQ4
     control path"
  ; use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial
  ```
- **This is the deprecation working as intended.** Q8 is now gated at load
  (`6a4e32b68`), and this path hard-requires it — so the fused grouped-MoE batch
  backend is a concrete port target for the kvarn migration, not an unrelated
  bug. It is exactly the "break it and the breakage names what needs fixing"
  outcome that was the point of gating rather than deleting.
- **The suggested fallback does not work.** Setting
  `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto` on the server process leaves the
  error unchanged at batch 4 and batch 16. Either `auto` still selects the fused
  backend, or the env does not reach the spawned daemon — untested which.
  `serial` is untried.
- Consequence: batch >= 2 on the 35B remains blocked, but the blocker has moved
  from memory exhaustion to a single KV-mode coupling with a named owner.
- **`serial` IS a viable interim, and it produced the first multi-stream
  generation on this model (2026-08-10).**
  `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial` with kvarn:

  | batch | ok | aggregate tok/s | per-stream tok/s |
  |---|---|---|---|
  | 2 | **2/2** | 7.86 | 3.93 |
  | 3 | **3/3** | 7.19 | 2.40 |
  | 4 | 0/4 | — | fails |

  (`auto` does NOT work — unchanged error. Only `serial` does.)
- **The Q8 coupling is in TWO paths, not one, and the second has a batch
  threshold.** Under `serial` the prefill error disappears and the SAME
  requirement reappears at decode:
  ```
  batch decode: qwen35 fused grouped-MoE native decode advance:
    "grouped MoE session fused prefix row 0 must use Q8 KV state ..."
  ```
  It is clean at batch 2-3 and fires at batch 4, so the fused grouped-MoE decode
  advance appears to engage at batch >= 4 and hard-requires Q8. The port target is
  therefore both the fused prefill AND the fused decode path.
- **Throughput result worth noting on its own: batching buys nothing here yet.**
  Aggregate is FLAT from batch 2 to 3 (7.86 -> 7.19 tok/s) while per-stream falls
  3.93 -> 2.40. ~~At these widths the sessions are serialising rather than sharing a
  pass over weights~~ — **RETRACTED 2026-08-11, see below: at batch 1-3 the fused
  path is not selected at all, so this measured the serial fallback, not fusion.**
- **Port target localized 2026-08-10 — ONE validator, not two paths.** Correcting
  the previous line: the fused decode does not carry its own copy of the check, it
  reuses the prefill contract, which is why the DECODE failure said "prefix" and
  matched the prefill message verbatim.
  - `validate_grouped_moe_prefill_session_batch_state_contract`
    (`qwen35/prefill_batch.rs:1061`) is the single enforcement site:
    ```rust
    if !signature.kv_quantized || !signature.kv_quant_q8 {
        return Err("... must use Q8 KV state for the MQ4 control path")
    }
    ```
    kvarn sets `quant_kvarn`, never `quant_q8`, so it fails here. Note kvarn is
    NOT in the adjacent asym/fwht rejection list — it is excluded only by the
    positive Q8 test.
  - The fused entry point is named
    `forward_prefill_grouped_moe_session_batch_prefix_q8_kv`
    (`prefill_batch.rs:2907`), and a sibling error reads "first MoE target is
    plain Q8 KV". Q8-only was a deliberate first-target scope with a named
    extension point, not an accident — so the port is generalising a contract
    that anticipated this, not undoing a mistake.
  - Every other caller of the validator is a test in `qwen35/mod.rs` (~4856-4951),
    including one asserting the fp32 rejection, so those pin the current contract
    and will need updating with it.
- **Measured 2026-08-11 — the flat curve below batch 4 was POLICY, and fusion
  itself gives 1.11x.** Three separate corrections, all from direct measurement:
  - `qwen35_grouped_moe_decode_auto_latency_gate_passed` is `session_count >= 4`
    (`hipfire-generate/src/lib.rs:1522`). Below that, `auto` selects
    `SerialReference` deliberately. `HIPFIRE_LAUNCH_TRACE=1` confirms it
    structurally: width 3 issues **exactly 3.00x the launches of width 1 with every
    grid dimension unchanged** (1322 launches per row, three times). So every
    batch 1/2/3 throughput number ever quoted in this entry measured the serial
    fallback. The flatness was designed, not broken.
  - **Where fusion does run, 4 rows buy 11%.** Under Q8 KV via
    `HIPFIRE_KV_ALLOW_DEPRECATED=1`, one daemon lifetime, `decode_step rows=4`
    confirmed on 32 steps: batch 1 = 7.92 tok/s aggregate, batch 4 = 8.80
    (**1.11x**), per-stream 7.92 -> 2.20. That is roughly what the amortization
    curve predicts at 4 slots — its knee is near `n_exp/k` = 512/8 = **64** — so it
    is NOT evidence the fused kernel is broken. It means no reachable batch width
    is anywhere near the knee.
  - **Batch 8+ is currently unmeasurable for an unrelated reason:** HTTP 429 from
    the `requests_per_minute` bucket in `hipfire-server/src/api_auth.rs`, not from
    anything in the batching path. Raise it in the server config before quoting any
    N >= 8 number.
- **[Independent, same repro] The refusal fires at EXECUTION, not selection, and
  takes the request down instead of falling back.** At batch >= 4 with kvarn the
  auto path sets `FusedGroupedMoeLayerChunked` — the selection-time capability
  validator does not test KV mode — and the Q8 requirement is then asserted deep in
  the decode advance, returning an error to the client rather than degrading to
  `SerialReference` as batch 1-3 does. Two defects in one:
  - the capability predicate is not wired into *backend selection*, which is the
    exact anti-pattern the v2 plan lists as a Tier-1 prerequisite;
  - the error is delivered as **HTTP 200 with an `{"error": ...}` body**, so any
    client checking status codes sees success. That alone is worth fixing
    independently of the port — it is how this failure hid inside a sweep harness
    that counted 200s as successes.
- **RESOLVED 2026-08-11 — batching DOES scale (2.08x at width 16); the flat curve
  was four caps, not one defect.** Measured with Q8 KV, loopback bind, raised
  `BATCH_MAX`, `auto` prefill; prefill and decode separated by differencing
  `max_tokens=1` against `max_tokens=64` in one daemon lifetime:

  | width | prefill | decode step | decode tok/s | vs w1 |
  |---|---|---|---|---|
  | 1 | 0.93 s | 11.3 ms | 88.6 | 1.00x |
  | 8 | 3.45 s | 52.2 ms | 153.3 | 1.73x |
  | 16 | 7.05 s | 87.3 ms | 183.3 | **2.07x** |

  End-to-end at width 16 is 16.5 tok/s vs 7.9 at width 1. The caps, each of which
  flattens the curve on its own:
  - `max_in_flight_text = 4` in `RatePolicy::default()` — a CONCURRENCY cap, and
    the actual source of the HTTP 429 above (not the per-minute bucket). Binding
    `--host 127.0.0.1` selects `loopback_default()` where it is 0 = unlimited.
  - `BATCH_MAX_DEFAULT = 8` (`hipfire-server/src/batch_runner.rs:421`) — the
    envelope never exceeds 8 rows however many sessions are waiting.
  - the `n >= 4` decode latency gate (above).
  - `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=serial`, which was a harness carryover:
    16 sequential prefills are 73% of wall time at width 16. **Under Q8, `auto`
    prefill works** (14.80 s -> 7.05 s at width 16), so the earlier note in this
    entry that "`auto` does NOT work — only `serial` does" holds for kvarn only.
- **Single-stream MoE decode is launch-bound on gfx1103.** A width-1 decode step
  issues **1322 launches in 11.3 ms** — ~8.5 us each, essentially ROCm's launch
  overhead. At width 16 the same 1322 launches take 87.3 ms (66 us each), i.e. it
  has crossed into real work. This is why batching amortizes ~2x when the
  expert-byte curve predicts only ~1.14x at that width, and it is an argument for
  grouping modules into FEWER launches.
- **Width 64 is not reachable on nix1.** Raising `BATCH_MAX` to 64 yields an
  achieved width near 18; batch 64 collapses to 2.22 tok/s with 20/64 sessions and
  `generate_batch_prefill ... failed to create checkpoint`. Since the MoE
  amortization knee is at `n_exp/k` = 512/8 = 64, the capacity argument cannot be
  evaluated on this box at 35B-A3B.
- **RESOLVED 2026-08-11 by the KVarN port; the fallback defect now matters MORE.**
  `qwen35_kvarn_fused_batch_enabled()` defaults ON as of this date, so KVarN — the
  supported mode — reaches the fused path and batched MoE decode is live again.
  But the refuse-at-execution defect above is now the kill switch's sharp edge:
  verified that `HIPFIRE_QWEN35_KVARN_FUSED_BATCH=0` with KVarN KV makes batch >= 4
  requests FAIL rather than fall back to `SerialReference`. Anyone reaching for the
  kill switch during an incident would trade a suspected numerical issue for hard
  request failures.
- **FIXED 2026-08-11 — the refusal now happens at SELECTION, on both paths.** Two
  separate holes, and the second was the interesting one:
  - prefill: `validate_qwen35_fused_grouped_moe_prefill_model_capability`
    (`serving-core/src/session.rs`) never looked at the KV mode, so selection
    could not see the incompatibility. Added a narrow check that rejects exactly
    KVarN-with-the-gate-off and leaves every other mode's routing untouched.
  - decode: `validate_qwen35_grouped_moe_decode_model_capability` built a
    SYNTHETIC probe signature with `kv_quant_q8: true` hardcoded, which made its
    KV test vacuous — it passed for every mode, including modes the body would
    then reject. The probe now derives its flags from `m.q35_kv_mode`. A
    capability probe that asserts the capability it is meant to test is worse than
    no probe: it reports "supported" for a configuration that fails on the next
    call.

  Verified end to end. With `HIPFIRE_QWEN35_KVARN_FUSED_BATCH=0` and KVarN KV,
  batch 4 now SUCCEEDS and its output is byte-identical to serial on 4/4 prompts
  (i.e. it really did route to `SerialReference`); with the gate at its default it
  diverges on 2/4, the fused signature. The divergence profile doubles as a
  backend detector, which is how selection was confirmed rather than assumed.
- **Consequence worth stating plainly: batched MoE decode is presently
  unreachable.** q8 is on `DEPRECATED_KV_MODES` (`serving-core/src/load.rs:533`)
  and the fused path requires q8, so on every supported KV mode there is no batch
  size that fuses — below 4 it is policy-serial, at 4+ it errors. The kvarn port is
  therefore not an optimization; it is a prerequisite for measuring the v2 plan's
  central amortization claim at all.
- **Port RE-SIZED 2026-08-11 — the 2026-08-10 sizing below was materially wrong.**
  It said the port was "(1) add a kvarn dispatch arm using the existing
  `attention_kvarn_routed_batched`, (2) widen the validator". Step 1 is real but
  incomplete, and it omits the entire write half:
  - **`attention_kvarn_routed_batched` is READ-ONLY.** Its doc is explicit — "K
    dequant is in place", "Mirrors `attention_q8_0_routed_batched`". It does not
    write KV (`hipfire-rdna/src/dispatch/attention.rs:1169`).
  - **It needs THREE pointer tables, not two:** `rec_ptrs` (4-bit K block
    records), `win_ptrs` (f32 recent window), `v_ptrs` (Q8_0 V). The f32/q8 arms
    take `kv_k_ptrs`/`kv_v_ptrs`. So
    `DensePrefillSessionBatch{Host,Device}PointerTables` and
    `...PointerTableShape` each need a third table plumbed through
    `validate_shape` and every construction site (19 refs, 2 files — contained).
  - **There is no routed kvarn write, at all.** In kvarn the write is fused into
    `kvarn_attend` (`hipfire-rdna/src/dispatch/kv.rs:1609`), which is
    single-session by construction: it takes `records`/`window`/`v_cache` as bare
    tensors with one scalar `start_pos`, and appends K via a HOST-side loop of
    `memcpy_dtod_at_auto` at 128-token block boundaries. Routed rows have per-row
    sessions and per-row positions, so none of that transfers. `kernels/src/`
    has `kv_cache_write_{f32,q8_0}_routed_batched.hip` and no kvarn equivalent.

  **Two ways to close the write gap, and they differ by ~an order of magnitude:**

  | option | cost | captures |
  |---|---|---|
  | A: new routed kvarn K-write kernel | a new HIP kernel (quantize + append routed by `row_session_indices`/`row_positions`) | everything |
  | B: keep the write per-session (loop the existing single-session append), route only ATTENTION | plumbing + a loop | nearly everything — see below |

  **Recommend B first.** The measurement that decode is *launch-bound* (a width-1
  step is 1322 launches in 11.3 ms, ~8.5 us each) also bounds what B costs: at
  width 16 a per-session write loop is 16 sessions x 10 attention layers = ~160
  copy ops per step against an 87 ms step — order 1.5%. Attention is the part that
  actually amortizes, and B routes it. A is the right end state but should be
  justified by a measurement of B's residual, not assumed.

  Ordering is unchanged and still load-bearing: **widen
  `validate_grouped_moe_prefill_session_batch_state_contract` LAST.** Widening it
  before a kvarn read/write path exists routes kvarn KV into the Q8 kernel, which
  is silent corruption rather than an error.
- **Port sized 2026-08-10 — and relaxing the validator ALONE would be a
  correctness bug.** `prefill_batch.rs` dispatches exactly two attention kernels:
  ```
  gpu.attention_f32_routed_batched
  gpu.attention_q8_0_routed_batched
  ```
  There is no kvarn arm and no asym arm, across 37 KV-mode branches in the file.
  So admitting kvarn at the contract without adding a dispatch arm would route
  kvarn-quantized KV into the Q8 (or fp32) kernel — silent wrong output, not an
  error. This is the same accept-and-miscompute class as the indexed-OQ null table
  earlier in this branch, and it is why the validator must not simply be widened.
- **The kernel needed already exists:** `attention_kvarn_routed_batched.hip`
  (alongside `attention_flash_kvarn_tile_batched.hip`). So the port is two
  coordinated edits, not new kernel work:
  1. add a kvarn arm dispatching `attention_kvarn_routed_batched` beside the
     existing f32/q8 arms, and
  2. extend `validate_grouped_moe_prefill_session_batch_state_contract` to accept
     `kv_quant_kvarn` — in that order, so the contract never admits a mode the
     dispatch cannot serve.
  The tests in `qwen35/mod.rs` (~4856-4951) pin the current contract and move with
  step 2.
- Until then `serial` + kvarn caps usable concurrency at 3.
- Scope: blocks multi-stream measurement on the target model
- Confidence: High (explicit runtime error naming the requirement)

## [FIXED] max_seq was inflated by the generation budget, sizing KV for a 132K context
**Resolved `21aca50bb`.** `--max-seq` is now a hard cap. Title corrected: the
cache was never sized from `max_position_embeddings` — that was an inference from
arithmetic that happened to land near 262144/2. The real chain is:

    ~/.hipfire/config.json sets max_tokens = 131072 (a deliberate 128K server)
      -> load_params_for_model_config: max_seq = max(max_seq, max_tokens + 1024)
      -> 512 becomes 132096
      -> every session allocates KV for a 132K context (~34 MiB per tensor)
      -> 42 GiB device full on the FIRST request; concurrency capped at 2

Confirmed by bisection: `--max-tokens 64` gave `max_seq=1088`, unset gave
`max_seq=132096`. After the fix, `max_seq=1028` and four sequential requests
succeed where 2-4 previously OOMed. Batched prefill stops being memory-bound and
now fails on a functional grouped-MoE error instead.

**`max_tokens = 131072` is NOT a defect** — an earlier revision of this entry
flagged it as unexplained and suspected a resolution bug. It is the operator's
own config asking for a 128K context, and the config was doing exactly that. The
bug was only that an explicit CLI cap did not win over it.

**Operational consequence worth knowing:** with that config as-is, the *default*
server posture on this box allocates ~1.37 GB of KV per session, so multi-stream
serving is impossible without either lowering `max_seq`/`max_tokens` in the
config or passing `--max-seq` per run.


- Category: Correctness / Capacity (KV sizing) — root cause of the batch-prefill OOM
- Measured 2026-08-10 on `Qwen3.6-35B-A3B--oq4` (nix1, gfx1103, 42 GiB GTT).
- **`--max-seq` has no effect on KV allocation.** `--max-seq 128` and
  `--max-seq 2048` produce the byte-identical request and the identical
  free-memory progression:
  ```
  --max-seq 128 : hipMalloc(71860224 bytes = 68.53 MiB), free=79.1 MiB
  --max-seq 2048: hipMalloc(71860224 bytes = 68.53 MiB), free=79.1 MiB
  ```
  A 16x change in configured context changes nothing.
- **It is sizing for the model's full context.** 71,860,224 / 262,144
  (`max_position_embeddings`) = **274.1 bytes/position**, which is exactly
  2 `num_key_value_heads` x 256 `head_dim` at 4-bit kvarn (256 B) plus ~18 B of
  scales. The allocation is for 256K positions regardless of what the operator
  asked for.
- **Cost per session: ~1.37 GB** (68.53 MiB x 10 KV-carrying layers x {K,V}),
  against an expected ~40 MB at max_seq=512. That is ~35x over-allocation and it
  is why concurrency caps at 2 on a 42 GiB device.
- **Also a per-request leak, separately.** Sequential requests (same session,
  which should reuse state) decline monotonically:
  ```
  req 1 ok | req 2 free=79.1 | req 3 free=53.1 | req 4 free=27.1 | then flat
  ```
  ~26 MiB lost per request until nothing is left. Consistent with the v2 plan's
  risk 1: `GpuTensor` has no `Drop` and `serving-core/src/model.rs:266` already
  documents ~2 GB/forward accumulation on a missed free.
- **After ONE request the device is at 79 MiB free of 43,008 MiB.** A 19 GB model
  leaves ~42 GiB consumed, i.e. ~2.2x its own size, before any concurrency.
- **This supersedes the earlier "batched prefill OOMs" diagnosis.** The
  checkpoint clone was not oversized — it was a normal-sized clone of a KV cache
  that is itself ~35x too large. Every "wall" seen previously (fp32's 258 MiB
  single buffer, the 68.53 MiB clone, kvarn hitting the same number) is the same
  root cause seen through different call sites.
- Diagnosis needed the allocator to report free/total at failure; a bare
  "out of memory" on a 68 MiB request looked like pool placement or
  fragmentation and was neither.
- **LOCALIZED 2026-08-10.** `physical_cap` is derived in `load.rs` as
  `requested.clamp(512.min(max_seq), max_seq)` — it is *clamped to* `max_seq` and
  cannot exceed it. The observed cache is ~264K positions, so `load_model`'s
  `max_seq` argument is itself ~264K, not the operator's 512. The value is lost
  BEFORE `load_model`, not inside the KV constructors (which faithfully use
  `m.max_seq` / `m.physical_cap`).
  `hipfire serve --max-seq N` inserts `max_seq` into the CLI config layer
  (`commands/serve.rs`), the same mechanism `--kv-cache` uses and which demonstrably
  works — so the gap is between that config value and the daemon load frame's
  `params`. The daemon path plumbs it correctly (its load frame reports
  `physical_cap: 2048` at `max_seq: 2048`); the server path is where it is dropped.
- **Re-measured after the kvarn session-arm fix (`c67e7ee91`)**, because the
  earlier `physical_cap=132096` reading came from the asym3 fallback cache that
  fix removed — the kvarn log line never printed a cap, so that number could not
  be carried forward:
  - failing allocation halved, 68.53 MiB -> **34.77 MiB** (second cache gone)
  - still `free=39.1 MiB of 43,008 MiB`
  - 36,458,496 B / 17,664 B-per-tile = 2064 tiles x 128 tok = **264,192
    positions**, i.e. still the model's full context rather than the requested 512
  So the sizing defect is independent of the mode-mismatch defect and survives it.
- Next: forward the resolved `max_seq` into the server's load-frame params, then
  re-measure the batch sweep, which should stop being memory-bound entirely.
- Scope: Capacity / correctness — caps concurrency at 2 and blocks all
  multi-stream measurement
- Confidence: High (16x max_seq change with byte-identical allocation;
  positions x bytes arithmetic matches max_position_embeddings exactly)

## [High] Batched prefill OOMs on the 35B at batch=4 — blocks all multi-stream measurement
- Category: Correctness / Capacity (batched prefill)
- Measured 2026-08-10 on nix1 (gfx1103, 45.1 GB GTT) via `hipfire serve` +
  concurrent `/v1/chat/completions`, `--max-seq 512`, 32 max_tokens.
- Error, identical in every failing config:
  ```
  batch prefill: daemon generate_batch_prefill error: HipError(2): hipMalloc: out of memory
  ```
- **It is the model, not residency.** Discriminated three ways:

  | model | residency | batch 4 |
  |---|---|---|
  | `qwen3.5-0.8b--oq4++` | resident | **OK** — 22.3 aggregate tok/s, 5.6/stream |
  | `Qwen3.6-35B-A3B--oq4` | paged, 2 GiB budget | **OOM** |
  | `Qwen3.6-35B-A3B--oq4` | resident (paging off) | **OOM** |

  So the batch runner itself works; the 35B specifically cannot batch-prefill.
  Single-stream on the same 35B artifact is fine (13.9 tok/s warm), so this is a
  batched-path allocation, not model capacity.
- **Why it matters beyond the immediate ask:** every performance number in this
  investigation is `batch=1`, which is the worst case for the MoE amortization
  curve. The whole capacity argument for module-major execution
  (`docs/plans/2026-08-09-...` 0.4) rests on behaviour at N=16..128 streams, and
  right now that regime **cannot be measured on the target model at all**.
- Suspicion, unverified: `rocm-smi` reports VRAM total = 256 MB on this APU (the
  dedicated carve-out; the 45.1 GB is GTT). An allocation that must land in real
  VRAM rather than GTT would OOM almost immediately and would scale with batch.
  Worth checking which allocation in the batched prefill path is not GTT-backed
  before assuming the sizes are simply too large.
- **CHASED 2026-08-10. Not a MoE bug at all — it is KV allocation, and there are
  two distinct walls.** `HipRuntime::malloc` now names its size, and
  `HIPFIRE_MALLOC_BACKTRACE=1` names the caller.
  - **Wall 1 — fp32 KV allocates 258 MiB in a single chunk.**
    ```
    hipMalloc(270532608 bytes = 258.00 MiB): out of memory
      0 hip_bridge::ffi::HipRuntime::malloc
      1 hipfire_rdna::pool::GpuPool::alloc
      3 hipfire_rdna::dispatch::Gpu::zeros
      4 hipfire_runtime::kv::KvCache::alloc_k_v_filtered
      7 hipfire_serving_core::session::qwen35_allocate_session_state
      9 run_generate_batch_prefill_serial_qwen35
     11 hipfire_daemon::handlers::batch::prefill
    ```
    The 19 GB model loads fine because it is many per-tensor allocations; this is
    the first single buffer to cross the line. `rocm-smi` reports the dedicated
    VRAM pool as exactly 256 MiB on this APU, so a 258 MiB request cannot be
    served from it.
    **`--kv-cache q8` clears this wall** (confirmed: log shows `KV cache: Q8`, the
    258 MiB OOM disappears). Note `AGENTS`-adjacent prior art: fp32 KV also forces
    per-token prefill, so it was never the right mode for batching anyway.
  - **Wall 2 — the batch path CLONES the KV cache per session.** With Q8 it gets
    further and then fails at:
    ```
    failed to create checkpoint qwen35-checkpoint:batch-...:
      clone qwen35 checkpoint kv.k_gpu[3] alloc:
      hipMalloc(71860224 bytes = 68.53 MiB): out of memory
    ```
    `kv.k_gpu[3]` is one layer's K tensor, so a checkpoint clone costs roughly a
    second full KV cache per session (10 KV-carrying layers x k and v). That is
    the real capacity model of batched prefill on this arch and it is not
    documented anywhere.
- **Answered 2026-08-10: the clone is UNCONDITIONAL, and it is not for batching
  correctness.** `run_generate_batch_prefill_serial_qwen35`
  (`qwen35_prefill.rs:1925`) ends with a bare loop:
  ```rust
  for session in &result.sessions {
      ...
      emit_qwen35_prefill_checkpoint(m, gpu, arena_backend, hook)?;  // no guard
  }
  ```
  and `emit_qwen35_prefill_checkpoint`'s own doc says what it is for: emitting a
  boundary "so clients can resume from a cached prefix". That is **prefix
  caching** — a feature — and `clone_gpu_tensor` implements it as a deep
  device-to-device copy "to snapshot session state without aliasing the live
  buffers". So peak batch-prefill memory is ~2x KV per session, paid whether or
  not any client ever resumes.
- **Recommended fix, in order of increasing scope:**
  1. Make the checkpoint opt-in per request (clients that will not resume should
     not pay for the snapshot). Smallest change, unblocks the batch sweep.
  2. Make it lazy / copy-on-write — snapshot only when the live KV is first
     mutated past the boundary.
  3. Leave it and document 2x KV as the batch capacity model, which caps batch
     width at roughly half what the KV budget suggests.
  This is a **semantics** change, not an allocation fix: prefix-cache resume is
  observable behaviour clients may depend on, so it wants a decision rather than
  a patch.
- Also noted in passing: the function is named `..._serial_qwen35` and prefills
  the batch's sessions in a loop. Whether "batched prefill" is actually fused
  across sessions on this path, or serial-with-a-batch-envelope, was not
  established and is worth checking before any throughput conclusion is drawn
  from it.
- Instrumentation landed with this: `HipRuntime::malloc` reports the requested
  size on failure, and `HIPFIRE_MALLOC_BACKTRACE=1` captures the allocating
  stack. The bare "hipMalloc: out of memory" this started from could not
  distinguish a sizing bug from pool placement from genuine pressure.
- Scope: Capacity — blocks the multi-stream half of the v2 thesis
- Confidence: High (three-way discrimination, identical error)

## [FIXED 2026-08-10] Paged Opus MoE on the 35B wedged the GPU (MES hang → driver reset)
**Resolved — paged Opus MoE now generates on the 35B.** The pointer table went
`non_null=0/256` → **`8/256`** (exactly the selected experts, real device
addresses), the reset counter held at **39 → 39** where every prior attempt cost
a reset, and the model produced coherent output. Same selected expert IDs as the
failing runs (132, 129, 38, 213, 21, 253, 244, 193), so those are precisely the
slots that used to read null.

The fix is five pieces, and four of them were correct-but-unreachable for several
commits because of the fifth:
1. **Push residency** — the pager maintains the table itself on page-in, evict and
   teardown, so no dispatch site can forget (`f8f0902b3`).
2. **Registration** — every MoE layer hands its tables to the pager at load
   (`517c9c60a`).
3. **`ExpertResidency` reshaped** to residency-only; patching is the pager's job.
4. **Readback + ensure** in the lowered path — it now learns which experts to page
   in. A D2H sync per MoE layer, accepted deliberately; option (b), speculative
   paging via the unused `CpuRouter`, is what removes it.
5. **Threading** — `Qwen35Bindings` → `run_moe` → `moe_ffn_dispatch`. That last
   helper **hardcoded `None` for the pager**, discarding it before
   `moe_ffn_decode_impl` could build a provider. Everything above it worked and
   was simply never reached.

Regression: tiny-quant 188 pass / 3 fail — the same pre-existing `oq4.25++` trio,
two byte-identical and `zaya` moving inside its documented flake band. Resident
path unaffected. Workspace 98 targets / 0 failures.

Kept below as the investigation record; the measurements are still accurate for
the state they describe.

<details><summary>Original entry and investigation</summary>

- Category: Correctness / Stability (paged residency, prefill)
- Measured 2026-08-10 on nix1 (gfx1103) at `b30c13d4d`, daemon rebuilt at HEAD.
- Repro: `HIPFIRE_QWEN35_PAGED_EXPERTS=1 HIPFIRE_QWEN35_EXPERT_CACHE_MB=16384
  HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`, load `Qwen3.6-35B-A3B--oq4.hfq`, generate
  anything. Load succeeds in ~8 s; the first generate never returns a token.
- **The kernel driver is the ground truth:**
  ```
  amdgpu: MES failed to respond to msg=REMOVE_QUEUE
  amdgpu: failed to remove hardware queue from MES, doorbell=0x1002
  amdgpu: MES might be in unrecoverable state, issue a GPU reset
  amdgpu: GPU reset(36) succeeded!
  [drm] device wedged, but recovered through reset
  ```
- **Userspace stack at the hang** (daemon launched as a gdb child —
  `ptrace_scope` forbids attaching, and no sysctl was changed):
  ```
  #0  rocr::core::InterruptSignal::WaitRelaxed        <- busy-poll
  #2  hsa_signal_wait_scacquire
  #3  rocr::AMD::AqlQueue::ExecutePM4
  #4  rocr::AMD::GpuAgent::InvalidateCodeCaches
  #8  hsa_executable_freeze
  #16 hipModuleLoad
  #18 hipfire_rdna::dispatch::Gpu::ensure_kernel
  #19 hipfire_rdna::dispatch::misc::..::deinterleave_f32
  #20 Qwen35Bindings::run_attend
  #22 forward_scratch_layers_lowered
  #25 forward_prefill_batch_with_pbs_opts             <- PREFILL, not decode
  #27 hipfire_serving_core::generate::generate
  ```
  It hangs in the FIRST-USE JIT load of `deinterleave_f32`, waiting on a PM4
  completion the wedged GPU never signals.
- **This retires the "~160 s per MoE layer with one core pinned" reading.** That
  symptom is `WaitRelaxed` busy-polling in an active wait state — HSA spinning on
  a dead GPU — **not** host-side repack work. One core at 100% with the GPU idle
  looked like CPU-bound dequant and is nothing of the kind.
- **Consequently the qt-53 `Oq4G256MoeBlocks` artifact does not help here, and the
  measurement says so.** Both arms are identical:

  | artifact | load | gen wall | cpu_frac | tokens |
  |---|---|---|---|---|
  | `--oq4.moeblocks.hfq` (qt 53, repack-free) | 7.6 s | 1846 s | 1.00 | **0** |
  | `--oq4.hfq` (canonical, repacks per page-in) | 8.9 s | 569 s (killed) | 1.00 | **0** |

  `module_requires_host_repack` correctly excludes qt 53 (verified), so the packed
  arm genuinely skips the repack — and hangs the same way. The repack was never
  the bottleneck. qt-53 remains the right storage shape for a pager; it simply
  does not address this defect.
- **Scope limit — do not over-read this.** Only the PAGED path was measured. A
  non-paged 35B was not tested, so "Opus MoE on the 35B is broken" is NOT
  established; "paged Opus MoE prefill wedges the GPU" is.
- Related: `AGENTS.local.md` records a gfx1103 hazard of exactly this shape
  (page-fault → MES hang → full reset). A memory note claiming that hazard was
  nullified should be treated as suspect until re-checked.
- **BISECTED 2026-08-10: the wedge REQUIRES `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`.**
  Same artifact, same paged config, flag dropped: **no wedge**. The authoritative
  check is the driver's own counter — `dmesg | grep -c 'GPU reset(.*) succeeded'`
  was **36 before and 36 after**, versus a fresh reset on every flagged run.
  - flag ON:  load 8.9 s, generate spins 569–1846 s at `cpu_frac` 1.00, 0 tokens,
    GPU reset.
  - flag OFF: load 8.8 s, generate exits in **0.1 s**, 0 tokens, **no reset**.
  So the fault is specific to the **indexed OQ routed path**, not to paged
  residency in general and not to Opus weights in general.
- **This resurrects the possibility I was careful not to rule out.** When the
  loader repack bug was fixed, I recorded that it did NOT show the `*_k8_indexed*`
  kernels to be numerically correct, because the tiny fixture is top-2 and never
  executes them. The 35B is **top-8 and does** — and it wedges the device. Those
  kernels are now the prime suspect rather than an exonerated one.
- **Refinement on the stack:** the gdb frame shows where the process NOTICED the
  dead GPU (the first `hipModuleLoad` after the fault, in `deinterleave_f32` under
  `run_attend`), not where it faulted. The fault is upstream, in whatever the
  indexed path dispatched before it. Do not read `deinterleave_f32` as the culprit.
- **CULPRIT IDENTIFIED 2026-08-10 — `gemv_oq4g256_moe_gate_up_k8_indexed_batched`.**
  Found in ONE run with `HIPFIRE_LAUNCH_TRACE=2` (added for this; it synchronizes
  after every launch so an async fault is attributed to the kernel that caused it
  instead of its successor):
  ```
  [launch] gemv_oq4_grouped                              <- sync ok
  [launch] scaled_add_inplace_gpu_scalar_f32             <- sync ok
  [launch] gemv_oq4g256_moe_gate_up_k8_indexed_batched grid=[1024,8,1] block=[32,1,1]
  [launch] gemv_oq4g256_moe_gate_up_k8_indexed_batched <- SYNC FAILED code=719
                                             hipDeviceSynchronize: unspecified launch failure
  ```
  - HIP **719** = in-kernel memory fault. Reset counter 36 -> 37 confirms the wedge.
  - **28 launches succeeded first, and this kernel had 0 prior successes** — it
    faults the very first time it runs, not after N iterations.
  - Immediately prior: `gemv_oq4_grouped` (dense OQ4) and `mq_rotate_x` both fine,
    so OQ4 weights and the rotation basis are not implicated. It is specifically
    the indexed routed gate_up.
- **Two structural explanations tried and DISPROVEN — do not re-litigate:**
  1. "batched prefill never ensures expert residency": `prefill_batch.rs` neither
     dispatches these kernels nor touches the pointer table (0 references each).
     The stack goes `forward_prefill_batch` -> `forward_scratch_layers` -> lowered
     -> `run_moe`, so `moe_decode.rs` is the dispatcher.
  2. "residency is not ensured on the paged path": `ensure_paged_experts_resident`
     is called unconditionally at `moe_decode.rs:1187` whenever
     `ffn.experts.is_empty()`, i.e. before the dispatch at :1256+. And
     `patch_expert_ptr_table` (`weight_pager.rs:1397`) *panics* on a non-resident
     module, so a missing expert would abort, not fault the device.
- **ROOT CAUSE, measured 2026-08-10: the expert pointer table is never populated.
  `patch_expert_ptr_table` is DEAD CODE — it has zero call sites.**
  Host-side dump immediately before the faulting launch
  (`HIPFIRE_MOE_PTR_TABLE_DUMP=1`, added at the dispatcher):
  ```
  [ptr_table] oq4 gate_up batched (pre-launch): n_exp=256 non_null=0/256 k_top=8 batch=1
  [ptr_table]   slot=0 expert=132 ptr=0x0000000000000000  <== NULL
  [ptr_table]   slot=1 expert=129 ptr=0x0000000000000000  <== NULL
  ...  all 8 selected slots NULL
  ```
  - **0 of 256 slots non-null** — not just the selected ones; the table was never
    written at all.
  - The **top-k indices are valid** (132, 129, 38, 213, 21, 253, 244, 193, all
    within `0..n_exp`), so the GPU-side top-k is fine. The kernel dereferences a
    null pointer, which is the 719 and the MES hang.
- **Why nothing populates it — CORRECTED 2026-08-10, my first reading conflated
  two functions:**
  - There are **two** patch functions. `patch_expert_ptr_table` (weight-level,
    keyed on `resident`) genuinely has **zero callers**. But
    `patch_expert_module_ptr_table` (module-level, keyed on `resident_modules`)
    has **two**: `ensure_paged_experts_resident` (`moe_decode.rs:467`, under a
    `Phase::PatchPtrTable` timing span) and an example.
  - So the earlier claim "the only function that would patch it has zero call
    sites" was **wrong**, and so was "paged + indexed MoE has never worked on any
    path". The patching machinery exists and is wired — on the qwen35 HAND decode
    path and on `prefill_chunk.rs`. Paged+indexed would work there.
  - The defect is narrower and entirely about *reachability*: the **lowered
    super-op pipeline is default-ON and calls neither**
    `ensure_paged_experts_resident` nor any patch function, and its `MoeParams`
    carries no pager. So in the default configuration the table is never written,
    which is what the `non_null=0/256` dump measured.
  - `ensure_paged_experts_resident` has exactly two callers —
    `moe_decode.rs:1187` (qwen35 hand decode) and `prefill_chunk.rs:855` (chunked
    prefill) — and **neither is the lowered pipeline**, which is default-ON and is
    what actually runs (`dispatch_super_op` is in the fault stack).
  - The lowered pipeline's `MoeParams` carries **no pager field at all**, so that
    path structurally cannot patch the table even if it tried.
- **Two doc comments describe this wiring as if it exists** and should be corrected
  along with the fix: `qwen35/layout.rs:90` ("the indexed kernels read pointers
  from `expert_*_ptrs` which the pager patches per-token via
  `patch_expert_ptr_table`") and `:253` ("The forward path uses interior
  mutability (`borrow_mut`) at the MoE dispatch site to call `ensure_resident` /
  `patch_expert_ptr_table`"). Neither call happens.
- **So paged + indexed MoE is broken on the DEFAULT path** (lowered pipeline),
  though it is wired on the hand decode path and `prefill_chunk`. The earlier
  "~160 s per MoE layer" reading was this same null-pointer wedge all along.
- **Fix is a design decision, not a patch** — either thread pager access into the
  lowered MoE path so residency + table patching happen where the dispatch does,
  or refuse the paged+indexed combination in backend *selection* with a named
  predicate (the repo rule for exactly this class).
  - **B (refuse) LANDED** `6a4e32b68`: `check_moe_decode_supported` case (c).
    Verified on the 35B — reset counter 39 -> 39, no wedge, named refusal.
  - **A (residency) seam LANDED** `b60ffaad7`: `ExpertResidency` trait in
    `hipfire-dispatch` + `MoeParams::expert_residency`. A trait rather than a
    pager field because `hipfire-runtime` depends on `hipfire-dispatch`, so
    holding a `WeightPager` here would be a dependency cycle. No provider is
    wired yet, so behaviour is unchanged and the GPU stays safe.
- **Before implementing the provider, decide pull vs push — the codebase already
  has a documented position (checked 2026-08-10).** The trait as written is a
  PULL model: the caller passes `selected`, which needs the top-k on the host,
  which needs a D2H sync per MoE layer per token. The lowered path
  **deliberately avoids exactly that**. `pipeline/mod.rs` already contains a
  `memcpy_dtoh` of `topk_indices`, but it is gated on
  `moe_router_histogram_active()` and carries the comment: *"unlike the fallback,
  this path keeps top-K on-device — recording costs a per-token device->host copy,
  so it must stay off by default."*
  So a pull-model provider would make unconditional a cost the lowered path
  currently treats as opt-in telemetry, at ~48 syncs per march on a 40-layer
  model. The v2 plan flags the same readback as "the most likely place the design
  underdelivers".
  - **The push alternative avoids it entirely:** register the pointer tables with
    the pager once at load, and have the pager write a slot whenever it pages a
    module in (and null it on eviction). No host-side top-k, no per-token sync,
    and no pager state in `MoeParams`. Open question is whether every residency
    mutation — crucially eviction — can be funnelled through one place.
  - If push is chosen, `ExpertResidency` should be re-shaped (or dropped) before
    a provider is written against the current pull signature.
- **Separate defect found by the flag-off run — FIXED 2026-08-10.** The refusal
  worked, and then killed the daemon. `generate.rs` now reports it and unwinds
  like the other fallible steps in that function, so the client receives
  `{"type":"error","message":"prefill failed: unsupported moe.decode-..."}` and
  the worker survives. Verified: daemon exit 101 (panic) -> 0, zero panics.
  Sizing note: an earlier revision of this entry called it a signature/caller
  refactor because `generate()` returns `()` and `?` is unavailable. That was the
  wrong conclusion from a right fact — the function already has a
  `write_error(...)` + `qwen35_restore_or_error(...)` + `return` idiom for exactly
  this, so the fix is local. The original defect:
  ```
  thread 'main' panicked at crates/hipfire-serving-core/src/generate.rs:2979:18:
  called `Result::unwrap()` on an `Err` value:
    "unsupported moe.decode-routed-dtype-unsupported-no-fallback"
  ```
  That is the guard added at `qwen35/moe_decode.rs` firing exactly as intended —
  paged + non-indexable + no resident experts — but it reaches an `.unwrap()`, so
  a legitimate "this configuration is unsupported" becomes a **process panic**
  instead of an error frame to the client. The client sees the socket close.
  This is a concrete, high-impact instance of the `[Low]` opportunistic-unwrap
  entry below; there are 9 `unwrap()`s in the surrounding region, so fixing it is
  a signature/caller change rather than a one-line edit. **Not** attempted here.
- Scope: Stability (wedges the device), blocks M5 on this box
- Confidence: High (kernel reset log + userspace stack + paired A/B)

</details>

## [FIXED] Routed OQ experts repacked for kernels that never ran — non-finite KLD
**Title corrected 2026-08-10.** This was filed as "indexed OQ MoE decode kernels
produce non-finite KLD". That was a **misattribution**: the kernels were never
dispatched at all on the fixture that reproduced it. Kept below because the
reproduction and the reasoning that led to the real cause are still useful.

- **Real root cause: two independent decisions about the same question, allowed
  to disagree.** The dispatcher admits the indexed MoE kernels through
  `use_gpu_topk`, which requires `k_top == 8` — every one of them is a
  `*_k8_indexed*` kernel launched with `grid.y = 8`. The loader
  (`load_moe_expert`) repacked routed OQ experts from the canonical block layout
  (OQ4 130 B `[f16 scale|128 nib]`, OQ8 258 B) into those kernels' layout
  (132 B / 260 B, f32 scale) on the strength of `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`
  **alone**.
  So on a model routing top-k != 8, the experts were rewritten into kernel layout
  and then decoded by the NON-indexed fallback, which reads them as canonical
  blocks — every group misaligned by 2 B and the scale reconstructed from
  `[f16 scale | first 2 payload bytes]`, which is trivially huge or NaN. Hence
  non-finite, on every Opus cell, in both the oq4 and oq8 families.
- **Why the tiny fixture reproduced it and nothing else did:** it is
  `hidden=256, moe_intermediate=128, num_experts=8, num_experts_per_tok=2`.
  Top-2, so `use_gpu_topk` is false and the fallback runs. MiniMax and the 35B
  route top-8 and stay on the indexed path, where the repack is exactly right —
  which is why arch 10 reports finite KLD through the same kernels.
- **Fix:** `INDEXED_MOE_K_TOP` (`qwen35/mod.rs`) is now the single definition of
  the 8, read by both `use_gpu_topk` and the loader; `load_moe_expert` takes the
  decision as a parameter instead of re-deriving it from the env.
- **Verified:** `HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`
  goes 7/7 non-finite -> **7/7 pass**, drift `-0` on six cells, `findings: 0`.
  Control (flag unset) still passes. Workspace lib suite 98 targets / 0 failures.
- **What this does NOT establish.** With `k_top=2` the indexed kernels still do
  not execute, so this is NOT evidence that they are numerically correct — it
  only removes the corruption that was being blamed on them. The tiny fixture
  cannot exercise them at all. Exercising them needs a top-8 MoE fixture, which
  the tiny battery does not currently have.
- **Latent hazard this surfaced — NARROWED, then fixed (2026-08-10).** The paged
  arm of that same guard is `use_gpu_topk || ffn.experts.is_empty()`, so a paged
  model takes the indexed path regardless of its top-k while the kernels
  hard-code `grid.y = INDEXED_MOE_K_TOP`. A paged top-k != 8 model would read
  `topk_indices` past its end.
  - **First filed as unqualified; that was too broad.** The lowered/super-op
    executor — which is the DEFAULT path — already refuses this at
    `hipfire-dispatch/src/pipeline/mod.rs:302` via `check_moe_decode_supported`
    (`!use_gpu_topk && !routed_experts_resident` →
    `decode-routed-dtype-unsupported-no-fallback`), and
    `coverage_tests.rs:446` already asserts exactly that case errors. So the
    exposure was only ever the **qwen35 arch hand path**, which had zero calls to
    that predicate, and is reached only through the four documented escapes
    (hidden-state ring, GDN tape capture, `HIPFIRE_RQ_HAND=1`,
    `hipfire_steer::is_active()`).
  - **Fixed** by calling the same `check_moe_decode_supported` from
    `qwen35/moe_decode.rs` before that branch, rather than adding a third copy of
    the rule. Costs nothing on supported paths: resident experts satisfy it, and
    paged + top-8 sets `use_gpu_topk`.

<details><summary>Original entry as filed (kept for the reproduction)</summary>
- Category: Correctness / Kernels (Opus routed experts)
- Location: `gemv_oq4g256_moe_*`/`gemv_oq8g256_moe_*` indexed kernels, gated by
  `qwen35_moe_oq_indexed_decode_enabled` (`hipfire-arch-qwen35/src/qwen35/mod.rs:1824`)
- Summary: with `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`, ALL SEVEN Opus cells of the
  `qwen3_5_moe` tiny fixture fail with **non-finite KLD** (oq4, oq8, oq4+, oq4++,
  oq4.25++, oq8+, oq8++). With the flag unset the same seven **pass** with drift
  `-0` against baseline. One variable, opposite verdicts.
- Reproduce in minutes:
  `HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe HIPFIRE_QWEN35_MOE_OQ_INDEXED=1 ./tests/tiny-quant-gate.sh`
- Why it matters beyond the flag: **paged routed-expert residency REQUIRES this
  path.** Under paging `routed_experts_resident` is false by design, so
  `check_moe_decode_supported` admits only the GPU-top-K indexed route. Paged
  Opus MoE is therefore blocked on this defect — on a 35B it presents as ~160 s
  per MoE layer with one core pinned, which is how it was found.
- Non-indexed Opus MoE decode is healthy; this is specific to the indexed routed
  kernels.
- **RE-VERIFIED on the rebased base 2026-08-10, and the result is now STRONGER.**
  The original A/B was taken on `da38cc16f`, which also carried the `from_flag`
  wildcard defect — so an obvious objection was that the three OQ8-family cells
  failed because `--format oq8` was being rejected outright, not because of any
  kernel. That objection is now dead: on `origin/master` `33d9dcbd2` the wildcard
  is fixed and all seven Opus cells **still** fail with non-finite KLD, while the
  same seven pass with the flag unset on that same base. 82 commits of master work
  did not touch it.
  - control (flag unset): oq4, oq8, oq4+, oq4++, oq4.25++, oq8+, oq8++ — 7/7 pass
  - `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`: the same 7 — 7/7 non-finite KLD
  - `q8f16`, `mq3`, `mq4`, `mq6` pass in both arms, but that is weak evidence:
    the flag is OQ-specific and is not expected to reach the MQ kernels.
  So the defect is in the indexed OQ routed kernels themselves, spanning **both**
  the oq4 and oq8 families, and it is not collateral from the OQ8 flag bug.
  **^ That last sentence was WRONG** — a clean one-variable A/B told me the flag
  caused the failure, and I read that as "the kernels the flag enables are
  broken". The flag also switches the *loader*, and that is what broke. A
  one-variable experiment localises the variable, not the mechanism.
- Suggested fix: debug on the tiny fixture, not a 35B. The gate comment already
  describes this as a known "finite-KLD failure" being debugged, so this entry is
  a reproduction and a scope statement rather than a new discovery.
- Scope: Correctness (premier quant family, paged path)
- Confidence: High (clean A/B, one variable)
</details>

## [RESOLVED by rebase] tiny-quant was RED for Opus across four MoE families
- Category: Correctness / Quant (Opus)
- Location: `tests/tiny-quant-gate.sh` cells; baselines in `tests/tiny-quant-baselines.txt`
- Summary: `./tests/tiny-affected-gate.sh --require-coverage` fails 14 cells, all
  Opus, on deepseek4, deepseek4_compressed, lfm2_moe and minimax. Worst is
  deepseek4 `kld:oq8` at KLD **0.038652 vs baseline 0.000193** — ~200x, against a
  RELATIVE budget of 25% of baseline (±0.000048). deepseek4_compressed oq8 is
  0.038414 vs 0.000107. minimax oq4/oq4+/oq4++/oq4.25++ drift ~2-3x. Two cells do
  not drift but hard-fail the quantizer: `lfm2_moe` and `minimax`
  `quantize:oq8++(calib)` exit 1 with "calibrated plus format requested, but no
  LDLQ-eligible tensors were attempted".
- Pre-existing, NOT from any change on the current branch: verified by reverting
  the branch's quant-format/pager/quant.rs edits to their parent and re-running
  the identical gate with the identical `--files-from` list — the failure sets are
  BYTE-IDENTICAL (same 14 cells, same drift values).
- Why it was not caught sooner: `tiny-affected-gate` selects a family allowlist
  from the touched paths, so a green run means "the SELECTED families passed", not
  "the suite passed". Runs that touched only qwen35 paths never selected these
  four families. Comparing two gate runs with different `--files-from` inputs
  compares different tests.
- **RESOLVED as to cause, AND ALREADY FIXED ON `origin/master` (corrected 2026-08-10
  after actually fetching).** Do not re-bisect this, and do not re-fix it.
  - `origin/master` advanced `da38cc16f` → `33d9dcbd2` (82 commits) during this
    work, and the reordered `from_flag` is there now, with a comment pinning why
    the catch-all must stay last.
  - **This branch is based on the stale `da38cc16f` and still carries the bug.**
    `cargo build -p hipfire-quantize` emits `unreachable pattern` at
    `main.rs:3924`/`:3925` — the `"oq8"` and `"oq8+" | "oq8++"` arms. The remedy is
    a rebase onto `origin/master`, not a code change here.
  - **Therefore every tiny-quant number measured on this branch was taken on a base
    with a known, since-fixed OQ8 defect.** One-variable A/Bs run on this branch
    remain valid as *relative* results (both arms share the base); absolute cell
    verdicts do not.
  - The analysis below was first written on the local-only branch
    `fix/oq8-from-flag-and-rotation-guards`
    (`docs/plans/moe-expert-residency-unification.md`, Phase 0, `691e7730e`); that
    *document* is on no origin ref, but its fix reached master by another route.
  - **9 of the 14 are one real regression**: deepseek4 ×3, deepseek4_compressed ×3,
    lfm2_moe ×3. Green at `0060481ee`, red at `8b9ee5392`.
  - **Root cause**: `8b9ee5392` inserted an unguarded `_` wildcard arm *above* the
    `oq8` literals in `HfqInputFormat::from_flag`
    (`hipfire-quantize/src/main.rs:3919`), replacing a guarded
    `_ if parse_opus_mixed_format(flag).is_some()`. That made `"oq8"`/`"oq8+"`/
    `"oq8++"` unreachable, so `from_flag` returned `None` for every OQ8 flag and
    each call site degraded differently — silently skipped tensors (lfm2_moe
    scoring KLD exactly 0.000000), a wrong-format fallback (deepseek4 at 0.0387),
    or the "no LDLQ-eligible tensors were attempted" hard error. `oq4` was
    unaffected because its arm sits above the wildcard. **rustc reported this as
    `unreachable_patterns` — the warning was the diagnosis.**
  - **The remaining 5 are minimax and are genuinely pre-existing**, failing
    identically at the baseline-record commit `5dc01e4b0`. Fixing the wildcard
    *unmasks* minimax rather than fixing it: its oq8 cells were passing only
    because the quantizer was not producing OQ8 at all. Cross-referenced to
    `2026-08-05-opus-across-model-families.md:82-93` (oq4 0.003531, oq8 0.000259).
    **minimax cannot serve as a parity oracle**; scope it out with
    `HIPFIRE_TINYQUANT_FAMILIES` until that separate cause is found.
- **DONE — rebased 2026-08-10, and the 14 cells are gone.** On the rebased base
  the gate is **188 pass / 3 fail**. deepseek4, deepseek4_compressed, lfm2_moe
  **and minimax** now pass every cell.
- **A prediction recorded here was wrong, and the reason matters.** The Phase 0
  analysis predicted 7 survivors, all minimax, on the theory that fixing the
  wildcard *unmasks* minimax rather than repairing it. Minimax passes clean. That
  analysis was performed on a different tree, and the rebase brought 82 commits of
  master work, not just the one-line reorder — so it was a hypothesis about
  another base, not a forecast for this one. Do not carry cross-base predictions
  forward without re-measuring.
- Scope: Correctness (premier quant family)
- Confidence: High (byte-identical reproduction on pristine code)

## [Medium] tiny-quant: three `oq4.25++(calib)` cells breach budget — two on the GOOD side
- Category: Correctness / Quant (mixed Opus) + gate tolerance design
- Location: `tests/tiny-quant-gate.sh`; baselines in `tests/tiny-quant-baselines.txt`
- Summary: on `origin/master` at `33d9dcbd2` the gate is **188 pass / 3 fail**, and
  all three failures are the same cell, `kld:oq4.25++(calib)`:

  | family | measured | baseline | budget | direction |
  |---|---|---|---|---|
  | qwen3_legacy | 0.004369 | 0.005979 | ±0.001495 | **better** by 0.00161 |
  | gemma4_moe | 0.005952 | 0.003077 | ±0.000769 | **worse**, ~1.9x |
  | zaya | 0.000023 | 0.000036 | ±0.000010 | **better** by 0.000013 |

- **"3 failures" overstates it.** Only `gemma4_moe` is a real degradation. The
  other two are *improvements* that trip a **symmetric** relative tolerance —
  the gate flags movement, not loss. `zaya` is the clearest case: at absolute
  magnitudes of 2e-5, a 25% budget is ±1e-5, so almost any change trips it.
- Pre-existing, NOT from the v2 branch: verified by running the identical three
  families in a detached worktree at pristine `origin/master` — the failures
  reproduce with byte-identical measured values, baselines and budgets.
- **`zaya` is FLAKY, not failing (observed 2026-08-10).** A later full-gate run on
  the same commit reported **2** failing cells, not 3: `zaya` passed. Nothing in
  that path changed between runs. This is the predicted consequence of scoring a
  2e-5 cell against a ±25% relative budget (±1e-5) — the cell flips on ordinary
  run-to-run variation. Treat `zaya/oq4.25++` as a tolerance defect, and do not
  read a single green run as having fixed it.
- **Do not re-record baselines from one run.** `--record` would bake in whichever
  side of the flake that run landed on, and would also silently absorb the real
  `gemma4_moe` regression.
- Also note the gate's `findings: N` counts **skips as well as failures** — a run
  showing 9 findings here is 2 fails plus 7 explicitly blocked `deepseek4_mtp`
  cells. Read the `fail` lines, not the findings count.
- Two separable actions: (a) investigate `gemma4_moe` oq4.25++ as a genuine ~1.9x
  mixed-Opus regression — it is the one cell that is reproducibly worse; (b) give
  the KLD budget an absolute floor (and consider making it one-sided) so near-zero
  cells stop flipping. Re-recording the baselines would hide (a) — do (a) first.
- Scope: Correctness (mixed Opus) + gate design
- Confidence: High (byte-identical reproduction at pristine origin/master)

## [RESOLVED] Quantized-from-HFQ artifacts lose config/tokenizer (dangling v2 tail pointer)
- Category: Correctness / Tooling (hipfire-quantize)
- Location: crates/hipfire-quantize/src/main.rs `HfqInputFile::open`
- Root cause: an HFQ v2 source keeps `config` / `tokenizer` / `tokenizer_config`
  / `generation_config` / `gguf_meta` in a TAIL blob addressed by a
  `tail_metadata` = `{offset, size, hash}` pointer in the FRONT metadata, where
  `offset` is a byte offset into that source file. `HfqInputFile::open` read only
  the front JSON (stopping at brace depth 0) and never dereferenced the tail, so
  the quantizer forwarded the front metadata verbatim to the derived artifact
  (main.rs ~L4705 builds the output metadata from `hfq.metadata_json`). The
  forwarded `tail_metadata.offset` then points PAST the (smaller) derived file,
  into the original source — dangling. Result: every bf16→oq/mq artifact loaded
  with NO config and NO tokenizer (`Tokenizer::from_hfq_metadata` → "tokenizer |
  gguf_meta" missing; and `config_from_hfq` → "failed to parse config"). Hit on
  all four MiniCPM5-1B.oq* variants produced 2026-07-27.
- Fix: `merge_source_tail_metadata` resolves and inlines the source tail into the
  front metadata at open time (mirrors the runtime's `merge_tail_metadata`:
  read+hash-verify the tail blob, merge its `metadata` object with front-wins
  semantics), and strips the container-level `tail_metadata` / `hfq_format` keys
  so `hfq_out::write_hfq` regenerates a correct tail for the OUTPUT. No-op for v1
  or already-inlined sources. Unit tests: `merge_source_tail_*` (inline,
  front-wins, no-op, hash-mismatch). Weights are untouched — the bug was
  metadata-only.
- Note: existing broken artifacts were repaired out-of-band (tokenizer/config
  re-injected, weights byte-identical); with this fix, re-quantizing from a v2
  bf16 source now embeds them correctly at emit time.
- Confidence: High (root-caused, unit-tested; re-quant end-to-end recommended).
## [RESOLVED] hipfire-daemon inference worker killed on client disconnect (was theorized as "GPU fault under model-swap churn")
- TRUE ROOT CAUSE (confirmed): a client closing the socket mid-generation, NOT
  model-swap churn. On cancel, chat.rs `execute_blocking_chat_cancellable` hits
  `Ok(None)` and `drop(engine)`s the `DaemonEngine`; `StdioTransport` is spawned
  `kill_on_drop(true)`, so the drop SIGKILLs the whole worker (destroying the
  loaded model; recovery was a lazy reload / "Broken pipe"). Intermittent because
  the disconnect must land while generating; no dmesg trace because SIGKILL(9)
  leaves none. The "model-swap churn / GPU fault" below was a red herring from an
  aggressive repro hammer — real trigger is disconnect (e.g. a coding agent
  `pkill`ing a request, or a client timeout).
- FIX: cooperative cancellation. The daemon installs a SIGUSR1 handler that sets
  a process-global `GENERATION_CANCEL: AtomicBool` (async-signal-safe: atomic
  store only); the shared decode loop (`arch.rs decode_loop_with_timing_terminators`
  + the qwen35 / multi-GPU loops) checks it at loop TOP (KV-safe: identical to a
  natural max_tokens stop, drops only the un-written pending sample) and stops,
  emitting a normal terminal `done`. The frontend, on disconnect, sends SIGUSR1 to
  the worker (`DaemonEngine::abort_and_drain` → `libc::kill`) and drains to the
  (now-fast) terminal event, then RESTORES the engine instead of dropping it —
  worker + model stay resident. Verified on gfx1103: worker PID stable across
  10+ disconnects (was killed in 1–2), post-disconnect request in ~0.16–0.78 s
  (proves the gen was cancelled, not run to max on the serial worker), normal gen
  unaffected. NOT covered (fall back to the old drop): spec-decode (mtp/dflash)
  and VL loops (multi-token-per-iter, not provably KV-safe to break) — a
  follow-up. Related: the worker still does not auto-respawn; #204 added durable
  `~/.hipfire/daemon.log` + honest `degraded` status.
- --- earlier (INCORRECT) hypothesis, kept for the record ---
- Category: Reliability / Correctness (worker process)
- Location: hipfire-daemon worker (spawned by `hipfire serve` via
  crates/hipfire-daemon-adapter/src/lib.rs); GPU load/unload + decode path.
- Summary: The `hipfire-daemon` inference worker (a child of the `hipfire serve`
  front-end) dies intermittently. Reproduced under sustained model-swap churn on
  gfx1103: crash at req 308 (`MiniCPM5-1B.oq4.25++.coarse`, a 48-token decode)
  after 307 clean reloads; also observed under a single coding agent's light
  normal use. It is cumulative, not a leak (worker RSS flat), not a concurrency
  race (a `text_concurrency` limiter serializes loads), and not a plain OOM
  (45 GB GTT budget). Signature: GPU-state accumulation across many model
  load/unload cycles, tripping a fault on a subsequent decode. Exact fault line
  still UNCAPTURED at the time of filing — now capturable, see below.
- Two contributing DEFECTS, both FIXED (observability, PR
  `fix/daemon-crash-logging-and-worker-health`):
  1. The worker died SILENTLY — its stderr was only re-emitted to the front-end's
     (variably-routed) stderr and its death was a silent EOF `break`, so the
     backtrace/signal evaporated. FIX: set `RUST_BACKTRACE=1` on the worker, tee
     its stderr to a durable `~/.hipfire/daemon.log`, log an EOF death marker, and
     log the exit status/signal via `DaemonEngine::worker_alive`
     (`try_wait`). Verified: a SIGKILL now logs `signal: 9 (SIGKILL)` + a
     daemon.log marker.
  2. `hipfire status` reported `healthy` while the worker was dead — it only pinged
     the HTTP front-end. FIX: `/health` now probes the worker and reports
     `status:degraded` + `worker_alive:false`; `hipfire status` renders
     `degraded (inference worker down)` with a pointer to daemon.log.
- Also found (NOT fixed — follow-up): the worker does NOT auto-respawn. After it
  dies, requests fail with raw `Broken pipe (os error 32)` until `hipfire restart`.
  Needs: respawn-on-death + a clean "worker down" error (crash-handling, #3).
- Next: re-run under the durable log to capture the real signal/backtrace, then
  fix the GPU fault. Mitigation: the `*.coarse` variants load heavier FP32 KV
  (no `q8` override) which raises swap-churn stress.
- Confidence: High on reproduction + the two observability fixes; root cause of
  the GPU fault itself pending a captured trace.

## [RESOLVED] Batched prefill garbage for bf16/f16 llama models (was: "attention_q8_0_kv_batched masked prefill garbage for decoupled head_dim")
- Category: Correctness / Dispatch
- Location: crates/hipfire-runtime/src/llama.rs `forward_prefill_chunk`
  (QKV / wo / gate+up / down projection dispatch)
- Root cause (CONFIRMED, not the attention kernel): `is_batchable_la`
  (crates/hipfire-runtime/src/dispatch.rs L100 `bf16_f16_wmma` arm) marks BF16/F16
  weights batchable on every WMMA arch incl. gfx1103, so a native-bf16 llama model
  (e.g. MiniCPM5-1B.bf16) routes into `forward_prefill_chunk`. But the chunk's four
  per-linear projection blocks only had arms for the quantized formats
  (6bit / q8 / mq3 / fp4 / else=HFQ4). BF16/F16 matched NONE, so every projection
  fell through the `else` to `gemm_qkv_hfq4g256` / `gemm_*_hfq4g256_residual`, which
  reinterpret the raw bf16 weight bytes as 4-bit HFQ4 blocks — garbage from layer 0.
  Only q_dim≠hidden was incidental (MiniCPM happens to be both bf16 AND decoupled);
  the true trigger is the bf16/f16 weight dtype. The attention kernels
  (attention_q8_0_kv_batched generic + gfx1103), the Q8 KV write, and the KV read
  stride were all verified CORRECT. The original bisection couldn't isolate it
  because BOTH the mis-projection and the attention live inside path C and it only
  compared final logits; path B (`prefill_forward`) uses the dtype-dispatched
  `weight_gemm`, which handles bf16 — that's why B was clean.
- Fix: add BF16/F16 arms to all four projection blocks in `forward_prefill_chunk`,
  routing through `crate::weights::weight_gemm` (identical to the correct
  `prefill_forward` path). Verified with `debug_batched_prefill_divergence` on
  MiniCPM5-1B.bf16 / gfx1103: path C (flash/masked) cosine vs per-token reference
  went from **−0.194 → 0.99996**, argmax now matches at every prefix length
  (n=4/6/8). Regression guard: `chunk_projection_handles_dtype` helper + a
  `debug_assert` in the chunk + the no-GPU unit test
  `llama::tests::chunk_projection_covers_all_batchable_dtypes`, which asserts every
  dtype `is_batchable_la` accepts has an explicit projection arm (so batchability
  and projection coverage can't drift apart again).
- Serving route (measured, decided): with the projection fix BOTH batched
  prefill paths are correct, so the earlier `prefill_forward` route in
  LlamaBackend::prefill (crates/hipfire-arch-llama/src/arch.rs) is no longer a
  correctness workaround. A clean same-build A/B on gfx1103 / MiniCPM5-1B.bf16
  (`hipfire bench`, 5 reps) shows `prefill_forward` (attention_causal_batched) at
  pp512 **602.3 ± 2.4 t/s** vs the fixed chunked path (attention_q8_0_kv_batched)
  at **580.8 ± 2.3 t/s** — the `prefill_forward` route is ~3.6% faster (tg128
  identical, ~11.9 t/s). So it is RETAINED on perf grounds, and the chunk-path
  projection fix stands as correctness + the coverage guard. (The one-time "1227
  t/s garbage" figure was the mis-dispatched HFQ4 kernel reading half the bytes,
  never a real correct speed.)
- Confidence: High (root-caused + numerically verified end-to-end + benchmarked).

## [Low] Opportunistic .unwrap() → error-handling cleanup (convention, not a tracked bug)
- Category: Reliability / Maintainability
- Location: Project-wide (~6.8k non-test `.unwrap()` sites; most guard true
  invariants, not user input)
- Summary: Prefer `?`/descriptive `expect()` over bare `.unwrap()` on paths
  that can fail on user input or external files. This is a fix-as-you-touch
  convention, not a specific reproducible crash — a blanket sweep is neither
  feasible (6.8k sites) nor desirable (many unwraps encode real invariants).
- Named exemplars — both resolved (2026-07-21/22):
  - `hipfire-runtime/src/weights.rs`: 14 raw
    `unsafe { …as_ref().unwrap().buf.alias() }` rotated-scratch sites → one
    documented `Gpu::mq_x_rot_f32()` accessor (SAFETY comment + actionable
    `expect()`).
  - `hipfire-quantize/src/main.rs` `SafetensorsFile::open`: the model-load
    header parse (`from_utf8`/`from_str`/`from_value`/8-byte length) now returns
    clean `io::Error(InvalidData)` messages instead of panicking on a
    truncated/malformed `.safetensors` file.
- Confidence: Low (convention; no open crash tracked)

## [Closed] "Excessive" global state via OnceLock — intentional, not a defect
- Category: Architecture / Maintainability
- Location: crates/hipfire-arch-deepseek4/src/forward.rs (`mod env_cache`),
  crates/hipfire-rdna/src/dispatch/mod.rs, crates/hip-bridge/src/ffi.rs
- Resolution (2026-07-22): Investigated. The flagged `OnceLock`/`thread_local!`
  statics are a deliberate, documented hot-path optimization: they cache
  `HIPFIRE_*` env-derived debug/tuning knobs read once, because an uncached
  `std::env::var` per lookup cost ~200μs/token (43 layers × ~5 lookups × ~1μs
  syscall). They are set-once, read-only, and idiomatic. Converting them to
  injected config context would re-add that per-token cost (or require threading
  a config struct through the entire hot path) for near-zero benefit — these are
  debug/tuning knobs, not core mutable state. Not a bug.
- Residual guidance (minor): do not introduce globals for *core mutable state*
  or *user-facing config*; those belong in explicit context objects. Env
  debug/tuning knobs behind `OnceLock` remain the accepted pattern.

## [High] Stale SWA ring-buffer slots after speculative reject (post-wrap corruption)
- Category: Reliability / Correctness
- Location: crates/hipfire-arch-deepseek4/src/spec_decode.rs:224-233,401-428;
  read side kernels/src/deepseek4_attn_swa.hip; config `sliding_window=128`.
- Mechanism (code-confirmed 2026-07-22, no empirical run — see blocker):
  1. The draft/verify loop increments `state.n_tokens` per step so SWA K/V
     writes land IN THE REAL per-layer ring at draft positions N+1..N+K
     (spec_decode.rs:224-230). Slot index = `n_tokens % sliding_window`.
  2. On partial accept only `state.n_tokens` is restored (line 428); the ring
     DATA at the K−n_accept uncommitted slots is never invalidated.
  3. The decode SWA kernel reads slots `[0, n_valid)` LINEARLY with no
     per-slot position mask (deepseek4_attn_swa.hip) — it trusts n_valid.
  Result: PRE-wrap (total seq < 128) the stale slots sit at indices ≥ n_valid
  and are excluded → safe. POST-wrap (seq ≥ sliding_window=128) the ring is
  full; uncommitted draft writes evict positions still inside the next
  forward's 128-wide window, so the linear read consumes rejected-token K/V →
  silent attention corruption.
  Refined boundary (2026-07-22, from the verify/accept indexing): verify feeds
  `[last_token, draft[0..k-2]]` at base `last_position+1`, and the NEXT decode
  overwrites exactly ONE stale slot (the corrected token's, verify column
  `accepted_len`). So the still-stuck stale columns are `[accepted_len+1, k)`,
  nonempty only when **k ≥ n_accept+3** (never k=2; k=3 only at n_accept=0) AND
  post-wrap. Real but narrower than "any partial accept". Only the modular SWA
  ring aliases; `full_k_cache` is absolute-indexed + causally safe, and the MTP
  ring only affects draft acceptance (verify still guarantees correct output).
- Fix: IMPLEMENTED, gated OFF pending GPU validation. `spec_decode::swa_rewind`
  (behind `HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND=1`) snapshots the K soon-to-be-
  evicted main-layer SWA slots before the verify (strided per-slot copy into
  per-layer `swa_k_snap`/`swa_v_snap`) and restores the uncommitted columns
  `[accepted_len+1, k)` after the accept, wrap-aware. Pure slot arithmetic is
  unit-tested (`cargo test -p hipfire-arch-deepseek4 swa_rewind`, 4/4). Enable-
  by-default is blocked on an AR-vs-spec losslessness A/B on a runnable model:
  a compressor-F16 `deepseek4-q8-mtp` re-quant is in progress on halo (the mq4
  artifact is unloadable — see below). Validation: pre-fix expect divergence
  post-128 with k=3; post-fix expect token-identical.
- Empirical status (halo, gfx1151): BLOCKED. The only deepseek4 artifact on
  halo (`deepseek-v4-flash--mq4.hfq`) will not run on the current daemon build:
  its MQ4 `compressor.wkv` is rejected by the F16-native compressor path
  (`HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=1` default), and `=0` routes it to an
  unsupported `gemv.unknown`. Black-box AR-vs-spec-decode A/B needs a
  re-quantized compressor-F16 model first.
- Scope: Architectural
- Confidence: High on mechanism (code-confirmed); reproduction pending a
  runnable model.
- Note: The sibling `forward.rs` chunk/ring path is NOT affected — its
  non-aligned-with-compress-events case returns an explicit `Err`.

## [High] bf16 KLD reference artifacts contain chunk 0 replicated 1175×
- Category: Correctness / Evidence tooling
- Location: `/srv/hipfire/kldrefs/qwen3.5-{0.8b,2b,4b}-bf16.kldref.hfq`
  (and the `.arch0.bak` copy under `/srv/Public`); produced 2026-06-05 by
  `build_kld_ref_hipfire` (hipfire 0.2.0). That producer is no longer in the
  tree — only the artifacts remain.
- Summary: `kldref.tokens` is correct (1175 contiguous 2048-token windows of the
  wikitext2 slice), but the `kldref.top_indices` / `top_log_probs` /
  `residual_mass` blocks for EVERY chunk are byte-identical to chunk 0's. The
  block cursor never advanced. Verified with
  `cargo run --release -p hipfire-runtime --example kldref_selftest -- <ref>`:
  chunk 0's argmax agrees with the corpus's next token 44–50% of the time (a
  healthy bf16 reference), chunks 1..N agree ~1% (chance), and a slide of chunk
  1's blocks over the token stream best-matches token position 1025 — chunk 0's
  scoring window. All three model sizes show it identically, so it is a producer
  bug, not file corruption.
- Impact: any absolute KLD-vs-bf16 computed from these files past chunk 0 is
  meaningless (a candidate is scored against a different passage's predictions;
  observed ~11.5 nats/tok vs ~0.3 for the valid chunk). Chunk 0 alone (1023
  positions) IS usable, which is how the defect stayed invisible to spot checks.
  The daemon's own loader independently refuses these files — their metadata
  `arch_id` is 0 (`read_hfqm_kld_ref_archive`, `hipfire-daemon/src/main.rs`) — so
  the in-tree evidence path was never exposed; the risk is ad-hoc harnesses that
  bypass that check.
- Suggested fix: regenerate against a bf16 `.hfq` (none currently on disk for
  qwen3.5-0.8b; `/srv/hipfire/archives/models--Qwen--Qwen3.5-0.8B.hfa` holds the
  HF source) with a per-chunk block-cursor assertion, and have any new reader
  run the `kldref_selftest` agreement check before trusting a reference. Until
  then treat these three artifacts as single-chunk.
- Scope: Tooling / evidence integrity
- Confidence: High (self-test is deterministic and reproduces on all 3 files)
