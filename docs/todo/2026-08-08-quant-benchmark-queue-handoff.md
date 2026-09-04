# GOAL: finish the quant benchmark queue (unattended handoff)

> **STATE RE-CHECKED 2026-08-31 — the "in flight" section below is dead, and
> most of "Done so far" no longer exists on disk. Read this first.**
>
> - **Both background jobs are gone.** The scratchpad they lived in
>   (`.../1aa40aa2-.../scratchpad`) does not exist, so per this document's own
>   note the scripts are gone with it. Nothing is running; re-derive from
>   §"Recipe" if resuming.
> - **The expensive Qwen3.5-35B-A3B outputs are gone.** No bf16 (43 GB), no
>   calib (1.8 GB), no `oq4.25++` (17.9 GB) anywhere under `/srv`. Searched by
>   several name shapes; the only 35B artifacts present are Qwen3.**6**
>   (`Qwen3.6-35B-A3B--oq4.moeblocks.hfq`, `...-DFlash--bf16.hfq`) plus a
>   `qwen3.6-35b-a3b-bf16.kldref.json` whose `.hfq` was deleted as damaged (see
>   BUGS.md: 0% agreement on every chunk). So this is NOT "kld reference running
>   -> score" with three stages banked — it is a restart from the `.hfa`.
> - **What DOES survive is the source and the knowledge.**
>   `models--Qwen--Qwen3.5-35B-A3B.hfa` (47.7 GB) is on disk, and §1 is accurate:
>   `97a06db3f` landed, `03f7a2612` landed and was reverted by `dc18bdd39` —
>   all three verified present on master.
> - **The durable result is §1's finding, and it does not depend on any
>   artifact:** the per-expert LDLQ factorization is irreducible for calibrated
>   recipes, because AWQ rebases the Hessian per tensor and the damping is
>   applied after the rebase, so `(D H D + lambda I)^-1 != D (H + lambda' I)^-1 D`.
>   That is worth keeping whatever happens to the queue.
>
> **Not restarted unattended, deliberately.** Re-running this is hours of GPU on
> a box whose daemon is serving, and whether the benchmark queue is still worth
> that is a priority call, not a technical one. See `DECISIONS-PENDING.md`.

Written 2026-08-08 at the end of a long session. Everything below is state you
can act on without asking. Read "Traps" before running anything — several cost
hours this session.

## In flight right now

Two background jobs are running on `feat/daemon-state-hoist`:

1. **Qwen3.5-35B-A3B pipeline** — `$S/q35.sh` (resumable; skips finished
   stages by artifact existence). Stages: restore ✅ → bf16 ✅ → calib ✅ →
   quantize ✅ → **kld reference (running)** → score. Prints `Q35_DONE`.
2. **Kernel-trace diagnostic** — `$S/ktrace.sh`, waits for `Q35_DONE`, then a
   1-chunk reference build with `HIPFIRE_KERNEL_TRACE=1`. Prints
   `KTRACE_DONE`.

`S=/tmp/claude-1000/-home-sadara--hipfire-src/1aa40aa2-e843-49ed-b7b6-98eb29cf770a/scratchpad`
(if that scratchpad is gone, the scripts are gone — re-derive from §"Recipe").

Done so far for the 35B: bf16 43 GB on disk / 68.8 GB logical; calib 1.8 GB
(350 hessian + 20826 imatrix, 10238/10240 experts); **oq4.25++ 17.9 GB with
`LDLQ pooled: 10240`** (= 40 layers × 256 experts — the pooled donor resolved
on a qwen-style MoE via the router `mlp.gate`, K=2048).

## Work queue, in order

### 1. Commit two pending fixes — DONE (one landed, one reverted)

- `97a06db3f` qt-49 untied lm_head — landed.
- `03f7a2612` Cholesky cache — landed, then **reverted in `dc18bdd39`**. It was a
  measured pessimization: `oq4.25++` is the AwqLdlq recipe, and the AWQ half
  rebases the Hessian PER TENSOR (`H' = diag(1/s) H diag(1/s)`, with `damp`
  derived from the rebased diagonal), so experts sharing a pooled donor do NOT
  share a factorization. The cache could never hit; all it added was its key, a
  serial FNV chain over K²=4.2M f32 per call. The 35B build reached layer 3 in
  the time the baseline reached ~layer 6.

  Consequence for §3 and for planning: **the per-expert factorization is
  irreducible** for calibrated recipes. 10240 × (K² rotation + K³/3 Cholesky at
  K=2048) is legitimately most of that 62-minute quantize. The algebraic
  shortcut does not rescue it — L' = D L holds for a plain Cholesky, but the
  damping is applied after the rebase and (D H D + λI)⁻¹ ≠ D (H + λ'I)⁻¹ D.

Original note, for context:

- `crates/hipfire-arch-qwen35/src/qwen35/loading.rs` — qt-49 (`Bf16Lut3`) arm on
  the SEPARATE lm_head path. Our own `--format bf16` makes LUT3 the default head
  coding and `expand_bf16_index` leaves head tensors packed, so every UNTIED
  qwen35-family model we produced was unloadable (`unsupported quant_type 49`).
  The tied path already had this arm; the untied one did not.
- `crates/hipfire-quantize/src/ldlq.rs` — Cholesky/rotation cache. Memoizes
  `rotate_hessian` + `inv_cholesky_dispatch` across all 7 packer call sites,
  keyed on a fingerprint of `(H, k, signs, damp)`. With a pooled donor shared by
  256 experts this turns 10240 factorizations into ~40. 13 ldlq tests pass
  including `ldlq_factor_cache_is_transparent`.

### 2. Merge `origin/master` — DONE, on a branch, awaiting a clean tree

Merged, resolved, and verified on **`merge/origin-master-2026-08-09`**
(commit `4761b42b6`), built in a worktree at `$S/merge-wt` so it never touched
the dirty main tree. `./tests/no-gpu-ci.sh` exit 0; `cargo test -p
hipfire-quantize --lib` 123 passed.

All 6 conflicts resolved; the reasoning is in the merge commit message. The one
that mattered is `codecs.rs`, exactly as this doc warned: upstream re-derived
the OLD alternating coordinate descent while generalising to G=128 slices, so
taking theirs would have silently reverted the joint selector. Resolved as our
algorithm + their slice generality, with a new
`mixed_clipsearch_is_the_grid_argmin_at_g128` test covering the new size.
`repack.rs` turned out to be safe to take wholesale — our side was provably
`rustfmt(base)` and nothing more.

**To land it**, the working tree has to be clean. As of 2026-08-09 it holds 23
modified files, and every one of the 19 `.rs` files among them is
**formatting-only** (an uncommitted `cargo fmt`; verified file-by-file as
`rustfmt(HEAD) == worktree`). Upstream reformats those same files anyway. The
only uncommitted work with real content is non-Rust and untouched by the merge:

| file | change |
|---|---|
| `docs/kernel_work/two-stage-lmhead.md` | +225 |
| `docs/npu/wire-in-r6-prefill-offload.md` | +218 |
| `scripts/adhoc/opus-outlier-budget-sweep.sh` | +5 −1 |
| `benchmarks/npu_gemm_tuning/r6/r6_cache.sh` | +5 |
| `CLAUDE.md` | −10 |

So: commit (or keep) those five, discard the formatting-only `.rs` churn, then
`git merge merge/origin-master-2026-08-09` — it fast-forwards. Left undone
deliberately, because discarding even formatting-only changes is the tree
owner's call.

Original note:
88 behind / 50 ahead. **rustfmt does NOT help** — measured, 6 conflicts before
and after (1.96.0 ships the same rustfmt we run). The 6 are all pre-session
branch work:

| file | note |
|---|---|
| `coexistence/repack.rs` | upstream rewrote ~1000 lines — the big one |
| `quantize/codecs.rs` | **two-sided.** Ours = joint scale+mask selector; theirs = G=128 slice generalization. Keep OURS and layer theirs on top. Taking upstream wholesale silently reverts the selector. |
| `coexistence/main.rs`, `hub/run.rs` | moderate |
| `coexistence/lib.rs` | trivial |
| `runtime/env_docs.rs` | generated — regenerate, don't merge |

Upstream also pinned toolchain 1.96.0 and reformatted; expect to `cargo fmt`
after. No duplicated work: upstream's quant work is `OqPlusCompact` G=128
(qt 52), a different axis.

### 3. Re-quantize the 35B with the Cholesky cache — MOOT (cache reverted)
The 3731 s baseline stands as the real cost. The shipped
`Qwen3.5-35B-A3B--oq4.25++.hfq` (17.9 GB, `mean_kld 0.038913`, 95% CI
[0.028996, 0.048830], n=16) is the benchmarked artifact; nothing re-wrote it.

Original note:
Same command as §Recipe step 4. Two outputs wanted: the wall-time delta against
the **3731 s** baseline, and an **md5 match** against the existing
`Qwen3.5-35B-A3B--oq4.25++.hfq`. A byte match on a real 10240-expert workload is
the strongest available proof the cache is transparent.

### 4. Batched MoE prefill — RE-OPENED 2026-08-09. Not policy-blocked after all.

The "policy conflict" below is **wrong**, and the experiment that disproves it
took an hour. Measured, not reasoned:

Widening the `pbs_eligible` state-quant arm to admit `DeltaNetMoe`/`FullAttnMoe`
alongside the two dense kinds lets a MoE model into batched prefill **under
FP32 state, with no Q8 anywhere** — so `state.rs`'s "never Q8" policy is never
engaged. The gate passed on Qwen3.5-35B-A3B on the first try.

Why that was always going to work, and why the old reading was too pessimistic:

- An FP32 batched DeltaNet path already exists —
  `gated_delta_net_f32_batch_seq` (`dispatch/gated.rs:146`), and for MoE
  specifically `gated_delta_net_f32_routed_batch_seq` (:197). FP32 is
  unsupported ONLY for tree replay (`prefill_chunk.rs:2962` returns an explicit
  error), which is spec-decode/MTP, not scoring. There is **no FP16 path** —
  FP32 and Q8 are the only two implemented, so `state.rs`'s FP16 alternative is
  aspirational.
- The per-layer admissibility closure ALREADY validates `DeltaNetMoe` and
  `FullAttnMoe` properly. Only the state-quant arm omitted them.
- The tree-verify forward keeps its own separate eligibility check, so widening
  this gate cannot route FP32 models into the unsupported tree path.

**What actually blocks it** is one missing kernel arm, not a policy:

    kernel not registered: FusedGateUpOq8G256

`oq4.25++` promotes the shared expert to OQ8, and the batched **shared-expert
fused gate_up** has no OQ8 implementation. Everything around it does:
shared-expert `down` (`gemv_oq8g256_residual_sigmoid_scaled_gpu_batched`),
routed `gate_up` and routed `down` are all batched-OQ8 already. The codebase
even documents the hole at `prefill_chunk.rs:2677` ("No fused oq8 PREFILL arm
exists"). Note `moe_ffn_batched_admissible` ADMITS the layer the registry
cannot serve — the predicate and the kernel table disagree, which is why this
surfaces as a runtime error rather than a clean decline.

Two ways to close it:

1. Wire the existing `fused_gate_up_oq8_wmma` (`dispatch/fused.rs:2670`,
   kernel source `kernels/src/fused_gate_up_oq8_wmma.hip`). It wants
   PRE-QUANTIZED activations (`xq`, `xs`), unlike the OQ4 arm which takes f32
   and quantizes internally, so it needs int8 activation scratch in
   `PrefillBatchScratch` plus a quantize step. Then register
   `(FusedGateUpOq8G256, ArchPredicate::HasWmma)` in `fused_qkv_table.rs` and
   add the arm in `families/fused_qkv.rs`.
2. Add a plain batched OQ8 projection and call it twice. Today the only batched
   OQ8 entry points are the three specialised ones above — there is no generic
   one to reuse.

Either way, tighten `moe_ffn_batched_admissible` in the SAME change so an
unservable dtype declines to per-token instead of erroring.

The prize is worth it: this is also what gates the 397B (§8), and it is the
difference between per-token and batched prefill on every MoE model.

The experimental gate change was reverted — on its own it converts
slow-but-working into a hard error for every `oq4.25++` model.

Superseded original note:

The premise below was wrong, and the trace says so. The predicted
`MoE grouped GEMM declined` line **never fired**; the grouped MoE path was never
the thing declining. The line that did fire is a different gate,
`pbs_eligible` (`crates/hipfire-arch-qwen35/src/qwen35/mod.rs:2015`):

```
force_fallback=false n=2048 dn_quant=FP32 all_layers_dense_la=false
moe_topk_ok=true (K=8, E=256) router_logits=true arch=gfx1151
```

Everything passes except the DeltaNet state precision. The conjunction reads:

```rust
&& matches!(dn_state.quant, StateQuant::Q8 | StateQuant::FP32)   // FP32 passes
&& (dn_state.quant == StateQuant::Q8
    || weights.layers.iter().all(|lw| matches!(lw, DeltaNet(_) | FullAttn(_))))
```

and the code's own comment states the structural consequence: *"a model with
DeltaNetMoe/FullAttnMoe layers fails the `all(DeltaNet|FullAttn)` arm by
construction, so for any MoE model batched prefill REQUIRES
`dn_state.quant == Q8`."*

So the only one-line fix is to make DeltaNet state Q8 — which is **explicitly
forbidden**. `qwen35/state.rs:34`: *"POLICY (2026-07-19): DeltaNet state must
NEVER be Q8… The only sanctioned future alternatives are FP16 or a purpose-built
DeltaNet-state codec."* Q8 state has already produced long-decode attractors, a
stochastic-rounding seed leaking execution history into target numerics (#17),
and an `s_ef_residual` rollback hazard that breaks spec-decode losslessness (#22).

This is a real conflict between two deliberate decisions, not a stray guard.
Unblocking it means the sanctioned work — FP16 state, or the DeltaNet-state
codec — plus relaxing the `all(DeltaNet|FullAttn)` arm to admit `*Moe` layer
kinds. That is a design task, so it is skipped per the goal's "if 4 is not
simple, skip it".

Original note (premise superseded):
`$S/ktrace.log` will contain
`[kernel-trace] MoE grouped GEMM declined: expert_gate_up=… arch=… use_path2=…
mixed=… experts_empty=… buckets=… n=… k_top=… n_exp=…` (source:
`crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:811`).

Context: the batched MoE prefill path EXISTS and covers our arch —
`hipfire_dispatch::pipeline::run_grouped_moe_gemm` has a dedicated
`DType::BF16 if gpu.arch == "gfx1151"` arm (`gemm_bf16_moe_grouped_wmma_gfx1151`,
raw BF16, not LUT3) plus a portable fallback. The per-token path in
`moe_decode.rs` is a labelled "Phase 1 dense-compute decode reference". So this
is a routing/gate bug, same class as the BUG-001 batchability guard removed
earlier this session (58.5 → 1646 tok/s on the dense model). Reference build
currently runs ~22.8 tok/s on a 3B-active model — decode speed.

Fix the named gate, re-measure the same 1-chunk build.

### 5. Queue three models — DONE. Results below.

All four `oq4.25++` builds, scored at n_ctx=2048, max_chunks=16, against each
model's OWN bf16 reference on the frozen eval slice. **KLD is per-model
self-relative, so these rank degradation, not absolute quality.**

| model | mean KLD | 95% CI | min / max chunk | artifact | LDLQ |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | **0.038913** | [0.028996, 0.048830] | 0.025 / 0.110 | 17.9 GB | 10550/20791, pooled 10240 |
| Qwen3.6-35B-A3B | **0.051324** | [0.042859, 0.059789] | 0.026 / 0.088 | 18 GB | 10550/20791, pooled 10240 |
| Qwen3.5-27B (dense) | **0.110658** | [0.062459, 0.158856] | 0.034 / **0.411** | 16 GB | 496/497 |
| Qwen3.6-27B (dense) | **0.143372** | [0.080524, 0.206220] | 0.024 / **0.398** | 16 GB | 496/497 |

Stage wall-times: MoE calib 3484s / quant 3518s / ref 3148s / score 1432s.
Dense streamed calib 1432s / quant **6997s** / ref 371s / score 1913s.

**The dense pair is worth a look before anyone trusts it.** It is NOT a
coverage defect — the dense models got *near-total* LDLQ (496/497, only
`lm_head` missing) while the MoE models necessarily skip 10240 expert
`down_proj`, and the dense pair still scored 2-3x worse. Two things to check:

1. Both CIs are wide and both are driven by a single ~0.4 outlier chunk against
   a ~0.03 minimum — heavy-tailed, so the mean is not a good summary. n=16
   cannot resolve this; re-score with more chunks before drawing conclusions.
   **Still open.**
2. ~~Both 27Bs are **VL** checkpoints...~~ — **RULED OUT 2026-09-04.** Checked
   against the artifacts rather than the config habit:
   - It is not a dense-vs-MoE difference. **All four** models are VL —
     identical `text_config`, `vision_config`, `image_token_id: 248056`,
     `Qwen3_5(Moe)ForConditionalGeneration`. The MoE pair carries the same
     nesting and scored 2-3x BETTER.
   - There are **zero vision tensors** in any of the four artifacts
     (`visual|vision|image|patch_embed`: 0 hits).
   - By design: `hipfire-quantize --include-vision` is documented as "include
     vision-tower tensors (default: skipped)". The tower was never a candidate,
     so the proposed mechanism has nothing to act on. The VL-ness is
     config-only.

**The coverage framing in the paragraph above is also backwards, and that is the
live lead.** "Near-total LDLQ (496/497)" counts TENSORS, and LDLQ "success" was
never evidence that LDLQ helped: `inv_cholesky_lower_rotated_fast` retries at
lambda x{1,10,100,1000,10000} and, until 2026-09-04, `continue`d on failure with
nothing recorded. A tensor that only factorized at the top of that ladder has
lambda swamping its Hessian — H+lambda I is effectively lambda I, so the OBS
solution collapses toward RTN — and it incremented the same counter as a clean
success.

With the ladder instrumented (`feat/ldlq-damping-telemetry`), a real
`oq4.25++ --ldlq` build of the Qwen3.5-9B DFlash drafter reports:

    LDLQ tensors:     success=36 attempts=36 missing=0 k_mismatch=0 pack_failed=0
    LDLQ damping:     1x=2 10x=29 100x=4 1000x=1 (escalated=34 exhausted=0)

**34 of 36 tensors needed escalated damping.** The old line — `success=36
attempts=36`, everything else zero — reads as a flawless build. Only two tensors
(`layers.{2,3}.self_attn.o_proj`) factorized at the damping actually asked for.

The rung tracks K, which is what makes it a conditioning problem and not a
coverage one:

| K | rung |
|---|---|
| 4096 | 10x (27 tensors), 100x (1) |
| 12288 (`down_proj`) | 10x (2), 100x (3) |
| 20480 (`fc`) | **1000x** (1) |

That is the asymmetry, and it points the opposite way from the paragraph above.
The dense 27B ran LDLQ on 64 x `mlp.down_proj` at **K=17408** — the largest and
worst-conditioned Hessians in either model, ~72% of its 31.5 GB calib. The MoE
pair SKIPPED its 10240 expert `down_proj` (K=512) entirely. So the MoE may have
been spared the tensors LDLQ was hurting, not handicapped by missing them.

Mechanism is measured and independent: `88a5b192c` ("Hessian PSD: bf16 STORAGE
is the dominant error, and f16 would cut damping 8x", on
`origin/feat/npu-oq4-decode`, NOT on master) shows bf16 on-disk storage forces
damping to ~13% of the mean diagonal where f16 would need ~1.6% — at the same
2 bytes/element — and that rank deficiency sets the floor. Our calibs store
`HessianBf16TrilDiagF32`.

Not proven for the 27B specifically: nobody has re-run it with the telemetry.
That run is now cheap and needs no re-calibration — every input is on halo.

Calibration route was forced by shape, not preference: a dense 27B has
`intermediate_size=17408`, so ONE `down_proj` Hessian is 17408²·4 = 1.13 GiB and
64 layers is 72 GB resident — `collect_artifacts` OOMed in 48s. The streamed
engine holds one layer (766 MB). MoE models are unaffected (expert
`down_proj` is K=512) and stay on the resident path, which is also the one
wired for the pooled donor.

Original note:
Same recipe, unchanged: **Qwen3.5-27B**, **Qwen3.6-27B**, **Qwen3.6-35B-A3B**.
Sources are `.hfa` under `/srv/hipfire/models/`. All fit resident (54/54/70 GB
bf16 against 124 GB RAM).

### 6. kldref legacy→HFKREF converter
Unlocks bf16-referenced benchmarking for models too big to hold resident.
`mode=score` consumes **HFKREF** (`RefMeta` header + `compat()` gating,
`crates/hipfire-kld/src/archive.rs`); the streamed calibrate embeds a **legacy**
kldref (`lm_head.kldref_{idx,logit,logz}`, written by `legacy_kldref_tensors`).
No inverse exists. Conversion:

| HFKREF section | from legacy |
|---|---|
| `top_indices` | `indices` — direct |
| `top_log_probs` | `logits − log_z` |
| `residual_mass` | `1 − Σ exp(log_probs)` |
| `tokens` | **not in the payload** — take from the calib job's samples (artifact records `corpus` + `n_calib_tokens`) |

Plus a `RefMeta` header (calibs now carry `source_fingerprint`).

### 7. Qwen3.5-122B-A10B
244 GB bf16 — too big resident, so use the **layer-streamed** path
(`hipfire-coexistence calibrate`, which does per-layer/per-expert capture and
can emit kldref). Quantized ≈65 GB, so scoring fits once §6 lands.

### 8. Qwen3.5-397B-A17B — REASSESSED 2026-08-09. Not a capacity problem.

Both reasons given below are wrong.

**The deadlock is fixed.** Root cause was `mmap.advise(Sequential)` triggering
291 GiB of readahead whose kworker held an inode lock against the slab loader's
O_DIRECT fd on the same inode. Fixed in `hfq.rs` — 64 GiB threshold (line 1109)
and `FADV_DONTNEED` (line 1342), both verified present after the origin/master
merge. Post-fix it loads and generates. It is not a crash risk.

**"~380 GB quantized" was wrong** — at the 35B's measured 3.84x it is ~190 GB.

**Capacity is not the blocker, because scoring does not need it resident.**
Real config from the archive index: 60 layers, hidden 4096, 512 experts, top-10,
moe_intermediate 1024 → 12.08 GB per layer bf16, 724.7 GB total. Streamed and
double-buffered that is 5.6 GB quantized resident, leaving ~118 GB. The entire
16x2048 = 32,768-token scoring job is ~12 GB of activations, so it fits in ONE
batch: a single streaming pass, 1.4-2.8 min of quantized weight read at
1-2 GB/s plus ~0.4 min of compute.

**The actual blocker is §4.** The 397B is `qwen3_5_moe` with hybrid
`linear_attention`/`full_attention` layers, so it hits the same `pbs_eligible`
gate. Without batched prefill, scoring runs per-token — and per-token against
streamed weights means re-reading ~190 GB PER TOKEN, which is the real reason
anything about this model ever looked hopeless. Land the §4 kernel arm and the
397B benchmark becomes ordinary work.

Still true: do not attempt a RESIDENT load, and the DFlash sidecar at
`/srv/hipfire/models/Qwen3.5-397B-A17B-DFlash--bf16.hfq` is the drafter (6
layers, 2.58 GB), not the model. The model exists only as the 550 GB `.hfa`.

Superseded original note:
794 GB bf16, ~380 GB quantized against 124 GB RAM. Calibration is possible
streamed, but scoring is not, so there is no benchmark at the end. Also has a
documented history of deadlocking amdgpu on this box (hard reboot).

## Recipe (the settled one — do not deviate)

```sh
M=Qwen3.5-27B                       # example
BF16=~/.hipfire/models/$M--bf16.hfq
CAL=~/.hipfire/calib/$M.calib.hfq
OQ=~/.hipfire/models/$M--oq4.25++.hfq
REF=~/.hipfire/kldrefs/$M--bf16.kldref.hfq

# 1 restore
hipfire-coexistence repack --input /srv/hipfire/models/models--Qwen--$M.hfa \
  --output ~/.hipfire/restore/$M
# 2 bf16
hipfire lock run bf -- hipfire-quantize --input ~/.hipfire/restore/$M \
  --output $BF16 --format bf16
# 3 calibrate  (DISJOINT corpus + sequence split)
HIPFIRE_CALIB_SEQ_LEN=2048 hipfire lock run cal -- \
  ./target/release/examples/collect_artifacts --model $BF16 \
  --corpus benchmarks/calib/calib-multi-8m.txt --max-tokens 32768 --output $CAL
# 4 quantize  (POOLED is mandatory for MoE)
HIPFIRE_POOLED_EXPERT_HESSIAN=1 hipfire lock run q -- hipfire-quantize \
  --input $BF16 --output $OQ --format oq4.25++ --hessian $CAL --ldlq
# 5 reference from the EVAL slice, 6 score  -> see $S/q35.sh
```

Token budget rationale: rows/expert ≈ N·k_top/n_exp. For 256 experts top-8,
32768 tokens ⇒ ~1024 rows/expert ⇒ n/K ≈ 2 at expert `down_proj` K=512, which is
where LDLQ saturates (`docs/quant-formats/moe-expert-hessians.md` Q2).

## Traps (each of these cost real time)

- **NEVER calibrate on `benchmarks/quality-baselines/`.** That is the frozen
  EVAL slice. Doing so trains on the test set and produced a fake −13.6% plus an
  inverted-U budget curve. `reject_eval_corpus` now hard-errors; override only
  to measure the effect deliberately. Calibrate from `benchmarks/calib/`.
- **`HIPFIRE_POOLED_EXPERT_HESSIAN=1` or every routed expert is silently RTN.**
  `--ldlq` logs `ldlq: skip` and continues reporting success.
- **Do not commit while a GPU job holds the lock** — the pre-commit tiny-quant
  gate blocks for hours. Use `--no-verify` only for throwaway probe commits.
- **`pkill -f <pattern>` kills your own shell** (bare `exit 144`). Use `pkill -x`.
- **`git stash` is unusable here** (`.agents/` symlink farm) — it fails, and a
  later `pop` restores an unrelated old stash over your work.
- **Rebuild EVERY affected binary.** The reference build failed once because
  `collect_artifacts` was rebuilt with the qt-49 fix and `hipfire-daemon` was not.
- **Do not wrap `hipfire-eval` in `hipfire lock`** — it deadlocks against itself.
- **The tiny-quant gate has 8 pre-existing failures** (qwen2/hfq4, gemma3
  q8f16+hfq4, minimax/mq4, qwen3_5/q8f16, qwen3_5_moe q8f16+mq6+mq4). They are
  NOT regressions; they were identical before every commit this session.
- **Statistics:** compare arms built from the SAME calib and scored against the
  SAME reference. `mode=score` draws tokens from the reference, so arms are
  paired by construction. n=16 chunks cannot resolve effects below ~3%.

## Retracted — do not resurrect

"Calibration token budget is the biggest lever, −12.9%/−13.6%." That was
train-on-test (calib and reference were the same file from offset 0). Budget is
**unmeasured**; re-run the 8k/32k/128k sweep from a disjoint corpus region before
believing any budget claim. Details in `moe-expert-hessians.md` Q7.

Still valid: pooled `gate_up` −3.09% (CI excludes 0), per-expert `down_proj`
worthless-and-unstable, and the 10.1× sequence-split speedup.

## Open design thread (not blocking)

Daemon-owned scheduler with per-pass jobs (forward/backward of a layer or an
expert) plus collection modifiers, chained for throughput but preemptable
mid-chain. Expert granularity is the right unit — it matches VRAM residency and
eviction, enables opportunistic cross-request micro-batching, and bounds WCET
for realtime (TTS/STT) preemption. Two hazards to design around:

1. Capture is keyed by weight-buffer POINTER (`gpu.capture_names:
   HashMap<usize, String>` from `buf.as_ptr()`). Expert paging changes addresses
   and calibration silently captures nothing.
2. Cross-request batching needs per-request output routing on top of
   `inverse_perm`, which only restores order within one pass.

Also proposed: count every batch=1 dispatch taken while `rows > 1`, keyed by
(kernel, dtype, arch), report unconditionally like the LDLQ counters, and fail
the gates on nonzero. Do not ban batch=1 kernels — they are genuinely faster at
N=1 (no WMMA padding waste, no activation staging, M-only grid, fused
residual epilogues). The defect is only ever taking one when `rows > 1`.

## DeltaNet Q8 removal — the last mile (2026-08-09)

Q8/Q4 are gone from `StateQuant`, but the Q8 **dispatch functions** in
`hipfire-rdna/src/dispatch/gated.rs` still exist and are still CALLED. Renaming
`gated_delta_net_q8.hip` to `-disabled.hip` did not make them unreachable; it
just made them look retired. Under FP32 the surviving callers are harmless, so
every check passed. Under FP16 the state is half-size with a vestigial
`s_scales`, and a teacher-forced score of a dense Qwen3.5-0.8B died with:

    Memory Fault Error ... kernel: gated_delta_net_q8

That is why FP16-by-default was reverted (c2412857a). FP16 stays available via
`HIPFIRE_DN_STATE_FP16`, and enabling it is UNSAFE until this is finished.

Deleting the dispatch fns + their kernel constants makes the compiler produce
the exact list, which no amount of grepping did — the call sites do not mention
`StateQuant` at all, so they are invisible to a variant search:

| site | arm |
|---|---|
| `prefill_chunk.rs:2972` / `:6479` | `gated_delta_net_q8_tree_batch_seq` |
| `prefill_chunk.rs:3013` / `:6507` | `gated_delta_net_q8` (per-token) |
| `prefill_chunk.rs:3030` / `:6524` | `gated_delta_net_q8_batch_seq` |
| `prefill_batch.rs:664` | `gated_delta_net_q8_routed_batch_seq` |
| `speculative.rs:3374` | (surfaced by the compiler; unexamined) |

Replacements all exist and are hardware-validated: `gated_delta_net_f32/f16`
`_batch_seq`, `_routed_batch_seq`, `_tree_batch_seq`.

The one piece of real work, and why this was not finished in-session: the TREE
arms need a tape buffer sized for the new precision. `pbs.dn_s_tape_q8` +
`dn_s_tape_scales` are Q8-shaped; an f32 tape is 4x the bytes and an f16 tape 2x,
with no scales at all. So `PrefillBatchScratch` has to grow the right buffer
before those two arms can be rewritten. The other six sites are mechanical.

Order that works: delete the dispatch fns first, let the compiler enumerate, fix
scratch, rewrite arms, then re-run the teacher-forced comparison (build a KLD ref
with `HIPFIRE_DN_STATE_FP32=1`, score the SAME weights with FP16 — only the state
precision differs, so any KLD is precision alone). That test found this bug in
5 seconds after four rounds of greps missed it.
