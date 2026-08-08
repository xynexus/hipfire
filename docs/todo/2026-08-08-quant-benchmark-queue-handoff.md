# GOAL: finish the quant benchmark queue (unattended handoff)

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

### 2. Merge `origin/master`
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

### 4. Fix the MoE grouped-GEMM decline — SKIPPED (policy-blocked, not a gate bug)

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

### 5. Queue three models that run end-to-end today
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

### 8. Qwen3.5-397B-A17B — BLOCKED, do not start
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
