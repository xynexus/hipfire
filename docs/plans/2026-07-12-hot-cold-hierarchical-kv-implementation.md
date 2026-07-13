# Hot/cold deferred-hierarchical KV — firm implementation plan

Status: **active**. Branch: `chaingun`. Date: 2026-07-12.

Consolidates and supersedes the *task lists* of
`docs/plans/2026-06-22-hierarchical-kv-followups.md` (rough follow-up order) and
`docs/plans/2026-06-23-kv-merge-design-levers.md` (encode-time levers) into one
sequenced plan with firm acceptance gates. Those two docs remain the design
rationale; this doc is the execution contract. The `2026-07-1{0,1}` *latent-KV*
plans are a **separate, rejected** thread (shared static rank-32 basis, dead
across 0.8B/4B/9B) and are out of scope here.

## The one governing fact

Established on a leak-free harness and not up for re-litigation:

- **The merge is the only quality cost.** `fold_m=1` (no merge, pure cold quant)
  scores PPL 26.13, *beating* the all-cold baseline 30.81. Cold quant is ~free
  even at 2-bit (34.56 vs 34.00). The loss is entirely the `m:1` importance merge.
- **The merge loss is RoPE-phase blur** — averaging K at different positions
  cancels their rotary phases (~+7 PPL per fold doubling).
- **Fine-tuning cannot recover it** (~27% held-out, overfits; proven, see
  `[[project_qat_recovery_probes]]`). Therefore the floor must be lowered at
  **encode/design time**, not trained away.

Every phase below is ordered by that fact: cheap validation first, then the
merge-floor attack (the only thing that moves quality), then storage/read-cost
housekeeping.

## Current shipped state (ground truth anchors)

Flag-gated by `HIPFIRE_KV_HIERARCHICAL=1` (default off → byte-identical
baseline; requires `head_dim==256`). Merged to `chaingun` 2026-06-22.

- **State machine:** `crates/hipfire-runtime/src/kv_hier.rs` — `HierKvState`
  (L104), `ColdSegmentGpu` (L80). Hot tier = per-layer slot-major f32 ring; cold
  = `Vec<Vec<ColdSegmentGpu>>`, **~1 segment accumulated per turn** (no defrag).
  `append_token` L267, `migrate_n` L295, `idle_compact` L457, `two_tier_read`
  L498, `reset` L246.
- **Codec:** `crates/hipfire-kvquant/src/kvarn.rs` — `quantize_tile_qmax` L187
  (supports bits 1/2/4 via `qmax = 2^bits-1`), record layout + per-channel scale
  overhead `kvarn_record_bytes_bits` L255. `compact_cold_kv` in
  `crates/hipfire-kvquant/src/kv_compact.rs` L56.
- **Kernels (all zero-LDS, register + `__shfl_xor`):** `attention_cold_slots`
  (dispatch/attention.rs L5004), `flash_tier_merge` (L5083), `flash_partials_ml`
  (L5142), `kvarn_dequant_tile` (dispatch/kv.rs L1338).
- **Serve hooks:** `crates/hipfire-serving-core/src/qwen35_prefill.rs` — idle
  drain L135, per-token forward guard L147. `generate.rs` — `kvarn_active` guard
  L2256 (skips DFlash/MTP spec paths), idle_compact L2858.
- **Env knobs** (`env_docs.rs`): `HIPFIRE_KV_HIERARCHICAL`, `_HOT_BUDGET` (256),
  `_MIGRATE_BATCH` (128), `_FOLD_M` (4), `_CORE_FRAC` (0.125), `_IMPORTANCE`
  (vnorm), `_POS_LOCAL` (1), `_COLD_BITS` (4), `_COLD_V_BITS`, `_COLD_V_PERSLOT`,
  `_IDLE_KEEP` (0).
- **Parity oracles:** `parity_{attention_cold_slots,flash_tier_merge,
  flash_partials_ml,cold_4bit_read,two_tier_e2e}` (hipfire-rdna/examples) +
  `parity_kv_hier` (hipfire-runtime/examples). Keep all green through every phase.

Best point today: **hot=64 / fold=4 / vnorm + pos-local → PPL 34.0 @ ~18× KV
compression**. Residual +3 PPL over the fold=1 machinery floor (26.13) = the
merge floor this plan attacks.

## Frozen eval methodology (applies to every phase)

Do not vary these between phases; a phase "passes" only against this rig.

- **Harness:** `crates/hipfire-runtime/examples/perplexity.rs` — the qwen3.5
  KV-cache-quant harness (`cargo build --release --example perplexity`). `--kv-mode`
  selects the resident KV representation; `kvarn` is a real mode (L326
  `KvCache::new_gpu_kvarn`, which reads the `HIPFIRE_KV_*` hier knobs via
  `HierKvState::from_env`). The removed `eval_hipfire` binary is **not** this tool.
- **KLD is a two-pass** (combined PPL+KLD): first `--dump-ref <ref.pkld>` on the
  **bf16 reference model** writes per-position top-K logprobs (`PKLD` magic); then
  `--kld-ref <ref.pkld>` on the candidate reports PPL and KLD vs that dump. Same
  `--ctx/--warmup/--offset` on both passes.
- **One window per run.** `perplexity` scores a single `[offset, offset+ctx)`
  window at positions `[warmup, ctx-1)` — a `--ctx 2048` run averages KLD/PPL over
  ~2040 positions, which supersedes the old "2-chunk noisy" concern (that was
  `eval_hipfire` terminology). For coverage, run **≥3 disjoint `--offset`s** and
  average; trust NLL/PPL over top-K KLD.
- **Corpus:** `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`.
- **Artifacts.** `perplexity` dumps from a **loadable bf16 model**, not from a
  `.kldref.hfq`. The `/srv/hipfire/kldrefs/*.kldref.hfq` files are
  `artifact_kind: hipfire.kldref` precomputed references for the *separate* unified
  hipfire-kld path (`config_from_hfq` panics on them — confirmed 2026-07-12).
  - 0.8B: candidate `/srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq`; bf16 ref model
    `~/.hipfire/w8probe/qwen3.5-0.8b-bf16.hfq` (**staged, validated** — bf16 PPL
    24.05 @ offset 0).
  - 9B: candidate `~/.hipfire/models/qwen3.5-9b-mq4.hfq`; bf16 ref model
    `~/.hipfire/models/qwen3.5-9b-bf16.hfq` is **NOT staged** — pack a bf16 9B hfq
    from `/srv/huggingface/models--Qwen--Qwen3.5-9B` first (offline `hipfire-quantize`),
    or run the 9B arm via the unified hipfire-kld path against the existing
    `qwen3.5-9b-bf16.kldref.hfq` (same slice, pinned md5). Locating/packing the 9B
    bf16 is a Phase-0 sub-task; it does not block the 0.8B arm.
- **Canonical invocation** (0.8B, fold=4 default; substitute env knobs per config):
  ```
  CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
  PX=./target/release/examples/perplexity
  $PX ~/.hipfire/w8probe/qwen3.5-0.8b-bf16.hfq $CORPUS --ctx 2048 --offset 0 --dump-ref /tmp/ref08_o0.pkld
  HIPFIRE_KV_HIERARCHICAL=1 HIPFIRE_KV_HOT_BUDGET=64 HIPFIRE_KV_FOLD_M=4 HIPFIRE_KV_COLD_BITS=4 \
    $PX /srv/hipfire/models/qwen3.5-0.8b-mq4+.hfq $CORPUS --ctx 2048 --offset 0 --kv-mode kvarn --kld-ref /tmp/ref08_o0.pkld
  ```
  Runs at ~13 tok/s on gfx1103 (~155s per 2048-ctx window). Coordinate under
  `hipfire lock acquire <label>` (positional label required; no `--wait` flag).
- **Coherence:** `./tests/coherence-gate-dflash.sh` after any change touching the
  merge, codec, or read path (KV changes touch the attractor surface). Watch for
  the known mangled-multibyte-emoji failure mode on the hier path.
- **Perf A/B:** `scripts/probe_commits.sh` with a byte-identical prompt; two-tier
  read cost vs single-tier KVarN, and confirm `idle_compact` removes
  mid-generation migration spikes.
- **Hardware gate:** nix2 (gfx1103) with CWSR workaround live (`cwsr_enable=0`,
  verify `hipfire doctor`). **Every new cold-path kernel stays zero-LDS** or it
  wedges the box. Coordinate GPU runs with `hipfire lock {acquire,release}`.

## Phases

### Phase 0 — Re-baseline on the frozen rig — **DONE for 0.8B/offset-0; 9B + multi-offset pending**

The shipped numbers are 2-chunk and predate the current tree. Build `perplexity`
and re-measure the headline configs over ≥3 offsets (2048-ctx windows) on 0.8B
**and** 9B, so every later phase gates against a trustworthy baseline.

- Configs: `fold=1` (machinery floor), `fold=4 vnorm+poslocal` (shipped default),
  `fold=4 2-bit` (`HIPFIRE_KV_COLD_BITS=2`).
- Note the shipped table used `hot=64`; the current `_HOT_BUDGET` default is 256.
  Pin `HIPFIRE_KV_HOT_BUDGET=64` to reproduce the historical PPL 34 point, and
  also record the hot=256 default as the new operating baseline.
- **Gate:** parity oracles green; the *relative* structure reproduces (merge is the
  only real cost); numbers recorded in `benchmarks/results/hier-kv-rebaseline-20260712/`.

**0.8B results (2026-07-12, mq4+ candidate, bf16 ref PPL 24.05, offset 0, ctx 2048):**

| config | PPL | KLD/tok |
| --- | --- | --- |
| plain-kvarn (hier off) | 26.38 | 0.083 |
| hier fold=1 hot=64 | 26.52 | 0.083 |
| hier fold=4 hot=64 (default) | 30.75 | 0.320 |
| hier fold=4 hot=64 2-bit | 31.05 | 0.331 |
| hier fold=4 hot=256 (new default) | 28.68 | 0.209 |

Confirms all three thesis points on the current tree: (1) machinery + cold read is
lossless (fold=1 ≈ plain-kvarn); (2) the merge is the only real cost (+4.2 PPL,
KLD 0.083→0.320 at fold=4 hot=64); (3) cold quant is ~free even at 2-bit (+0.3 PPL).
Also: **hot budget is the strong dial** (hot=256 fold=4 = 28.68 vs hot=64 = 30.75) —
design-lever #3. Absolute PPLs sit below the historical hot=64/fold=4 PPL 34.0
because the candidate is mq4+ (clip-search weights), not plain mq4; the relative
structure is what carries. `--offset {0}` only so far; ≥3 offsets + 9B still pending
(9B needs its bf16 ref packed — see methodology).

### Phase 1 — 1-bit cold probe — **DONE (rejected), 2026-07-12**

Codec already supports `bits=1`. Ran the env sweep on the Phase-0 rig (0.8B,
offset 0), `parity_kv_hier` oracle **PASS** first.

| config | PPL | KLD/tok |
| --- | --- | --- |
| fold=1 hot=64 4-bit (phase 0) | 26.52 | 0.083 |
| fold=1 hot=64 2-bit | 26.92 | 0.109 |
| fold=1 hot=64 **1-bit** | **191.94** | **1.960** |
| fold=4 hot=64 4-bit (phase 0) | 30.75 | 0.320 |
| fold=4 hot=64 2-bit | 31.05 | 0.331 |
| fold=4 hot=64 **1-bit** (K+V) | **42.98** | **0.566** |

**1-bit is rejected.** Isolating quant (fold=1), 1-bit blows PPL 26.5 → 192 — the
cold-quant floor is between 2-bit and 1-bit. **2-bit is the practical minimum**
(near-free: +0.4 PPL fold=1 / +0.3 PPL fold=4). Per the gate, recorded as the
cold-quant floor; **rotation-to-rescue-1-bit is not justified** — 2-bit already
delivers the storage at ~free quality, so the deferred ConQuR lever stays closed.
(`COLD_V_BITS` defaults to `COLD_BITS`, so the two 1-bit rows are the same K+V
config; irrelevant given the wholesale rejection.) 9B not run — 1-bit's 0.8B
failure is decisive and larger models carry *more* low-rank structure, not less, so
1-bit will not fare better; skipped to save GPU time.

### Phase 2 — RoPE-dephased merge — **CLOSED / NO-GO (2026-07-13, evidence below)**

**Verdict:** rejected by the ceiling analysis — the merge blur is content, not phase;
de-rotation recovers ~0% (see "Decisions taken 2026-07-13"). The design rationale
below is retained for the record; the kernel is not being built. Live quality levers
are now Decision 2 (bigger/f16 hot budget) and Phase 4 (low-rank content residual).

<details><summary>Original design rationale (not implemented)</summary>

This is the only phase that moves quality. Attacks the RoPE-phase-blur root cause
FT could not recover.

1. **Ceiling first (no runtime change).** On captured post-RoPE FA K
   (`HIPFIRE_DUMP_HIDDEN_ALL` dumps exist from the explore work), measure the
   best-case KLD of a de-rotate → average-in-dephased-frame → re-rotate merge vs
   the current flat vnorm merge at fold=4. If the ceiling does not clearly beat
   flat-merge, stop — the residual is value-difference, not phase, and this lever
   is closed. If it beats it, proceed.
2. **Implement behind a flag** in `compact_cold_kv` (kv_compact.rs): de-rotate
   each K in a cold group by its position phase to a common reference, average in
   the dephased frame, store `(mean_dephased, ref_position)`; fold the inverse
   reference-phase re-apply into `kvarn_dequant_tile`'s read (dispatch/kv.rs
   L1338) — **must stay zero-LDS**. Position-local merge (the current −2% proxy)
   becomes the coarse fallback; explicit dephasing is the principled version.
- **Gate:** fold=4 dephased **beats PPL 34** on 0.8B and holds/improves on 9B, at
  equal or fewer bytes; coherence gate green; parity_kv_hier + a new
  `parity_dephased_merge` oracle green. Default stays off until the gate passes,
  then flip the encode default.

</details>

#### Phase 2 kickoff notes (2026-07-12) — setup + a strategic question

Groundwork done: `parity_kv_hier` PASS on current build; the merge floor and its
fold/hot sensitivity are re-measured (Phase 0 table). Two things to resolve before
building the dephasing kernel:

1. **Capture tap — DONE (2026-07-12).** `HIPFIRE_KV_CAPTURE_K=<path>` in
   `migrate_n` (kv_hier.rs) appends the post-RoPE K about to be merged, token-major
   `[mb × kv_dim]`, with its absolute base position (record =
   `[u32 base_pos][u32 mb][u32 nkv][u32 HD][f32 ck…]`). Debug-gated, no behavior
   change when unset; validated via `parity_kv_hier` (well-formed records,
   non-intrusive). **Next:** run `perplexity` with the flag on a real model to get
   real post-RoPE captures, then the analysis below.
   - **Analysis is deliberately NOT auto-run.** qwen35 uses *partial* rotary
     (`n_rot = head_dim * partial_rotary_factor`, θ=1e7); the de-rotation must
     replicate that exact partial-rotary pairing or the ceiling verdict is
     confidently wrong — and it would gate a major cross-arch kernel. The analysis
     (per merge group: intra-group variance flat vs de-rotated-to-reference; ratio →
     phase-vs-content split) must include a RoPE round-trip self-check
     (de-rotate∘re-rotate = identity) before its verdict is trusted.

#### Decisions taken 2026-07-13 + execution status

User decisions: **1 = A** (long-context is the goal; short-ctx numbers are
misleading — move the whole evaluation to long ctx). **2 = raise hot budget AND
try an f16 hot ring.** **3 = defer group-scale.**

- **Model facts (qwen3.5-0.8b):** `max_position_embeddings = 262144` (long-ctx runs
  fit), `rope_theta = 1e7`, **`partial_rotary_factor = 0.25` → only n_rot=64 of 256
  dims are RoPE-rotated**, interleaved convention (pairs `(2i,2i+1)`,
  `kernels/src/rope_partial_interleaved_batched.hip`). **Strong prior against the
  dephasing kernel:** dephasing can touch at most the rotated-dim share of the merge
  blur; ≥75% of dims are pure content it cannot recover.
- **Ceiling analysis READY + verified:**
  `benchmarks/results/hier-kv-rebaseline-20260712/rope_dephase_ceiling.py` — de-rotates
  the interleaved n_rot=64 pairs, measures (rotated-dim variance share) × (phase
  fraction within rotated dims) = net dephasing headroom. Self-test PASSES (RoPE
  round-trip exact; pure-phase→1.000; pure-content→0.000). Needs a real capture:
  run `perplexity` with `HIPFIRE_KV_CAPTURE_K` on the real model, then feed the dump.
- **Long-ctx merge-penalty measurement DONE** (`longctx_08b_ctx16384.md`): ctx=16384,
  hot=512, 2-bit, bf16 ref PPL 16.10. fold=1 = 18.12 (KLD 0.128), fold=4 = 20.07
  (KLD 0.247), fold=8 = 20.64. **Merge penalty = +1.95 PPL / KLD ~doubles — moderate,
  does NOT balloon** vs short ctx. Cold 2-bit re-emerges as an equal cost (+2.0 vs
  bf16 in fold=1) now that ~97% of the cache is cold.
- **CEILING VERDICT — dephasing kernel is NO-GO (2026-07-13).** Ran
  `rope_dephase_ceiling.py` on 12,096 real merge groups (capture at ctx=4096,
  hot=64, fold=4): rotated-dim variance share 0.168, **phase fraction within rotated
  dims = -0.034 (≈0)**, net removable ≈ 0. **The merge blur is CONTENT, not phase**
  — adjacent tokens genuinely differ and position-local grouping already nulled the
  phase gap, so de-rotation recovers ~nothing. Confirms QAT-irrecoverable +
  position-local-only-2%. **Do not build the RoPE-dephased merge kernel.**
- **Redirect:** the loss being content (not phase) means the **low-rank cold residual
  (Phase 4)** — a per-group *content* correction — is the lever that could reduce the
  merge penalty, not dephasing. And **bigger hot budget (Decision 2)** shrinks the
  merged fraction directly. These are the live quality levers now; Phase 2 is closed.
- **Decision 2 (f16 hot ring + bigger budget) — DONE + validated (2026-07-13).**
  `hot_k`/`hot_v` now F16; write via `cast_f32_to_f16` into a reused `hot_cast` f16
  scratch; read via `attention_cold_slots` `layout=2/2`; migrate uses `download_raw`
  + `f16_to_f32` widen; ring shift/append offsets are 2 B/elem. Parity PASS (baseline
  4.0e-5, defrag, idle-drain). **f16 is near-lossless:** hot=512 fold=4 2-bit ctx=2048
  = PPL 27.5422 / KLD 0.1531 vs f32 hot 27.54 / 0.153 — identical to 4 s.f., for half
  the hot VRAM. **Default hot budget raised 256→512** (f16 makes it cost the old
  f32-256 VRAM).
- **PAYOFF (2026-07-13, ctx=16384, fold=4, 2-bit):** bigger f16 hot window is the
  long-context quality lever. hot=512 → PPL 20.07/KLD 0.247; **hot=1024 → 19.19/0.195;
  hot=2048 → 18.48/0.146.** hot=2048 cuts KLD 41% and lands only +0.36 PPL over the
  no-merge floor (18.12) — **a large exact window nearly eliminates the merge penalty
  at long ctx**, at ~112 MB f16 (half the f32 cost). This delivers exactly what the
  dead dephasing kernel targeted, via the cheap validated lever. Long-context users
  should set hot=1024–2048; a seq-len-adaptive hot budget is a natural follow-up.
2. **Strategic question raised by Phase 0 — resolve before the kernel.** Hot budget
   is a *very* strong dial: at fold=4, hot=64 → PPL 30.75 but **hot=256 → 28.68**
   (KLD 0.320 → 0.209), for the cost of keeping more recent tokens exact in the
   ring. The RoPE-dephased merge is a large, RoPE-geometry-coupled kernel change
   whose whole value is shaving the +4.2 PPL merge penalty. Before building it,
   confirm it beats the *cheap* alternative of simply operating at a larger hot
   budget (+ 2-bit cold for the storage). Concretely: sweep hot ∈ {64,128,256,512}
   × fold ∈ {4,8} at 2-bit on 0.8B + 9B and find the quality/byte knee first
   (design-lever #3). If a bigger hot ring at 2-bit already reaches the target
   Pareto point, the dephasing kernel may not be worth its complexity — decide on
   evidence, not on the plan's original ordering.

**Knee sweep result (2026-07-12, 0.8B, 2-bit cold, offset 0, ctx 2048):**

| hot | fold=4 PPL / KLD | fold=8 PPL / KLD |
| --- | --- | --- |
| 64 | 31.05 / 0.331 | 29.93 / 0.274 |
| 128 | 31.14 / 0.280 | 30.92 / 0.288 |
| 256 | 29.05 / 0.220 | 29.71 / 0.234 |
| 512 | **27.54 / 0.153** | 27.80 / 0.161 |

Hot budget is the dominant dial (64→512 fold=4: 31.05→27.54). The residual merge
penalty at hot=512 is only ~+1.1 PPL over the no-merge floor (~26.4), vs ~+4.7 at
hot=64 — a bigger hot ring shrinks the very penalty the dephasing kernel targets.

**But this sweep cannot settle the Phase-2 go/no-go, because ctx=2048 under-stresses
the cold tier.** At hot=512/ctx=2048, 25% of tokens stay exact — that is barely
compressing, the opposite of the regime hierarchical KV exists for. In real long
context (32k–128k) a fixed hot ring is a *small* fraction, the merged cold tail
dominates again, and the merge floor reasserts — which is exactly what the
RoPE-dephased merge (lever #1) attacks. So:

- **Cheap win, now:** at short/medium context, raising the default hot budget
  (256→512) at 2-bit cold is a real Pareto improvement with zero new code. Adopt it
  as the near-term operating point.
- **Open decision (needs long-ctx eval):** whether the dephasing kernel is worth
  building depends on the merge penalty at *long* context, which the staged
  `wikitext2-1024s-2048ctx` slice cannot measure (window caps at 2048). Deciding
  Phase 2 requires a longer-context corpus (≥16k–32k) where hot ≪ ctx, on 9B/27B
  where long-context compression matters. **Do not build the dephasing kernel until
  a long-ctx measurement shows a merge penalty large enough to justify it.**

### Phase 3 — Segment defrag **DONE (2026-07-12)** + group-scale overhead (#2, #3)

- **Defrag (#2) — implemented, wired, validated.** `HierKvState::defrag`
  (kv_hier.rs): when a layer holds > `HIPFIRE_KV_DEFRAG_SEGMENTS` cold segments,
  dequant them all → concat their real slots → re-`compact_cold_kv(core_frac=1,
  fold_m=1)` (pure repack, no re-merge) → one wider segment. Called from
  `idle_compact` after the drain (off the latency path). Pack path extracted into a
  shared `push_cold_segment` helper (reused by `migrate_n`). New knob
  `HIPFIRE_KV_DEFRAG_SEGMENTS` (default **0 = off**, byte-identical). Validated by
  `parity_kv_hier` (new `HIPFIRE_KV_HIER_TEST_DEFRAG` mode + env-path test):
  baseline PASS 4.9e-5, post-defrag oracle PASS 4.3e-5, env-path (idle→defrag) PASS
  1.2e-4.
- **Finding that reshapes group-scale (#3):** defrag is **not free**. Folding 6
  narrow tiles (per-channel scale over ~8 slots) into one wide tile (scale over 46
  slots) coarsens the 4-bit grid → measured ~1.6% output error per fold-6→1 (pure
  requant coarsening, not corruption — the oracle confirms valid data). Group-scale
  (#3) pushes the *same* direction (fewer scales) and would coarsen further. So the
  original "#2+#3 free win" framing is wrong: there is a real quality↔storage
  tension. **Group-scale must be gated on a measured SQNR/PPL budget, not assumed
  free**, and defrag itself carries a compounding ceiling (re-folding the wide
  archive each idle cycle re-quantizes the whole cold history; `ponytail:` note in
  `defrag` — upgrade path is generational/LSM compaction). Keep
  `HIPFIRE_KV_DEFRAG_SEGMENTS=0` (off) as default until either the compounding is
  bounded or a large session shows the read-cost win outweighs the coarsening.
- **Remaining for #3 (group-scale):** unstarted. It is a codec + on-GPU record
  format change touching the `kvarn_dequant_tile` HIP kernel — must work on
  RDNA2/3/4 and pass `coherence-gate-dflash.sh`. Higher blast radius than defrag;
  do it only after quantifying the coarsening budget above.
- **Gate (partial):** defrag read-cost-per-token flat across turn count ✅ (segment
  count bounded); `parity_kv_hier` green ✅. Group-scale storage/quality gate + full
  `coherence-gate-dflash.sh` still pending (defrag is default-off; run the coherence
  gate before enabling it by default).

### Phase 4 — Low-rank cold residual (deferred alternative — design-lever #2)

Only if Phase 2 fails its gate, or a long-context regime makes per-token
recoverability matter (explore measured rank-64 SVD KV at cos 0.991 @256B). Store
group mean + rank-r correction, SVD computed in the idle/between-turns budget.
Gate on KLD beating flat-mean at equal bytes. Keep parked unless triggered.

## Deferred / closed — do not re-tread

- **Rotation + ConQuR on cold tiles** (follow-up #6): Sinkhorn variance-norm
  already does the incoherence job; 2-bit proved quant is not the bottleneck.
  Phase 1 confirmed 1-bit is quant-limited (PPL 26.5→192 at fold=1), but 2-bit
  already gives the storage at ~free quality, so rescuing 1-bit with rotation is
  **not worth its runtime cost** — stays closed.
- **attention-mass importance** (`_IMPORTANCE=attn`): documented worse than vnorm
  (recent-window bias). Do not repeat.
- **HoloKV / superposition:** closed negative (`docs/reports/2026-06-22-holokv-report.md`)
  — dominated by CASK+KVarN on real diffuse attention. Prior art only.
- **Latent-KV static/equivariant rank-32 basis:** rejected across scale
  (`2026-07-11-latent-kv-large-model-confirmation.md`). Separate thread.
- **Batched two-tier attention** (follow-up #1c): hier is a memory feature; the
  per-token serial route is correct and sufficient. Build only if batched-prefill
  *throughput* is ever demanded.

## Risk / rollback

Every phase lands behind the existing default-off `HIPFIRE_KV_HIERARCHICAL` gate
(and Phase 2 behind its own encode flag until its gate passes). Baseline stays
byte-identical, so rollback is flipping a flag. The full negative-result history
lives in the auto-memory `[[project_kv_compression_explore]]`.
