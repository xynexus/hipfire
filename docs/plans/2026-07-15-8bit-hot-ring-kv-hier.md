# Plan: 8-bit hot tier for the hierarchical KV cache

Status: **active** (design locked; implementation not started). Branch: `chaingun`.
Date: 2026-07-15. Companion: `docs/todo/kvarn-hot-bitwidth.md` (findings + probe).

## Context

The hierarchical KV cache (`HIPFIRE_KV_HIERARCHICAL=1`) keeps the most recent
`hot_budget` (512) tokens as an **exact f16 ring** (hot tier) and compacts older
tokens into kvarn-quantized cold segments. The f16 hot ring is `hot_budget ×
kv_dim × 2` B/layer ≈ **16 MB per session**.

Single-session that's negligible — but hipfire is built to batch **thousands of
concurrent sessions** on one Strix Halo (throughput comes from multi-session
batching). At ~1000 sessions that's ~**16 GB** of hot KV; making the hot tier
8-bit frees ~**8 GB** on a 128 GB box → materially more concurrent sessions /
longer contexts. That is the motivation; the per-session number is irrelevant.

8-bit KV is near-lossless at the model level (measured KV-only KLD 1.1e-5;
`docs/todo/kvarn-hot-bitwidth.md` §Model-level KLD), so dropping f16 costs nothing
meaningful.

## What's already settled (don't re-derive)

- **The hot tier needs a rotation, not just quantization.** Codec probe
  (`cargo run -p hipfire-kvquant --example kvarn_precision_sweep`), realistic K
  tile, 8-bit: kvarn(Sinkhorn) attn-KLD 3.8e-4, **plain per-token affine 8.1e-3
  (~21× worse — outlier channels)**, **per-token affine + FWHT 5.5e-4 (≈ kvarn)**.
  So a per-token 8-bit affine hot ring MUST rotate K; plain affine is a quality
  regression.
- **`signed_fwht` is orthonormal** (1/√n), inverse = call with signs swapped; so
  rotating K (store) + the query (read) preserves q·K.
- **The cold tier is NOT rotated.** `migrate_n`/defrag call
  `compact_cold_kv(rotate=false)` — Sinkhorn variance-norm does the incoherence
  job, no FWHT, so the **cold read uses the un-rotated query** (`kv_hier.rs:665`,
  `kvarn.rs:38`). This is the crux of the fork below.

## Current hot-tier code (change surface)

`crates/hipfire-runtime/src/kv_hier.rs`:
- Struct: `hot_k`/`hot_v` = `Vec<GpuTensor>` `[nkv × hot_budget × HD]` **F16**,
  slot-major (`~L159`, alloc `~L289`).
- Write: `append_token` (`~L372`) — `cast_f32_to_f16` per token into the ring slot.
- Migrate: `migrate_n` (`~L407`) — `download_raw` f16 → `widen`→f32 →
  `compact_cold_kv(rotate=false)` into a cold segment.
- Read: `two_tier` attend (`~L840`) — hot via `attention_cold_slots(..., k_layout=2,
  v_layout=2)` (slot-major f16), each cold seg via `kvarn_dequant_tile` →
  `attention_cold_slots(k_layout=1)` → `flash_tier_merge`.

## The fork (decide first)

Both halve the hot tier; they differ in how K's outlier channels are handled and
how much they reuse the cold path.

### Approach A — per-token 8-bit affine on FWHT-rotated K (fits the ring)
- Store hot K FWHT-rotated + per-token 8-bit affine; hot V per-token 8-bit affine
  (no rotation — V has no outlier pathology).
- **Cost:** the hot read needs a **separate FWHT-rotated query** `q_rot`
  (`mq_rotate_x` exists, `dispatch/rope.rs`), because the cold read uses the
  un-rotated `q`. So the two-tier read carries both `q` (cold) and `q_rot` (hot).
- **Savings ~47%** (no window). Needs a GPU per-token affine(+FWHT) quant/dequant
  for the ring (close to an "fwht8" of `attention_flash_fwht*`/`q8_0` kernels).
- Quality ≈ kvarn (probe 5.5e-4).

### Approach B — unmerged 8-bit kvarn blocks + small fp16 window (reuses cold path)
- Hot tier = completed N-token blocks kvarn-quantized **unmerged** (`fold_m=1`,
  Sinkhorn, `rotate=false` — same as cold) + an fp16 window for the current
  partial block. The hot blocks are literally **unmerged cold segments**, read by
  the **existing** cold loop (`kvarn_dequant_tile` + layout-1 + merge) with the
  **un-rotated query** — no `q_rot`, no new kernel, no new layout.
- **Cost:** the fp16 partial window is fixed overhead. Block size N is a knob:
  N=128 → ~34% savings; N=32 → ~42% (smaller window, more scale overhead).
- Quality = kvarn (3.8e-4). Migrate moves whole blocks (no re-quantize).

### Recommendation
**Approach B**, block size 32–64. Rationale: it reuses the *exact* cold read path
and the un-rotated query, so it adds the least new surface to the most
coherence-sensitive code (no new kernel, no `q_rot` plumbing, no rotation/tier
mismatch). It conceptually unifies hot+cold into kvarn segments (merged vs
unmerged). Approach A saves ~5% more but adds a kernel + a rotated-query path +
the hot/cold rotation split — more risk for modest extra memory. Revisit A only if
the block/window overhead proves too costly at the target block size.

## Implementation phases (Approach B)

1. **Hot-tier representation.** Replace the per-layer f16 ring with: (a) a small
   fp16 partial-block window `[nkv × N × HD]`, and (b) a `Vec<ColdSegmentGpu>` of
   unmerged hot blocks (reuse `ColdSegmentGpu`). Keep counts per layer.
2. **Write (`append_token`).** Append to the fp16 window; when it reaches N,
   `compact_cold_kv(fold_m=1, rotate=false, bits=8)` the window into a hot block,
   reset the window. (Reuses the migrate quantize path.)
3. **Read (`two_tier`).** Fold the fp16 window (`attention_cold_slots` layout 2,
   its live count) + each unmerged hot block (the existing cold-segment read) +
   each merged cold segment, all via `flash_tier_merge`. Un-rotated `q` throughout.
4. **Migrate.** Move whole unmerged hot blocks into the cold list and merge them
   (`fold_m>1`) — they're already the right record type, so this is cheaper than
   the current widen-f16 → re-quantize.
5. **Knobs + default.** `HIPFIRE_KV_HOT_BITS` (default 8; 4/8) and hot block size;
   keep the f16 path selectable for A/B. Wire into the default policy per
   `project-kv-default-kvarn-hier`.
6. **Bit accounting.** Log the per-session hot-tier bytes so the multi-session win
   is visible in `hipfire doctor`/telemetry.

## Reuse (do not re-invent)

`compact_cold_kv` (`fold_m=1` = unmerged), `ColdSegmentGpu`, `kvarn_quantize_tile`
/ `kvarn_dequant_tile` (bits ∈ {2,4,8} already), `attention_cold_slots`
(layouts 1/2), `flash_tier_merge`. Approach A additionally: `mq_rotate_x`
(`dispatch/rope.rs`), `attention_flash_fwht*`/`q8_0` tile kernels.

## Verification (staged, coherence-critical)

- Per stage: `cargo build --release --features deltanet -p hipfire-runtime`.
- Parity oracles FIRST (CPU/GPU, exact): `parity_kv_hier`,
  `parity_two_tier_e2e`, `parity_cold_4bit_read`, `parity_attention_cold_slots`,
  `parity_flash_tier_merge` — extend/re-run for the 8-bit hot tier.
- `coherence-gate-dflash.sh` (KV/dispatch gate; note: 27B-dflash cases SKIP on
  nix2 without staged pairs — run on halo or with a local pair for real coverage).
- End-to-end: hierarchical decode on qwen3.5-9b, needle recall + a long-context
  KLD-vs-bf16 delta (`--kv-mode kvarn --kv-hierarchical` + the KLD bridge) — 8-bit
  hot must match f16 hot within noise. Coordinate GPU with `hipfire lock`.

## Risks

- **Coherence.** The two-tier read is the most decode-sensitive path; a subtle
  layout/merge error is silent corruption. Gate every stage on the parity oracles
  before the model-level run.
- **Partial-window boundary.** Off-by-one at the window→block transition (which
  tokens are exact vs quantized) is the likely bug; test at N-1/N/N+1 tokens.
- **Commit cost.** The pre-commit hook runs the full coherence battery on eval/
  runtime changes (~5 min) — commit in the background.
