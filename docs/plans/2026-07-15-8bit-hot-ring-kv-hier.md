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

### Key simplification — pre-rotate the cold tier (collapses the fork toward A)

Approach A's only real downside was the **rotation mismatch**: the cold tier is
un-rotated (Sinkhorn only, `rotate=false`), so an FWHT hot tier needed its own
`q_rot` separate from the cold read's `q`. **Fix: also FWHT-rotate the cold tier**
(`compact_cold_kv(rotate=true)` in `migrate_n`/defrag). Verified this is sound:
- The codec pipeline is **merge → FWHT(K) → Sinkhorn → affine** (kv_compact.rs
  ~L213 rotates `kvec` *before* `quantize_tile`, which runs `variance_normalize`).
  So FWHT does **not** stop Sinkhorn — Sinkhorn balances the rotated tile
  (standard incoherence stack). `rotate=false` today is "Sinkhorn alone suffices",
  not "rotation breaks it"; expect neutral-to-slightly-better cold quality.
- **V stays un-rotated** (per-slot Q8, as now). Only K is rotated, so
  `q_rot·K_rot = q·K` and the attention output (Σ w·V over un-rotated V) needs
  **no inverse rotation**.

With both tiers rotated, the two-tier read rotates the query **once** → `q_rot`
and uses it for both K reads — one unified rotated path, no `q`/`q_rot` split.

### Recommendation (updated)
**Approach A with a pre-rotated cold tier.** The per-token FWHT-affine hot ring
(~47% savings, fits the ring) plus `compact_cold_kv(rotate=true)` unifies the
rotation: one `q_rot`, one read path, K-only rotation, V un-rotated. This is both
the smaller hot tier *and* the simpler kernels the whole exercise was after.
Approach B (unmerged kvarn blocks + fp16 window) remains the fallback if the
per-token affine+FWHT ring kernel proves harder than expected; it reuses the cold
read but keeps the block/window overhead (~34–42%).

**New verification step:** A/B the cold tier at `rotate=true` vs `rotate=false`
(KLD vs bf16 + `parity_cold_4bit_read`) to confirm quality is neutral/better
before flipping it — it changes stored bytes' meaning (query must now be rotated).

## Implementation phases (Approach A, rotated frame)

The rotation lives at the **hot write**; the cold tier inherits it through migrate
(no double-rotate, no separate cold-rotate flip). One `q_rot` for the whole read.

0. **Prove the rotated frame first (lowest-risk, no new codec).** Flip the *hot
   write path to rotate* by making `append_token` FWHT-rotate each token's K
   before the existing f16 store (V un-rotated), and keep `migrate_n` at
   `compact_cold_kv(rotate=false)` (K already rotated → cold inherits it). Rotate
   the query once (`mq_rotate_x` → `q_rot`) and use it for BOTH the hot and cold
   K reads in `two_tier`. This changes *only* the frame (still f16 hot), so it is
   a clean, separately-validatable step: `parity_kv_hier` / `parity_two_tier_e2e`
   must still pass, and a KLD-vs-bf16 A/B must be neutral. Migrate math check:
   `q_rot·(H·K_merged) = q·K_merged` since H is orthogonal and linear commutes
   with the merge average.
1. **Hot codec = per-token 8-bit affine (on the already-rotated K).** Add a GPU
   per-token/slot 8-bit affine quant + dequant for the ring (V too). Close to the
   `q8_0` slot path; the rotation is already done in step 0, so the codec is just
   affine. Store slot-major 8-bit instead of f16.
2. **Hot read.** dequant 8-bit hot → transient shared f16 scratch → the existing
   `attention_cold_slots` layout-2 read with `q_rot`.
3. **Migrate.** dequant 8-bit hot → f32 (instead of `widen` f16) → the existing
   `compact_cold_kv(rotate=false, bits=8)` (K already rotated).
4. **Knobs + default.** `HIPFIRE_KV_HOT_BITS` (default 8; 4/8), keep f16 hot
   selectable for A/B. Wire into the default policy (`project-kv-default-kvarn-hier`).
5. **Bit accounting.** Log per-session hot-tier bytes so the multi-session win is
   visible in `hipfire doctor`/telemetry.

Invariant throughout: **only K is rotated; V stays per-slot un-rotated**, so the
attention output needs no inverse rotation.

Fallback = Approach B (unmerged kvarn blocks + fp16 window) if the per-token 8-bit
affine ring proves harder than the rotate-in-place step 0 suggests.

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
