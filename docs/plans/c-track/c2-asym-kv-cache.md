# C2 — Asym3 quantize the draft's k_ctx_cached / v_ctx_cached

**Device:** R9700 #2 on hiptrx (`ROCR_VISIBLE_DEVICES=2`)
**Branch:** `feat/c2-asym-kv-cache` (off `perf/dflash-phase1-target-hidden-collapse`)
**Target saves:** **~0.8-1.2 GB at ctx=64K, ~1.6-2.5 GB at ctx=128K**
**Quality risk:** MED-HIGH — cache precision affects drafter attention numerics

## What this lever quantizes

After B2, the draft's per-layer K/V cache is F16:
- `k_ctx_cached[i]`: `[L, kv_dim]` F16 per layer × 5 layers = 1.64 GB at 64K
- `v_ctx_cached[i]`: same = 1.64 GB at 64K

C2 drops these from F16 to either:
- **INT8** (`Q8_0`-style): ~50% smaller than F16 → saves 0.82 GB at 64K
- **asym3** (3-bit, K only — V stays F16 or Q8): ~75% smaller than F16 on K → saves 0.61 GB on K + 0 on V = 0.61 GB at 64K
- **asym3** (K) + **Q8** (V): ~62.5% smaller combined → saves ~1.0 GB at 64K
- **asym4** (4-bit K and V): ~50% smaller → ~0.82 GB at 64K

Recommend **asym3 K + Q8 V** as the first try: mirrors the target's
KV-cache convention (asym3 K + Q8 V is already shipped in the target
path on master). Reuses existing asym3 dequant kernel infrastructure.

## Plan (asym3 K + Q8 V variant)

1. Change `k_ctx_cached[i]` alloc to asym3 (use existing asym3 K
   layout from target KV: `4 + head_dim/2` bytes per (kv_head, pos)
   → for the draft, kvd = 1280, head_dim = 160, so per-pos K =
   `4 + 80 = 84 bytes / kv_head × 8 kv_heads = 672 bytes / pos`).
2. Change `v_ctx_cached[i]` alloc to Q8: `head_dim/32 * 34 = 5 * 34 =
   170 bytes / kv_head × 8 = 1360 bytes / pos`. Wait that's wrong —
   verify against `llama.rs:new_gpu_q8` for the right Q8 byte layout
   on this kv-dim.
3. **wk/wv write side**: existing B2 path writes F32 → narrow to F16.
   Replace with F32 → **asym3-quantize-on-write** for K and
   **Q8-quantize-on-write** for V. Use existing quantize helpers
   from `llama.rs` if available; otherwise port the kernel pattern
   from the target KV write path.
4. **Concat path**: replace `convert_f16_to_f32_to(cache → k_cat)`
   with a dequantize step. asym3 K → F32 k_cat is the existing
   `kv_cache_write_asym_k` REVERSE direction — find or write the
   dequant kernel. Q8 V → F32 v_cat ditto.
5. **k_norm rmsnorm**: currently runs on F32 chunk inside the chunk
   loop (B2). Keep as-is — it runs BEFORE the asym3 quantize, on
   the F32 wk output. The rmsnorm output then gets quantized.
6. **Eviction mirror** (`apply_eviction_retain_to_draft` in
   speculative.rs): currently downloads target_hidden as F16
   (after B3). The K/V caches it manipulates also need asym3/Q8
   handling. **Cross-check**: with B3 already F16 on
   target_hidden, the eviction path already needs updating. C2
   extends that to the asym3/Q8 cache format.

## Watch out for

- **Quality risk is real** but **the memory's τ=4.71 baseline number is wrong**: a 2026-05-16 bisect found Roman-empire q8 prose τ at ~1.0-1.2 on EVERY commit from master through HEAD across max=120/192/300. The original memo's "0.88 → 4.71" figures aren't reproducible. **Don't use τ > 4.0 as a stop-the-bus gate.** Instead, compare to master at the same max length on the byte-exact prompt — your baseline is ~1.15 not ~4.71. The DRAFT cache is internal-only (not the cross-attention input from target), so precision drift here may behave differently from target-side KV quantization. Run all 4 short coherence cases (asym3 + q8) as the primary validation; long-prose τ comparison is informational, not a hard gate.
- **Per-cycle write overhead**: asym3 quantize is more compute
  per byte than F16 narrow. Expect a small per-cycle perf hit;
  measure with the canonical bench.
- **wk output rmsnorm**: K cache rows are NORMED (per-head RMSNorm)
  before they're stored. If asym3 K stores pre-norm and the dequant
  applies the norm, that's wrong. Keep the F32 → rmsnorm → quantize
  order from the existing B2 chunk loop.
- **Cross-cycle replay**: the cache is read every cycle but only
  the delta is WRITTEN. If quantize is per-write only, you don't
  re-quantize old rows — good. But verify the kvd quantization
  scale is per-row (not per-cache), so different rows can have
  different scales.

## Implementation order

1. (`c2-asym-k-only`): change K alloc + write + dequant only. V
   stays F16. Test: canonical bench + coherence gate (both asym3
   and q8 KV-mode for target). Expected: ~0.6 GB saved.
2. (`c2-q8-v`): change V to Q8. Test again. Expected: another
   0.4-0.5 GB saved.
3. (`c2-eviction-mirror`): update apply_eviction_retain_to_draft
   to handle the asym3/Q8 cache shapes correctly. Test with a
   long-context bench that triggers eviction.

## Validation matrix

| Test | Command | Pass condition |
|---|---|---|
| Build | `cargo build --release --features deltanet --example dflash_spec_demo` | clean |
| Canonical bench | `dflash_spec_demo ... --max 256 --kv-mode asym3 --no-chatml` | τ=13.2727 ±5 % each step (NOT byte-exact; quantize introduces drift) |
| Coherence gate asym3 (canonical) | `HIPFIRE_FORCE_SPEC_GATE=1 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Coherence gate q8 | `HIPFIRE_FORCE_SPEC_GATE=1 HIPFIRE_GATE_KV_MODE=q8 ./scripts/coherence-gate-dflash.sh` | no hard errors |
| Long-prose 1 | run with 800-tok prose prompt, max=300, asym3 | τ stays within ±15% of master at same prompt/length |
| Long-prose 2 | same with q8 | τ stays within ±15% of master at same prompt/length (master ≈ 1.15 on Roman-empire q8 — don't use 4.71) |
| Ctx bisect | as in coordination doc | ceiling lifts ≥6K on 24GB (or ≥10K on hiptrx) |

## Done criteria

- [ ] Each substep gate-clean
- [ ] Coherence gate "no hard errors" on asym3 + q8 short + long-prose
- [ ] τ on canonical drops by < 5 % cumulative
- [ ] Long-prose τ on q8 stays within ±15% of master baseline at same prompt+max-length (master Roman-empire q8 is actually ~1.15, NOT 4.71 — original memory was wrong)
- [ ] Net VRAM saved at ctx=64K ≥ 0.6 GB measured

## Handoff prompt

```
Implement C2 — quantize the draft's per-layer k_ctx_cached /
v_ctx_cached from F16 to asym3 (K) + Q8 (V), mirroring the target's
KV-cache convention. Detailed plan at
`docs/plans/c-track/c2-asym-kv-cache.md`.

Your branch: `feat/c2-asym-kv-cache` (off
`perf/dflash-phase1-target-hidden-collapse` HEAD 2533a1b4). Your
worktree on hiptrx: `~/hipfire/.worktrees/c-track-c2-asym-kv/`. GPU:
device 2 (`export ROCR_VISIBLE_DEVICES=2`).

This lever has REAL quality risk — asym3 K on the target broke
DFlash drafter prose τ (memory entry
project_dflash_net_loss_on_prose_2026_05_15). The draft cache is
internal-only (not the cross-attention input), so the behavior may
differ — but you MUST validate prose τ doesn't collapse.

Substep order:
  1. (`c2-asym-k-only`): K alloc + quantize-on-write + dequant-on-read.
     V stays F16. Coherence + canonical + 1 long-prose at q8.
  2. (`c2-q8-v`): V to Q8. Repeat validation.
  3. (`c2-eviction-mirror`): update apply_eviction_retain_to_draft
     for the new cache shapes. Long-context bench that triggers
     eviction.

Reuse the target's asym3/Q8 helpers from `llama.rs` where possible.
Per-head rmsnorm on K runs BEFORE quantize (preserve B2 order).
Per-row quantization scales (not per-cache).

Validation: canonical + coherence-gate-dflash on asym3 + q8 + a
long-prose bench at 800 tokens prose / max=300 on q8. If prose τ
drops below 4.0 (pre-C2 baseline ~4.71), revert and re-think.

Commit each substep separately.

Don't push to origin without coordinator approval.

Begin by reading `docs/plans/c-track/c2-asym-kv-cache.md` and
`docs/plans/c-track-parallel-coordination.md`.
```
