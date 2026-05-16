# Next-session pickup — finish asym3 128K ctx unlock

**Read me first when picking this back up.**

## Status snapshot (EOD 2026-05-15)

- ✅ Landed via PR #261 (master `4e37618b`):
  - KV cache filter for LinearAttention layers (1.24 GB saved at 17K)
  - mq_x_rot first-call FWHT chunking (1.74 GB saved constant)
  - 3 tier-1 trivial cleanups (env caching, MQ3 chunk-loop fix)
- ✅ Empirical: asym3 ctx ceiling **16K → 24K (+50%)** on 7900 XTX 27B
- ❌ Target: ~128K (asym3 KV math fits ~12 GB on 24 GB card)
- 🔲 Gap: ~100 KB/token of context-linear non-KV scratch buffers

## Why the agent's "post-fix 64K" estimate was wrong

The two bugs fixed account for ~3 GB saved at ctx=17K. The OTHER context-linear buffers (which the agent flagged as "tier-3") are still in play:

- `DflashScratch.target_hidden`: 6.55 GB at ctx=64K (interleaved layout)
- `HiddenStateRingBuffer.layer_bufs`: 6.55 GB at ctx=64K (per-extract layout) — **duplicate of target_hidden**
- Per-layer k_ctx_cached + v_ctx_cached: 2.6 GB at ctx=64K (5 layers × full max_ctx)
- target_hidden_proj F32: 1.34 GB at ctx=64K

The two top items hold the SAME hidden-state payload, just in different layouts. The bridging logic already exists: `scatter_hidden_block_to_interleaved` at `speculative.rs:2277` translates between them every cycle.

## Phase 1 — target_hidden + hidden_rb collapse (THE big win)

**Estimated: 3-5 days. Risk: yellow (single-file refactor + verify with coherence-gate).**

**Approach:** keep ONE canonical buffer + a permutation/scatter helper. Consumers that need the other layout read through the helper. Saves ~6.4 GB at ctx=64K. **Estimated to unlock ctx 24K → ~64K.**

**Files to touch:**
- `crates/hipfire-runtime/src/dflash.rs` — `DflashScratch.target_hidden` field + populator (`draft_forward` body, ~lines 700-800)
- `crates/hipfire-arch-qwen35/src/speculative.rs` — `HiddenStateRingBuffer.layer_bufs` field + `commit_staging_to_ring` + `scatter_hidden_block_to_interleaved`

**Verification protocol:**
1. Build: `cargo build --release --features deltanet --example dflash_spec_demo -p hipfire-runtime`
2. Canonical bench (must still hit 254-256 tok/s τ=13.27 on merge_sort_thinking_off.txt, asym3, ctx=2048)
3. Ctx bisect: 32K / 49K / 64K / 80K with asym3 — should fit at 64K
4. Coherence-gate-dflash (if GPU free): `HIPFIRE_FORCE_SPEC_GATE=1 ./scripts/coherence-gate-dflash.sh`

## Phase 2 — per-layer k_ctx_cached / v_ctx_cached restructure

**Estimated: 2-3 days. Risk: yellow.**

Options (probably combine):
- (a) F16 quantize the per-layer K/V cache (~750 MB savings at ctx=17K, 2.6 GB at ctx=64K) — audit T4.3, needs τ A/B
- (b) Single shared K/V tensor with per-layer slicing (eliminates per-layer Vec<GpuTensor> overhead)
- (c) Lazy allocation (only fully-allocate for layers currently in-use)

Most likely path: (a)+(b). Verify with same protocol as Phase 1.

## Phase 3 — minor mop-up

- `target_hidden_proj` F32 → F16 (1.34 GB at 64K)
- `k_cat` / `v_cat` F16
- Remaining audit tier-2 items

**Estimated to push ctx 64K → ~96K-128K.**

## Reference docs (read these first when picking up)

- `docs/plans/dflash-vram-bloat-2026-05-15.md` (master) — original investigation + Phase 1/2 designs
- `docs/audits/2026-05-15-bloat-audit.md` (this branch) — 28-item bloat catalog with all the supporting findings
- Memory: `project_next_session_asym3_128k_unlock.md`

## Do NOT do without explicit reason

- Don't switch the canonical bench to a different prompt — `merge_sort_thinking_off.txt` is the verified anchor at 254-256 tok/s τ=13.27 on current master
- Don't enable q8 globally as the "solution" — it costs context budget, the K-split idea preserves both. But also: q8 globally is fine as a temporary opt-in for prose users while T4.1 work proceeds
- Don't attempt Phase 1+2 as one PR — keep them separate so each is reviewable + revertable independently
