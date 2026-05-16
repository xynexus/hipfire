# C-track parallel push toward 128K-on-24GB DFlash asym3

**Status:** Coordination doc, 2026-05-16. Spawns 4 parallel Claude Code
sessions, one per R9700 on hiptrx, each working on a separate C-track
VRAM lever stacked on top of the B3+B1+B2 F16 cascade.

**Baseline (everyone branches from):**
`perf/dflash-phase1-target-hidden-collapse` HEAD `2533a1b4` (pushed to
origin + hiptrx).

## Why parallelize

The C-track levers identified in the asym3-128K plan are independent
in code (different files, different kernels) and quality-orthogonal
(each has its own validation bench). Four R9700s on hiptrx, each idle,
let us iterate on all four levers concurrently. With
`ROCR_VISIBLE_DEVICES=N` per agent, GPU contention is avoided without
needing the (still-unbuilt) GPU-TCAS daemon.

The advisory `scripts/gpu-lock.sh` doesn't gate per-device, but each
agent's `ROCR_VISIBLE_DEVICES=N` is enforced at the ROCm runtime level
— processes can only see / allocate on their assigned device.

## Assignment

| Dev | Branch | Lever | Saves @ 64K | Quality risk | Owner |
|---|---|---|---:|---|---|
| 0 | `feat/c1-fused-fc-wk-wv` | C1 — fused FC+wk/wv (eliminate target_hidden_proj) | 0.66 GB | LOW (math unchanged, no intermediate buffer) | Agent A |
| 1 | `feat/f16-attention` | F16 k_cat/v_cat + F16 attention_dflash + F16 RoPE | 0.33 GB | LOW-MED (F16 attn with F32 accumulate) | Agent B |
| 2 | `feat/c2-asym-kv-cache` | C2 — asym3 k_ctx_cached / v_ctx_cached (drop from F16) | ~0.8 GB | MED-HIGH (drafter internal precision) | Agent C |
| 3 | `feat/c3-asym-target-hidden` | C3 — asym3 target_hidden (drop from F16) | ~1.6 GB | HIGH (drafter cross-attn on quantized hidden) | Agent D |

If all four ship: ~3.4 GB more saved at 64K ≈ +24K context ceiling on
24 GB → ~89K. Still short of 128K but a significant step. The
remaining ~5 GB to 128K likely needs C4 (drafter retrain w/ ne=5→3)
or a deeper architectural rework — out of scope for this push.

## Per-agent rules of engagement

Each agent reads its dedicated plan doc + handoff prompt and:

1. **Branches off** `perf/dflash-phase1-target-hidden-collapse` —
   they all inherit the B3+B1+B2 F16 cascade.
2. **Works in a dedicated worktree on hiptrx**:
   `~/hipfire/.worktrees/c-track-<short-name>/`. Pre-created (see
   "Setup commands" below) or self-created with
   `git -C ~/hipfire worktree add <path> <branch>`.
3. **Pins `ROCR_VISIBLE_DEVICES=N`** in their shell, where N matches
   their assigned device per the table above. Verify with
   `rocminfo 2>&1 | grep -c "Marketing Name:.*R9700"` → should print
   `1` (their single visible device), and `gpu_status` should print
   the host-wide advisory state (the `gpu-lock.sh` lock is still
   whole-host advisory — agents shouldn't acquire it; rely on
   ROCR_VISIBLE_DEVICES isolation for correctness).
4. **Bench protocol** — canonical merge_sort gate:
   ```bash
   ROCR_VISIBLE_DEVICES=N ./target/release/examples/dflash_spec_demo \
       --target ~/.hipfire/models/qwen3.5-27b.mq4 \
       --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
       --prompt "$(cat benchmarks/prompts/merge_sort_thinking_off.txt)" \
       --max 256 --kv-mode asym3 --no-chatml
   ```
   Expected on gfx1201/R9700: ~46 tok/s short-ctx, τ=13.2727 byte-exact.
   *Don't* gate on absolute tok/s (gfx1201 is unoptimized) — gate on
   `τ=13.2727 exact` + token sequence byte-match the pre-change
   baseline.
5. **Coherence gate** must pass at each phase:
   ```bash
   ROCR_VISIBLE_DEVICES=N HIPFIRE_FORCE_SPEC_GATE=1 \
       ./scripts/coherence-gate-dflash.sh
   ```
   Must report "no hard errors". For levers C2 / C3 (quality-risk),
   also run `HIPFIRE_GATE_KV_MODE=q8` as a second gate to catch
   regressions outside the asym3 path.
6. **Ctx-bisect verification** — after each phase ships, run:
   ```bash
   for ctx in 65536 81920 98304 114688 131072; do
       ROCR_VISIBLE_DEVICES=N ./target/release/examples/dflash_spec_demo \
           --target ~/.hipfire/models/qwen3.5-27b.mq4 \
           --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
           --prompt "$(cat benchmarks/prompts/merge_sort_thinking_off.txt)" \
           --max 10 --kv-mode asym3 --no-chatml --ctx $ctx 2>&1 | \
           grep -E "vram_used_mb|OOM|panicked|decode_tok_s"
   done
   ```
   Report the highest ctx that passes + VRAM used.
7. **Commit messages**: include per-buffer alloc deltas (the
   bytes-saved table from `c-track-parallel-coordination.md` style).
   Each commit message must show: ctx-ceiling on 24 GB (k9lin) AND
   32 GB (hiptrx), peak VRAM at ceiling, gate result.
8. **Don't touch other agents' files.** Stay in your own scope.
   Cross-lever interactions (e.g. C1 changes the dispatch layout
   that C2 wants to extend) get resolved at merge time by the
   coordinator (me — or whoever picks up the integration).

## Quality validation thresholds (per agent)

| Lever | Min gate result | Max acceptable τ drift on canonical | Notes |
|---|---|---|---|
| C1 | gate "no hard errors" on asym3 + q8 | < 0.5 % (byte-exact preferred) | structural change, should be byte-exact |
| F16 attention | gate "no hard errors" on asym3 + q8 | < 2 % | F16 attn introduces fp drift |
| C2 (asym KV) | gate "no hard errors" on asym3 + q8 + long-prose | < 5 % on code, < 10 % on prose | quantize cache — track prose carefully |
| C3 (asym target_hidden) | gate "no hard errors" on asym3 + q8 + long-prose + 16K context | < 5 % on code, < 15 % on prose | drafter cross-attn precision — riskiest |

If gate fails or τ drift exceeds threshold: agent's commit is REJECTED
from the integration, and the lever is parked with an empirical "what
broke" report.

## Integration order (after agents ship)

When all four agents report green on their levers, the integration
sequence is:

1. **C1 first** — pure structural, no math change, minimal risk.
   Rebase C1 onto current master (which already has B3+B1+B2 via
   PR landing). Merge as a single PR.
2. **F16 attention** — independent of cache changes. Rebase on
   post-C1 master, merge.
3. **C2 (asym KV) second-to-last** — needs to land on top of the
   F16 cascade; verify the F16 wk/wv outputs interact correctly
   with the asym3 quantize-on-write path. Coherence-gate
   exhaustively before merge.
4. **C3 (asym target_hidden) last** — biggest risk, depends on the
   F16 cascade staying clean. Land alone, with extended coherence
   validation (3-prompt prose battery + 16K-prefix code gate).

## Setup commands — pre-create the 4 worktrees on hiptrx

(Run from your control node before spawning agents, OR include
in each agent's prompt.)

```bash
ssh hiptrx '
  cd ~/hipfire
  git fetch origin perf/dflash-phase1-target-hidden-collapse:perf/dflash-phase1-target-hidden-collapse 2>/dev/null || \
      git fetch origin perf/dflash-phase1-target-hidden-collapse
  for spec in \
      "c1-fused:feat/c1-fused-fc-wk-wv" \
      "f16-attn:feat/f16-attention" \
      "c2-asym-kv:feat/c2-asym-kv-cache" \
      "c3-asym-th:feat/c3-asym-target-hidden"; do
    short="${spec%%:*}"
    branch="${spec##*:}"
    if ! git rev-parse --verify "refs/heads/${branch}" >/dev/null 2>&1; then
      git branch "${branch}" origin/perf/dflash-phase1-target-hidden-collapse
    fi
    if [ ! -d ".worktrees/c-track-${short}" ]; then
      git worktree add ".worktrees/c-track-${short}" "${branch}"
    fi
  done
  git worktree list | grep c-track
'
```

## Out of scope for the agents

- Don't push to origin without coordinator approval (use `git push hiptrx <branch>` locally if you want a backup).
- Don't merge into `perf/dflash-phase1-target-hidden-collapse` directly — that's the shared baseline.
- Don't run `scripts/gpu-lock.sh gpu_acquire` (host-level lock, blocks other agents). Rely on ROCR_VISIBLE_DEVICES for isolation.
- Don't touch `crates/gpu-tcas/` (separate parallel work on `feat/gpu-tcas`).
- Don't migrate the wave-1 lock callers — that's wave-2 of GPU-TCAS.
- Don't run gfx1100 cross-validation — coordinator handles that on k9lin during integration.
