# MI300x Phase 2 — MTP-extended verify

**Date:** 2026-05-19
**Hardware:** AMD Instinct MI300X VF / gfx942 / wave64, 192GB HBM3, ROCm 7.0.0
**Branch:** `feat/mtp-mi300x`, builds on Phase 1.5 commit
**Goal:** Prototype the `MtpChainMode::ExtendedVerify` variant from the master plan
(`docs/plans/mtp-dflash-composition-master-plan.md`) and measure whether the
discrete-token MTP chain produces a different acceptance rate vs the existing
Task 11 linear-chain (lossy feature-only chain).

## Headline

| Cell                                                       | tok/s          | τ_dflash | τ_mtp  | τ_total | accept_mtp_total |
|------------------------------------------------------------|---------------:|---------:|-------:|--------:|-----------------:|
| Phase 1 (linear-chain) composition reference (`max=120`)   | **112.90**     | 9.1667   | 0.0000 | 10.1667 |                0 |
| Phase 2 (linear-chain) re-bench, 3-run median (`max=120`)  | **111.97**     | 9.1667   | 0.0000 | 10.1667 |                0 |
| Phase 2 (extended-verify) 3-run median (`max=120`)         | **111.80**     | 9.1667   | 0.0000 | 10.1667 |                0 |
| DFlash solo on MI300x (re-measured) (`max=120`)            | **123.95**     | 9.1667   | n/a    | 10.1667 |              n/a |

Delta extended-verify vs linear-chain (Phase 2): **−0.15%** (within stddev).
Delta extended-verify vs DFlash solo: **−9.8%**.

**Result: NEGATIVE.** Extended-verify does NOT lift the K=2 composition above
linear-chain on MI300x at the canonical PEP-8 LRU prompt. Output is **byte-identical**
between modes. `accept_mtp_total = 0` in both modes — same failure mode as Phase 1.

Goal target ("≥ 130 tok/s extended-verify on MI300x") **not met.** Stretch target
("≥ 160 tok/s, stacking past DFlash solo") **structurally infeasible** with the
current MTP head: even step 0's MTP candidate (which is identical between modes)
is rejected ~98% of the time, so the K-th-step improvement that extended-verify
brings cannot recover the composition gap.

## What changed

Two surgical edits, both in this branch's working tree:

### `crates/hipfire-arch-qwen35/src/mtp_compose.rs` (+98 lines)

1. New `MtpChainMode` enum with two variants:
   - `LinearChain` — original Task 11 behaviour. Step 0 uses
     `next_token=drafted[B-1]` + `prev_hidden=drafter_hidden_last`;
     steps 1..K-1 use the lossy substitution
     `next_token_embed=Some(prev_step_t_mtp_out)` that bypasses the
     embedding table. One batched lm_head at end of chain produces all K
     candidates.
   - `ExtendedVerify` — Phase 2 prototype. Step 0 identical to linear-chain.
     Steps 1..K-1 close the loop with a per-step `mtp_head_apply_lm_head_batched`
     of size N=1 + `argmax_f32_batched` of size N=1, producing a discrete token.
     Step k+1 then re-enters the block with `next_token=that_token` and
     `next_token_embed=None` (standard embedding-table lookup).

2. New public `spec_step_dflash_mtp_with_mode(..., mode)` that accepts the
   mode parameter. The existing `spec_step_dflash_mtp` is now a wrapper
   passing `MtpChainMode::LinearChain` for backwards compatibility — no
   callers had to change.

### `crates/hipfire-runtime/examples/dflash_mtp_demo.rs` (+18 lines, −3)

1. New `--mtp-mode {linear-chain,extended-verify}` flag (default
   `linear-chain` to preserve baseline).
2. Surfaces the active mode in the bench header + the printed metrics block.
3. Dispatches via `spec_step_dflash_mtp_with_mode`.

Both flags are **opt-in**. Existing callers continue to see linear-chain
behaviour; no default changes.

## Why both modes produce identical output and identical acceptance

`accept_mtp_total = 0` is the structural problem. The trunk verify accepts
composite candidates only when each `composite[i+1]` matches the trunk's
greedy argmax at position `position + i`. The MTP candidates live at slots
`B..B+K-1` of the composite — and they are reached only after **all** B-1
DFlash drafts are accepted (full-DFlash cycles). On the canonical PEP-8 cell:

- `full_dflash_cycles = 2` out of 12 → trunk's MTP-slot logits are checked
  in only those 2 cycles.
- Within those 2 full-DFlash cycles, neither `mtp_candidates[0]` (identical
  in both modes, same step 0 inputs) nor `mtp_candidates[1]` (different
  between modes) matched the trunk's argmax.

The deeper issue is that **step 0's MTP prediction is itself OOD relative
to the trunk's distribution at position `position + B`.** Both modes use
identical step-0 inputs (drafter's hidden + drafter's drafted[B-1]), so
both produce the same step-0 prediction. Extended-verify changes step
1..K-1's predictions to be in-distribution for the MTP head, but those
slots are NEVER reached unless step 0 is accepted — which doesn't happen.

This is a "MTP head's hidden-state alignment with the trunk's verify-position
hidden state" problem, exactly as the task brief anticipated ("if extended-verify
produces zero MTP accepts (same failure mode as linear-chain), document why.
It's possible the MTP head's hidden-state alignment with the trunk's
verify-position hidden state is the actual blocker"). Confirmed.

## Per-cycle tax (rocprof confirmation)

12-cycle canonical bench at `max=120, K=2, q8 KV, --no-chatml`:

| Kernel                                  | LinearChain calls | ExtendedVerify calls | Δ |
|-----------------------------------------|------------------:|---------------------:|---:|
| `gemm_hfq4g256_wave64` (MTP head lm_head)|              123  |                 135  |+12 |
| `argmax_f32_batched`                    |                36 |                  48  |+12 |
| `gemm_gate_up_hfq4g256_fp16_wave64`     |              704  |                 704  |  0 |
| `gemm_hfq4g256_residual_fp16_wave64`    |             1408  |                1408  |  0 |
| `gemm_qkvza_hfq4g256_fp16_wave64`       |              528  |                 528  |  0 |
| Cijk Tensile triplet (B=18 verify)      |            1233   |                1233  |  0 |
| `gated_delta_net_q8`                    |             1200  |                1200  |  0 |

Extended-verify adds exactly K-1 = 1 extra lm_head+argmax per cycle (12
cycles × 1 extra per cycle = +12 calls). The extra wall-clock is small
(~3 ms over 12 cycles ≈ 0.25 ms / cycle, ~0.2% of wall time), explaining
the −0.15% perf delta.

Raw CSVs:
- `phase2-linear-chain-kernel-stats.csv` (current branch, linear-chain mode)
- `phase2-extended-verify-kernel-stats.csv` (current branch, extended-verify mode)

The kernel mix is otherwise byte-identical to the Phase 1 composition
reference (`phase1-compose-kernel-stats.csv`) — Phase 1.5's coherence cell
already established this, Phase 2 reconfirms via the linear-chain re-bench.

## Acceptance analysis

| Mode             | cycles | full_dflash | accept_mtp_total | tau_mtp |
|------------------|-------:|------------:|-----------------:|--------:|
| LinearChain (K=2)|     12 |      2 (17%)|                0 |  0.0000 |
| ExtendedVerify (K=2)|  12 |      2 (17%)|                0 |  0.0000 |
| LinearChain (K=3)|     23 |      3 (13%)|                1 |  0.0435 |
| ExtendedVerify (K=3)|  23 |      3 (13%)|                1 |  0.0435 |

At K=3 with `max=200`, **one** MTP candidate gets accepted (0.04 / cycle).
That single accept is identical between modes — confirming that the only
MTP candidate the trunk ever accepts here is step 0's (which both modes
compute identically). Extended-verify's step-1+ candidates are NEVER
accepted because the chain stops at step 0's rejection.

## Coherence

Output **byte-identical** between linear-chain and extended-verify at
`max=120`, `K=2`, canonical PEP-8 prompt. End-of-output line
`lru_node = self.tail.prev` matches the Phase 1 reference. No
panics, no `!!!!!`, no token loops.

Also byte-identical at K=3 (max=200) on the canonical prompt, and at K=2
on the brief-story prompt (`max=200`, `--no-chatml`).

## Recommendation

**Keep extended-verify opt-in default-off, retain in-tree as a research
artifact.** The variant is correct (composition output preserved at the
trunk-verify level), self-isolating (`MtpChainMode` enum + `--mtp-mode`
flag), and adds negligible per-cycle cost. It would become valuable
when paired with a **retrained MTP head** whose step-0 output distribution
better matches the trunk's actual hidden distribution at position `B-1`
post-DFlash. The master plan's Phase 1 (Track A: trained sidecar on hiptrx)
remains the next-priority lever.

**Do NOT default extended-verify on.** The −0.15% perf tax over linear-chain
is small but real, and there's no compensating τ_mtp lift at this MTP
head's current quality.

**Next decision points** (deferred):

- Phase 0 of the master plan called for a "composition probe" — that's
  what this is. The probe outcome is: composition architecturally works,
  but the current MTP head's predictions don't pass trunk verify often
  enough to lift over DFlash solo on MI300x. The MTP head needs better
  training (Track A) or the composition needs an architecture change
  (e.g., per-slot tree composition, already implemented as
  `spec_step_dflash_mtp_tree` but not yet measured on MI300x).
- A useful follow-up bench (5–10 minutes on this droplet): run extended-verify
  with the **3.6-27B mq4-mtp** model (different MTP head training distribution)
  on the same canonical prompt and compare τ_mtp. The prior memory entry
  `mtp-unsloth-target-2026-05-15` notes 3.6 is generally slower than 3.5,
  but per-cycle τ_mtp could differ.

## What I did NOT do

- **Did not** measure tree-mode (`spec_step_dflash_mtp_tree` — per-slot
  composition) on MI300x. The tree variant exists in `mtp_compose.rs`
  but is gated on `MtpComposeTreeState` + `ddtree_scratch` which the
  current demo doesn't allocate. Wiring that up is a separate change.
- **Did not** test 3.6-27B (no model present on droplet, would need a 14 GiB
  pull). Rental spend was a constraint.
- **Did not** ship a default-on flip for `HIPFIRE_GFX942_NATIVE_LM_HEAD`.
  That's still recommended for a follow-up PR per Phase 1.5's findings.

## Reproduction

```sh
cd /root/hipfire     # feat/mtp-mi300x at the Phase 2 commit
export PATH=$HOME/.cargo/bin:/opt/rocm/bin:$PATH

cargo build --release --features deltanet --example dflash_mtp_demo

# Linear-chain (baseline, Phase 1 behaviour)
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/dflash_mtp_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --drafter ~/.hipfire/models/qwen35-27b-dflash.mq4 \
  --mtp-head ~/.hipfire/models/qwen3.5-27b.mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --mtp-k 2 --temp 0 \
  --mtp-mode linear-chain

# Extended-verify (Phase 2 prototype)
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/dflash_mtp_demo \
  ... --mtp-mode extended-verify
```

## Rental state

Cumulative session spend: Phase 0 (~$10) + Phase 1 (~$3) + Phase 1.5 + Phase 2
(~$3 estimated, ~1 hr wall on warm droplet, build cache hit). Total
~$16 / well under the $20 budget. Droplet left running at `129.212.180.71`.

## Cross-refs

- Phase 1 (lm_head + dispatch fix): `docs/investigations/2026-05-19-mtp-mi300x-phase1/README.md`
- Phase 1.5 (coherence gate): `docs/investigations/2026-05-19-mtp-mi300x-phase1-5-coherence/README.md`
- Master plan: `docs/plans/mtp-dflash-composition-master-plan.md`
- Patched files:
  - `crates/hipfire-arch-qwen35/src/mtp_compose.rs` — `MtpChainMode` enum + `spec_step_dflash_mtp_with_mode`
  - `crates/hipfire-runtime/examples/dflash_mtp_demo.rs` — `--mtp-mode` flag
- Linear-chain semantics: `mtp_head_forward_block_only(..., next_token_embed=Some(prev_row))` at `crates/hipfire-arch-qwen35/src/mtp_head.rs:943`
