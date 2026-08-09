# HANDOVER: DeltaNet state precision (2026-08-09)

Written at the end of a long session on `feat/daemon-state-hoist`. Everything
here is actionable without asking. Tree is GREEN; the only modified files are
the user's own (`CLAUDE.md`, two `docs/`, two `scripts/`+`benchmarks/` shell
files) — leave them alone.

## Where things stand

`origin/master` is merged (0 behind). Four `oq4.25++` benchmarks are done and
recorded in `2026-08-08-quant-benchmark-queue-handoff.md`. A random-access
`.hfa` reader landed (`hipfire-quantize/src/hfa.rs`, `271d3e140`).

DeltaNet state is the live thread:

- **Q8/Q4 are gone from `StateQuant`**, which is now `{ FP32, FP16 }`. Verified
  `cargo clippy --workspace --all-targets` = 0 errors.
- **FP32 is the default** and is BIT-IDENTICAL to its pre-removal baseline
  (Qwen3.5-35B-A3B, 16/16 chunks, `mean_kld 0.038913`, `max|diff| = 0`).
- **FP16 is opt-in via `HIPFIRE_DN_STATE_FP16` and currently UNSAFE** — see
  below. The env var's own description says so.
- Four f16/f32 kernels exist and are hardware-validated on gfx1151:
  `gated_delta_net_f16` (state), `_f32_tree`, `_f16_tree`,
  `_f16_routed_batch_seq`. f32 tree is byte-exact vs the linear f32 kernel;
  f16 tree is byte-exact vs the f16 linear kernel; routed f16 is byte-exact
  vs per-session replay with interleaved rows.

## THE ONE TASK: finish the Q8 removal

### Why FP16 is unsafe right now

The Q8 **dispatch functions** in `hipfire-rdna/src/dispatch/gated.rs` were never
deleted — only the `StateQuant` variants and the call sites that SELECTED them.
Renaming `gated_delta_net_q8.hip` to `-disabled.hip` made them look retired
without making them unreachable. Under FP32 the surviving callers are harmless;
under FP16 the state is half-size with a vestigial `s_scales`, so:

    Memory Fault Error ... kernel: gated_delta_net_q8

reproduced by teacher-forced scoring of a dense Qwen3.5-0.8B. That is why the
FP16 default was reverted in `c2412857a` the same day it was set.

### The 8 call sites

Deleting the dispatch fns makes the compiler enumerate them in one build. Grep
CANNOT find them: none mentions `StateQuant`, they call the kernel directly
inside `else` arms.

| site | arm |
|---|---|
| `prefill_chunk.rs:2972` / `:6479` | `gated_delta_net_q8_tree_batch_seq` |
| `prefill_chunk.rs:3013` / `:6507` | `gated_delta_net_q8` (per-token) |
| `prefill_chunk.rs:3030` / `:6524` | `gated_delta_net_q8_batch_seq` |
| `prefill_batch.rs:664` | `gated_delta_net_q8_routed_batch_seq` |
| `speculative.rs:3374` | surfaced by the compiler, unexamined |

### Step 1 — scratch buffer (known-good, just re-apply)

`prefill_batch.rs`. Replace the Q8 tape pair with ONE precision-agnostic tape:

- `:127-128` — `dn_s_tape_q8` + `dn_s_tape_scales` become
  `pub dn_s_tape: Option<GpuTensor>`
- `:4288-4304` — one allocation:
  `max_batch * linear_num_value_heads * linear_value_head_dim^2 * 4` bytes,
  `DType::Raw`
- `:4353-4354` — `free_gpu` drops the second entry

**4 bytes/element is deliberate**: f32 uses the buffer exactly, f16 uses the
first half. Sizing for the wider case keeps the allocation independent of
`dn_state.quant`, which is per-session and not known at scratch-alloc time.
Neither precision has scales, so the scales tape is deleted outright.
23.1 MB at max_batch=22, n_v_heads=16, head_dim=128 (was 5.77 MB + 180 KB).

### Step 2 — the two tree arms, BY HAND

`prefill_chunk.rs:2961-2987` and `:6464-6494`. **Do not regex these.** A
non-greedy `.*?` over the block runs past the intended `)?;` and eats the
following else-arm; it surfaced as a bogus `no field ffn on
DeltaNetLayerWeights` at `:3149`. They are ~25 lines each; edit them directly.

For each:
1. DELETE the `if matches!(dn_state.quant, StateQuant::FP32) { return Err(...) }`
   guard. That error ("FP32-state batched prefill does not support tree DeltaNet
   replay yet") is exactly what stranded DDTree under the never-Q8 policy.
2. Take `pbs.dn_s_tape` (single tape, no scales).
3. `match dn_state.quant` onto `gated_delta_net_f32_tree_batch_seq` /
   `gated_delta_net_f16_tree_batch_seq`.

Signature vs the Q8 kernel: drop `s_scales_init` and `s_tape_scales`; everything
else carries over in the same order.

### Step 3 — the other six sites

Mechanical. Per-token and `_batch_seq` arms become FP32/FP16 arms calling
`gated_delta_net_{f32,f16}_batch_seq`; `prefill_batch.rs:664` becomes the
routed f32/f16 pair. NOTE `speculative.rs` deliberately loops PER-TOKEN for
non-FP32 state: the batched kernel narrows once after N tokens, which breaks
rollback parity when replaying an accepted prefix. Keep that shape for FP16.

### Step 4 — delete the dispatch fns and constants

`dispatch/gated.rs`: `gated_delta_net_q8`, `_batch_seq`, `_routed_batch_seq`,
`_tree_batch_seq`, `gated_delta_net_q4`. `overlays/gfx1151.rs`:
`gated_delta_net_q8_reg_gfx1151` (note: `pub(crate) fn`, not `pub fn`).
`kernels.rs`: the five `GATED_DELTA_NET_Q8_*`/`Q4` constants. `dispatch/mod.rs`:
the `gated_delta_net_q8_tree` warm-up precompile entry. Then delete the
`kernels/src/gated_delta_net_q8*.hip` sources.

### Step 5 — validate

The instrument that found the bug, and the one to trust:

```sh
# reference: SAME weights, FP32 state
HIPFIRE_DN_STATE_FP32=1 hipfire-daemon <<< '{load}{kld_eval build_ref}{unload}'
# candidate: SAME weights, FP16 state
HIPFIRE_DN_STATE_FP16=1 hipfire-daemon <<< '{load}{kld_eval score}{unload}'
```

Only the state precision differs, so any KLD is precision alone — no sampling,
no task confounders. Scripts: `$S/tf-state.sh` (0.8B) and `$S/pbs-test.jsonl`
(35B) if the scratchpad survives.

## Evidence for FP16, and its limits

- 35B oq4.25++: FP16 `0.039176` vs FP32 `0.038913`, **+0.68%** — ~40x smaller
  than the CI half-width. Max per-chunk `|diff|` 1.084e-02 (~28% of the mean),
  so the per-chunk spread is wide and +0.68% is a mean over it, not a bound.
- 2B greedy decode: tracked FP32 byte-exactly for **720 tokens**; degeneration
  metrics unchanged (`unique_ratio` 0.3325 vs 0.3317, `max_freq` 0.0450 vs
  0.0458) against Q8's recorded signature of 0.625→0.555 and 0.055→0.078.
- 0.8B, 6 prompts: **inconclusive, and it cannot be conclusive.** FP32's own
  `unique_ratio` ranges 0.11–0.36 across prompts — that within-arm spread dwarfs
  the FP16-vs-FP32 delta. Greedy decode on a small model sometimes degenerates
  regardless of precision, and which arm falls in is arbitrary. On prompt 4 FP32
  collapsed (0.1100) and FP16 did not.

So: no evidence of harm, insufficient power to claim safety. Teacher-forced
comparison is the instrument that can settle it; generation-based tests cannot,
because greedy chaos is the dominant term.

## Traps (each cost real time today)

- **Scripted bulk edits misfire in this codebase, three times today.**
  `StateQuant::Q8 -> FP32` matched `Mamba2StateQuant::Q8` as a SUBSTRING and
  rewrote nemotron (a separate Mamba-2 enum). An inserted `_ => FP32` arm
  SHADOWED a later `Ok(other) => panic!`, turning a loud error into a silent
  default. A `.*?` block match ate the following else-arm. Prefer hand edits for
  match arms; if scripting, verify with `--workspace --all-targets`.
- **`cargo build --bin X` does NOT compile tests or examples.** Reporting a
  refactor complete off a binary build missed 18 files twice. Use
  `cargo clippy --workspace --all-targets`.
- **Do not run a ~25-minute job in a foreground Bash call.** The 10-minute tool
  timeout SIGTERMs it at exit 143 and it looks like a hang. Use
  `nohup ... &` plus a polling waiter.
- `pkill -f <pattern>` kills your own shell (bare `exit 144`). `pkill -x` cannot
  match names >15 chars (`hipfire-coexistence`). Kill by PID from `pgrep -f`,
  excluding `$$`.
- `git stash` is unusable here (`.agents/` symlink farm).
- The tiny-quant gate has **8 pre-existing failures** (qwen2/hfq4, gemma3
  q8f16+hfq4, minimax/mq4, qwen3_5/q8f16, qwen3_5_moe q8f16+mq6+mq4). Not
  regressions.
- `dense_prefill_bf16_is_batchable_in_qwen35` fails and is NOT from this work —
  `is_batchable_la` appears zero times in the DeltaNet diffs. Stale BUG-001
  assertion; left failing deliberately rather than edited green.

## Adjacent thread, not blocking

Batched MoE prefill is still declined for MoE models (`pbs_eligible`, the layer
kind list). Widening it was MEASURED to produce garbage — KLD 10.69 vs 0.0389 —
so it is not a one-line fix. Details in
`2026-08-08-quant-benchmark-queue-handoff.md` §4. That gate is also what makes
the 397B benchmark impractical (§8): per-token prefill against streamed weights
would re-read ~190 GB per token.
