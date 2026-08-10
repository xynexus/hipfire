# DONE: DeltaNet Q8 state removal (2026-08-09 → 2026-08-10)

The task this file described — "finish the Q8 removal" — is complete. Kept as
the record of what was removed, what replaced it, and what was deliberately
left behind.

## What was wrong

`StateQuant` lost its `Q8`/`Q4` variants on 2026-08-09, but the Q8 **dispatch
functions** in `hipfire-rdna/src/dispatch/gated.rs` were never deleted — only
the variants and the call sites that SELECTED them. Renaming
`gated_delta_net_q8.hip` to `-disabled.hip` made them look retired without
making them unreachable. Eight surviving `else` arms still called them
unconditionally, so FP16 state (half-size, vestigial `s_scales`) faulted:

    Memory Fault Error ... kernel: gated_delta_net_q8

Grep could not find those arms — none mentions `StateQuant`. Deleting the
dispatch functions first and letting the compiler enumerate the callers is what
made this a bounded job.

## What replaced it

| was | now |
|---|---|
| `gated_delta_net_q8_tree_batch_seq` | `gated_delta_net_{f32,f16}_tree_batch_seq`, picked by `dn_state.quant` |
| `gated_delta_net_q8` / `_batch_seq` | `gated_delta_net_{f32,f16}_batch_seq` |
| `gated_delta_net_q8_routed_batch_seq` | `gated_delta_net_f16_routed_batch_seq` |
| `dn_s_tape_q8` + `dn_s_tape_scales` | one `dn_s_tape`, 4 bytes/element |

The tape is sized at 4 B/element deliberately: f32 uses it exactly, f16 uses the
first half. That keeps the allocation independent of `DeltaNetState::quant`,
which is per-session and not known at scratch-alloc time. 23.1 MB at
max_batch=22, n_v_heads=16, head_dim=128 (was 5.77 MB + 180 KB).

Deleted outright: 5 dispatch fns, the `gated_delta_net_q8_reg_gfx1151` overlay,
6 kernel constants, 7 `.hip` sources, the warm-up precompile entry,
`gdn_requant_seed`, `profile::gated_delta_net_q8_bytes`,
`HIPFIRE_GDN_Q8_REG_GFX1151`, and `examples/test_gated_delta_net_tree.rs`
(`test_gated_delta_net_tree_f32.rs` already covers all four replacements).

## Two defects the compiler surfaced that the call-site table missed

- **`prefill_chunk.rs` MoE branch had NO FP32 arm.** It fell through to the Q8
  kernels regardless of state precision, reading FP32 state as int8 and a
  `[n_heads]` scales vector as `[n_heads × head_dim]`. Unreached today only
  because batched prefill is declined for MoE models (`pbs_eligible`), which is
  why the 35B FP32 benchmarks never hit it. The arm now exists.
- **`prefill_batch.rs` `delta_q8` was `matches!(dn_quant, FP32) && false`** —
  permanently dead. Now `delta_f16`, selecting the f16 routed kernel when the
  state is f16. Not optional: the f32 routed kernel over f16 state reads every
  element at the wrong offset.

Two side fixes: `DDTree` tree replay is no longer blocked (the
"FP32-state batched prefill does not support tree DeltaNet replay yet" guard is
gone, since an FP32 tree kernel now exists), and two `is_bf16_artifact` bindings
in `hipfire-serving-core/src/load.rs`, dead since the FP32 short-circuit was
removed, were deleted — `no-gpu-ci.sh` runs `-D warnings` and was red on arrival
for that reason, which a plain `cargo clippy` only warns about.

## Validation

Teacher-forced KLD, same weights, only the DeltaNet state precision differing —
so any KLD is precision alone, with no sampling and no task confounder:

```sh
./target/release/hipfire-daemon        < ref.jsonl    # build_ref, FP32 state
HIPFIRE_DN_STATE_FP16=1 ./target/release/hipfire-daemon < score.jsonl  # score
```

Corpus `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`,
n_ctx=2048, 8 chunks, 8184 scored tokens per arm.

| model | arm | mean_kld | p99_kld |
|---|---|---|---|
| Qwen3.5-2B bf16 (dense) | FP32 vs FP32 ref | 4.27e-10 | 6.08e-10 |
| Qwen3.5-2B bf16 (dense) | FP16 vs FP32 ref | 5.65e-05 | 6.29e-05 |
| Qwen3.5-35B-A3B oq4.25++ (MoE) | FP16 vs FP32 ref | 2.57e-03 | 6.98e-03 |

The dense 2B run is the same instrument that produced the memory fault; it now
completes clean. The FP32 self-score at ~4e-10 is top-k reference
reconstruction noise, i.e. the FP32 path is unchanged end to end.

The 35B number is ~45x the 2B one, which is the finding worth carrying forward:
FP16 state costs more on the larger MoE model, not less. Do NOT read it against
the "+0.68%" in the pre-removal notes — that measured a different quantity
(quant-vs-bf16 KLD under each state arm, where the quant error dominates and the
state term is a small perturbation of it). This measures the state term
directly. For scale, the model's own oq4.25++ budget is ~0.0389, so FP16 state
adds roughly 6.6% of that budget again. Small, but an order of magnitude larger
than the dense case, and one more reason the default stays FP32 until this is
measured across more models.

`cargo clippy --workspace --all-targets` clean; `./tests/no-gpu-ci.sh` exits 0.

## FP32 is still the default

FP16 stays behind `HIPFIRE_DN_STATE_FP16`. The blocker that forced the
2026-08-09 revert is gone, but the evidence for flipping the default is still
one prompt on one model plus the runs above. Moving it wants a teacher-forced
comparison across more models — the runs above are the instrument for that, not
the verdict.

## Deliberately left behind

None of these is reachable by a kernel any more; each is its own cleanup.

- `DeltaNetState::s_scales` and `s_ef_residual` — allocated, zeroed, snapshotted
  and freed, read by nothing. Threaded through `DeltaNetSnapshot`, the multi-GPU
  split and the route-shape plumbing (`dn_scale_ptrs`, `dn_scale_layers`), so
  removing them is a real diff rather than a deletion.
- `debug_gdn_requant_frame` / `debug_set_gdn_requant_frame` and the
  `q8_gdn_verify_*` env toggles — ~60 sites in `speculative.rs` driving a
  counter that no kernel reads now.
- `deltanet_state_fp32_below` — `#[deprecated]`, returns `usize::MAX`.
- Stale `~/.hipfire/kernels/gfx1151/gated_delta_net_q8*.hsaco` in the local
  kernel cache. Never loaded (nothing names them); safe to delete by hand.

## Traps (kept — each cost real time)

- **Scripted bulk edits misfire in this codebase.** `StateQuant::Q8 -> FP32`
  matched `Mamba2StateQuant::Q8` as a SUBSTRING and rewrote nemotron. An
  inserted `_ => FP32` arm SHADOWED a later `Ok(other) => panic!`. A `.*?` block
  match ate the following else-arm. Hand-edit match arms; verify with
  `--workspace --all-targets`.
- **`cargo build --bin X` does NOT compile tests or examples.** Use
  `cargo clippy --workspace --all-targets`.
- **Do not run a ~25-minute job in a foreground Bash call.** The 10-minute tool
  timeout SIGTERMs it at exit 143 and it looks like a hang. Use `nohup ... &`
  plus a polling waiter.
- `pkill -f <pattern>` kills your own shell (bare `exit 144`). `pkill -x` cannot
  match names >15 chars. Kill by PID from `pgrep -f`, excluding `$$`.
- `git stash` is unusable here (`.agents/` symlink farm).
- The tiny-quant gate has **8 pre-existing failures** (qwen2/hfq4, gemma3
  q8f16+hfq4, minimax/mq4, qwen3_5/q8f16, qwen3_5_moe q8f16+mq6+mq4). Not
  regressions.
- `dense_prefill_bf16_is_batchable_in_qwen35` fails and is NOT from this work —
  `is_batchable_la` appears zero times in the DeltaNet diffs. Stale BUG-001
  assertion; left failing deliberately rather than edited green.

## Adjacent thread, not blocking

Batched MoE prefill is still declined for MoE models (`pbs_eligible`). Widening
it was MEASURED to produce garbage — KLD 10.69 vs 0.0389 — so it is not a
one-line fix. Details in `2026-08-08-quant-benchmark-queue-handoff.md` §4. That
gate is also what makes the 397B benchmark impractical (§8): per-token prefill
against streamed weights would re-read ~190 GB per token.
