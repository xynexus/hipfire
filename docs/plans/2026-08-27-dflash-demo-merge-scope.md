# Scope: merge dflash_spec_demo's speculative wins into serving-core

Date: 2026-08-27. Companion to
`2026-08-27-dflash-blocksize-gfx1103.md` (the measurement this scopes:
demo 9.29 tok/s τ=5.9 vs daemon 1.31 tok/s τ=2.24, same target, same
drafter, same host).

Both paths call the same `spec_step_dflash`
(`hipfire-arch-qwen35/src/speculative.rs:7297`). The gap is setup and
per-cycle arguments, not the kernel. Of `dflash_spec_demo.rs`'s ~3300
lines, only ~400 are load-bearing for the win; the rest is bench
scaffolding and stays put.

## ① Bug fix, not feature: use the drafter's declared extraction layers (~6 lines)

The demo builds its hidden-state ring from the layer ids the drafter
declares (`dflash_spec_demo.rs:1365-1385`,
`HiddenStateRingBuffer::new_for_layers`). The qwen35 serving loader
re-derives a spread instead (`serving-core/src/load.rs:4556-4564` →
`dflash_extract_layer_ids`, `speculative.rs:5485`), and never reads
`draft_config.target_layer_ids`. The derived and declared sets agree
only for the shipped 40-layer drafter (`speculative.rs:5507-5510`); on a
64-layer Qwen3.8-27B they diverge, so the drafter is fed hidden states
from layers it never saw in training. This is the prime suspect for
τ 5.9 → 2.24. The LFM2 and dspark loaders already use declared ids
(`load.rs:4376-4379`, `load.rs:4462`) — only the qwen35 DFlash loader
re-derives.

Fix: branch to `new_for_layers(...)` when
`!draft_config.target_layer_ids.is_empty()`; `new` is already a wrapper
over it (`speculative.rs:5783-5800`). Leave an assert that
`hidden_rb.extract_layers == draft_config.target_layer_ids` next to
`load.rs:4715`.

Land this first, then re-take the daemon sweep with
`HIPFIRE_SPEC_PHASES=1`. If τ moves to ~5.9, the block-flatness result
in `2026-08-24-qwen38-27b-specdecode-blockers.md:203-215` was measured
against a mis-fed drafter and needs re-taking too — do that before
touching the verify kernel it blames.

## ② One-line sizing fix: verify_scratch hidden_k

Demo passes `target.weights.output.k` (`dflash_spec_demo.rs:1452`);
serving passes `target_config.dim` (`load.rs:4621`). A compact Opus head
with padded `output.k > dim` trips the undersized assert
(`speculative.rs:517-521`). Pass `output.k` in as a parameter at the
`load.rs:2857-2880` call site.

## ③ Adaptive block size — reuse the dspark controller, don't port the demo's

The measurement says B matters (7.46–9.29 tok/s over B∈[5,14]; best
B=10 > the drafter's trained 8). Serving has no controller: the trained
`block_size` is used forever (`load.rs:4625` → `model.rs:123` →
`generate.rs:973`), overridable only by the process-global
`HIPFIRE_DFLASH_BLOCK` (`generate.rs:3393-3398` — whose own comment
admits no serving caller ever passed the override).

Do NOT port the demo's 246-line controller
(`dflash_spec_demo.rs:2411-2472`).
`hipfire-specdecode-dspark/src/dspark_block_controller.rs` is a pure,
no-GPU, 396-line cost-model controller already doing this job better
(argmax τ(N)/wall-time, warmup/ramp/survival). Lift it to
`hipfire-specdecode` (or make it `pub`) and wire it in `generate.rs`:

- `generate.rs:973`: `controller.block()` in place of
  `dflash_block_override()`; keep `HIPFIRE_DFLASH_BLOCK` as a hard pin.
- Feed it next to `spec_metrics.record_window(...)`
  (`generate.rs:997`) with elapsed-per-committed-token
  (`dflash_spec_demo.rs:2736-2744`); one `Instant::now()` at the loop
  top (`generate.rs:896`).
- Clamp `max_block` to `df.block_size` (demo rule,
  `dflash_spec_demo.rs:1282-1289`) so no scratch resizing is needed.

The `dflash_adaptive_b` knob is already plumbed end-to-end
(`hipfire-config/src/lib.rs:345`, `hipfire-model/src/lib.rs:2726`) and
then dropped on the floor at `daemon/handlers/lifecycle.rs:280`
(`let _adaptive_b`) — wire it through to the controller.

## ④ Only if B above trained size is wanted: size scratches for B_MAX

Four sites in `load_dflash_state_source` size from
`draft_config.block_size`: `load.rs:4546, 4561-4562, 4608, 4613-4618`.
The demo pre-sizes to `max(block_size, adaptive_b_max)`
(`dflash_spec_demo.rs:1293-1297`). Skip this if ③'s clamp is taken —
but note the gfx1103 optimum (B=10) is above dflash2's trained B=8, so
the clamp costs ~10% of the measured win. Decide after ① is re-measured:
the optimum may move.

⚠️ Until ④ lands, `HIPFIRE_DFLASH_BLOCK` above the trained B overruns
`DflashScratch`/`GdnTape`/`VerifyScratch` in serving. Don't sweep the
daemon past the trained size.

## ⑤ Gates to revisit after ① is measured

- `HIPFIRE_DFLASH_ALLOW_OPUS` (`load.rs:1080`): the comment at
  `load.rs:1048-1079` says correctness was fixed in `b7b7a9ae5` and the
  gate stands purely on the measured perf number (2.01 tok/s τ=1.778 —
  the mis-fed regime). If ① recovers τ, flip the default in the
  `load.rs:1086` match arm and un-park the halo artifacts.
- KVarN + spec decode: serving silently runs plain AR under kvarn
  unless `HIPFIRE_KVARN_BATCHED_PREFILL=1` (`generate.rs:3866-3877`);
  the demo measured the drafter engaging under kvarn either way
  (`dflash_spec_demo.rs:1254-1259`). Re-measure, consider defaulting on.
- `HIPFIRE_DN_STATE_FP16`: needs no port — both paths already route
  through `default_state_quant`
  (`hipfire-arch-qwen35/src/qwen35/state.rs:181-185`), and serving
  additionally accepts per-load `state_quant:"fp16"` (`load.rs:345`).
  hipfire-env already records +5.5% for it on this exact workload.

## Non-goals

The demo's PLD/Markov head, n-gram cache, cactus-delta, `--ctx-slice`,
NPU draft, and profiling harness all default off in the demo too — they
are not the delta and do not move (`generate.rs:957-980` already passes
the same defaults). DDTree dispatch is already identical on both paths
(`generate.rs:907-955` vs `dflash_spec_demo.rs:2535-2605`).

## Order of work

1. ① + ② (bug fixes, ~7 lines) → re-measure daemon τ on nix2, B pinned
   to trained 8.
2. If τ recovers: ③ controller lift + `dflash_adaptive_b` wiring;
   re-run the block sweep through the daemon.
3. ⑤ gate flips guarded by those measurements; un-park
   `Qwen3.8-27B--dflash2.oq4+` on halo and /srv.
4. ④ only if the daemon's post-① optimum still sits above the trained B.
