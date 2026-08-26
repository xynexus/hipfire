# TODO: run the co-resident training test — M8's last untested half

**Status:** OPEN, and now **unblocked**. PR #363 wired `train_lora` to accept a
`.hfq` base, which was the thing stopping this. Runnable on **nix1**; no halo
booking, no download, no `.pth` conversion.

## What is still untested

M8's exit is *"a LoRA training step and an interactive decode step interleaved at
module granularity in one executor, with the loss curve over 200 steps inside
the run-to-run noise band established by two solo runs."*

Two halo rounds passed it (`docs/experiments/2026-08-26-m8-halo-*.md`), and round
2 added the negative control that made the pass falsifiable. But **both rounds
trained a 3.28M-param fixture**, so the argument that motivated running on halo
at all — that a suspended `LoraTrainSession` holds its base in fp32, ~28 GB for a
7B — was never exercised. The served side supplied all the memory pressure; the
training side supplied essentially none.

That is the half this TODO covers.

## Why it is runnable now

`handlers/train.rs` used to accept only a safetensors directory. Both boxes'
HF caches ship Meta `.pth` only (verified — the blob starts `PK\003\004`), so
no large llama base was reachable. PR #363 dispatches on the extension to
`loader::load_llama_fp32_hfq`, which had existed all along.

Confirmed working: `Llama-3.2-3B-Instruct--bf16.hfq`, 20 single-step quanta,
loss 12.63 → 2.72 monotonic, no errors.

## The run

| role | artifact | resident cost |
|---|---|---|
| trainer base | `Llama-3.2-3B-Instruct--bf16.hfq` (3.99 GB on disk) | **~13 GB** widened to fp32 |
| served model | `Qwen3.6-35B-A3B--oq4.hfq` | **17.79 GB** |

~31 GB of nix1's **42.0 GB** GTT. Both already on local NVMe.

Follow the round-2 protocol
(`docs/experiments/2026-08-26-m8-halo-training-interleave.md`), which already has
the traps written down. Specifically:

1. **Noise band first.** Two solo training runs *before* the interleaved one.
   If they come out bit-identical, the band has zero width and the criterion
   becomes "interleaved is bit-identical too" — check for that first, it is
   stronger and cheaper.
2. **Negative control.** Perturb `lr` by ~2% and show the curve differs. On halo
   this diverged at step 1 (Δ 9.44e-05). Without it, bit-identical curves prove
   nothing about the comparison's sensitivity.
3. **Positive probe on the decode side.** A silently-dead decode stream turns the
   run back into solo training and would "pass" a loss-only test. Count decode
   steps, and use `max_tokens` ≥ 24 with varied prompts — 4 tokens against a
   thinking model never escapes the `<think>` preamble, which is how the halo
   probe fooled itself.
4. **Measure GTT with `mem_info_gtt_used`**, not the daemon ledger and not
   `rocm-smi`. At three co-resident models the ledger read 31.25 GB against
   39.47 GB actual, and `rocm-smi` reports only the 0.2 GB dedicated carveout.
   See `docs/experiments/2026-08-26-load-eviction-semantics-nix1.md`.
5. **Echo back the config the daemon actually used.** `steps` must be nested
   under `train` or the run silently uses the 200 default while reporting
   otherwise — a real bug found on halo, and the class of thing that produces a
   confident wrong answer.

## What would make it fail, and that is fine

This is the first run where the trainer holds real memory. ~31 GB of 42 GB is
comfortable on paper, but the ledger understates by ~8.5 GB at three models, so
the true figure could land near the ceiling. If it OOMs, that is a result: the
failure is clean (survivors bit-identical, no leak — the retained pool memory is
reused, verified over three consecutive failures), and it would be the first
direct evidence for M6's VRAM guard on the *training* path rather than the
serving one.

## Known ceiling

`load_llama_fp32_hfq`'s own note: fp32 widening "does not scale, deliberately".
A 7B fp32 is ~30 GB, which plus any served model exceeds nix1's 42 GB. So **3B
is the ceiling on this box**, and the original §M8 arithmetic (fp32 7B ≈ 28 GB)
still needs halo — where 128 GB makes a 7B trainer plus a served MoE fit
comfortably. Do the 3B run on nix1 first; it is cheap and will surface the
harness problems before anyone books the shared box.

## Also still open, separately

- **Module sharing between workloads.** `handlers/train.rs` trains hipfire-train's
  own un-fused `LlamaModel`, not the served qwen35 adapters, so even a passing
  run demonstrates *one executor, two resident workloads* — not two workloads
  sharing module-granular residency of the same weights. The stronger reading of
  M8 needs that follow-on.
- **Eviction / downgrade / refuse** needs the HTTP server path;
  `plan_model_residency` is never called by the daemon.
