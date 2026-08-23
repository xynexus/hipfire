# PFlash works — and cannot pay here until MoE batched prefill lands

2026-08-23, halo/gfx1151, target Qwen3.8-27B--oq4.25++, kvarn KV, 9904-token
prompt, `prefill_keep_ratio=0.25`, `prefill_threshold=2000`.

## It is implemented, and the module header lies

`pflash.rs` still opens with *"Phase 1.0 status: scaffolding only.
`maybe_compress_prompt` always returns `Bypass` regardless of mode."* That is
**stale**: the file carries real scoring/selection and returns
`Ok(PflashDecision::Compressed(..))`. Do not trust that header.

It is also NOT configured by env at request time. `HIPFIRE_PREFILL_*` hydrates
a config, but the daemon builds PFlash state at **model-load** time from LOAD
params: `prefill_compression`, `prefill_drafter`, `prefill_threshold`,
`prefill_keep_ratio`, `prefill_profile`. Setting only the env vars produces no
PFlash at all — not even a bypass line.

## Measured

| drafter | active | tokenizer_compat | outcome |
|---|---|---|---|
| qwen3.5-2b--bf16 | 2B | **false** | BYPASS `tokenizer_mismatch` |
| qwen3.5-4b--bf16 | 4B | **false** | BYPASS `tokenizer_mismatch` |
| Qwen3.8-27B dflash sidecar | — | n/a | LOAD FAILED: no tokenizer metadata |
| **Qwen3.6-35B-A3B--oq4.25++** | **3B** | true | **COMPRESSED 9904 -> 2480** |

The A3B is the right SHAPE of drafter — 3B active against the target's 27B — and
compression works exactly as designed: 9904 -> 2480 tokens, and the target then
prefills the short stream at 273.8 tok/s instead of 179.8.

**And it is still a 3.3x net LOSS**, because scoring took `score_ms=182548`:

    PFlash ON   scoring 182.5 s + prefill  9.1 s = 191.6 s
    PFlash OFF                    prefill 55.1 s =  55.1 s

## Why, exactly

The drafter's scoring pass is just its own prefill, and **MoE prefill is not
batched**. Measured standalone on the same prompt:

    Qwen3.6-35B-A3B (3B active)   54.8 tok/s
    Qwen3.8-27B     (27B active)  179.8 tok/s

9904 / 54.8 = 181 s, which is the observed 182.5 s. So scoring is not
inefficient — the drafter is simply 3.3x slower per token than the model it is
supposed to accelerate, despite having 1/9th the active parameters.

The cause is the same gate found during the KVarN coherence battery: a model
with `DeltaNetMoe`/`FullAttnMoe` layers fails the `all(DeltaNet|FullAttn)` arm of
the batched-prefill gate by construction, so every MoE model prefills per-token.

## What this means

**PFlash is not the blocked thing; MoE batched prefill is.** The mechanism is
sound and the only tokenizer-compatible drafter on this box is exactly the
right shape for it.

Rough sizing of the payoff once MoE batches: if the A3B reached even 3x the
dense rate (a modest ask for 1/9th the active params), scoring 9904 tokens would
be ~18 s against a 9 s compressed prefill — ~27 s total against 55 s, i.e. **~2x**
on long prompts, growing with prompt length.

Two smaller blockers worth recording:

- **No small tokenizer-compatible drafter exists here.** qwen3.5-2b and -4b are
  both `tokenizer_compat=false` against a Qwen3.8 target.
- **DFlash sidecars cannot serve as PFlash drafters**: they carry no tokenizer
  metadata (`tokenizer metadata field missing or wrong type`). A PFlash drafter
  must be a standalone model, which matches the design — it tokenizes the source
  prompt itself.

## Not enabled

Left off. Enabling it today would make every long prompt 3.3x slower.
