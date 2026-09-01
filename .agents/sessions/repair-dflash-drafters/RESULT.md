# RESULT: DFlash drafter repair — 2026-09-02

All six were repaired. **Two of them can never be driven**, and the brief's
"unblocks five families" premise does not survive measurement: the runtime marks
DFlash `full` for the **qwen3.5 family only**.

## What was produced

`~/.hipfire/models/<Name>--dflash.bf16.hfq`, six of them, built from the `/srv`
artifacts' own metadata — no HF source, `/srv` untouched:

    hipfire inspect $SRC --json  ->  dflash.json   (fields below)
    hfq rearch   $SRC s1.hfq  --arch-id 20
    hfq meta-set s1.hfq  out.hfq --key dflash --value-file dflash.json --json

Every one reports `arch=20` and stores `dflash` as a JSON **object**, and every
field cross-checks against the artifact's own tensor shapes — e.g. 122B
`fc.weight [3072, 18432]` = hidden 3072 x 6 extracts, `q_proj [4096, 3072]` =
32x128, `k_proj [512, 3072]` = 4x128. The containers are right.

## Verified by driving them — `dflash_spec_demo`, τ against a healthy control

| drafter | loads | τ | verdict |
|---|---|---|---|
| **Qwen3.6-27B** (pre-existing, control) | yes | **2.100** | reference for a healthy oq4.25++ target |
| **Qwen3.5-9B** | yes | **3.000** | WORKS — above control |
| **Qwen3.6-35B-A3B** | yes | **2.000** | WORKS — at control |
| Qwen3.5-122B-A10B | yes | **0.333** | loads and drafts, but 6x below control |
| Qwen3.5-397B-A17B | — | — | no 397B target on this box; unverifiable |
| gemma-4-26B-A4B-it | **no** | — | arch 24, `dflash = "none"` — no runtime path |
| gemma-4-31B-it | **no** | — | arch 24, `dflash = "none"` — no runtime path |

The control matters: without it, τ=2.0 on the 35B looks poor next to the brief's
quoted 10.6667. Against an `oq4.25++` target on this box, ~2.1 IS the healthy
number, so 9B and 35B-A3B are fine and 122B is a genuine outlier.

## The premise that did not survive

`docs/model-support.toml` sets `dflash = "full"` for exactly one row —
`ids = [5, 6]`, label `qwen3.5`. Every other arch is `"none"`, and
`load.rs::require_arch_feature` refuses on that basis. **The gemma-4 drafters are
correctly repaired containers with nothing that can run them.** Repairing them
was still right — the container work is done whenever gemma-4 DFlash lands — but
it unblocks two families, not five.

## 122B: container proven right, cause is downstream

Its non-layer tensors are identical in shape to the working 9B drafter (`fc`,
`hidden_norm`, `norm`; no `lm_head`/`embed` — drafters reuse the target's). What
is unusual is the TARGET: `Qwen3.5-122B-A10B--oq4.25++` carries
`lm_head.weight qt=36` while its embed is `qt=49` — the separate-lm_head anomaly
already on record as this model's known defect. Drafting reads that head. Not
chased further; recorded as a lead, not a conclusion.

## Fixed along the way

`dflash_spec_demo` had **no arch check**, so a gemma-4 target died as
`tensor not found: layers.0.mlp.gate.weight` from inside the *qwen35* loader —
a message about a MoE router in a family the model is not in, which reads as a
broken drafter. It now consults `arch_features(...).dflash` and refuses by name.
Tested both ways: gemma-4 refuses, qwen3.5 still loads.

## Corrections to the brief

- **`num_target_layers` is NOT informational.** `from_source` requires it (a hard
  `?`) and `validate_target_geometry` checks it against the target's layer count.
  It is present in every artifact's `config`, so no derivation was needed —
  `max(target_layer_ids) + 3` agreed with the stored value in all six, which is
  worth knowing but was not load-bearing.
- **`block_size` is not always at `config` top level.** On Qwen3.5-397B and
  Qwen3.6-35B-A3B it lives under `config.dflash_config`.
- **`rope_theta` is not always at `config` top level.** Those same two carry
  `config.rope_parameters.rope_theta` instead. A reader that only checks
  `config.rope_theta` gets `None` and silently defaults to 1e7.
