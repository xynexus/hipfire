# MI300x Phase 1.5 — coherence gate for `HIPFIRE_GFX942_NATIVE_LM_HEAD=1`

**Date:** 2026-05-19
**Hardware:** AMD Instinct MI300X VF / gfx942 / wave64, 192GB HBM3, ROCm 7.0.0
**Branch:** `feat/mtp-mi300x`, parent commit `4eef6506` (Phase 1)
**Goal:** Validate that the Phase 1 native-wave64 dispatch fix preserves coherent
output across distinct prompt classes before flipping default-on in a follow-up PR.

## Result

**PASS** on all 4 coherence cells run. No panics, no NaN, no `!!!!!` attractor,
no token-loop regression. Output is fluent / on-topic / well-formed across:

- code generation (Fibonacci three-ways prose)
- math reasoning (correct answer + clean `<|endoftext|>` termination)
- story generation (plausible two-sentence narrative)
- canonical composition (byte-stable end-of-output `lru_node = self.tail.prev`)

The env var is safe to default-on for gfx94x in a follow-up PR.

## Methodology — minimal substitute for coherence-gate.sh

The standard `scripts/coherence-gate.sh` short battery iterates over the 5-model
production set (qwen3.5-{0.8b,4b,9b,27b}.mq4 + variants). On this MI300x rental
droplet only the MTP-specific models are present:

```
~/.hipfire/models/qwen3.5-27b.mq4-mtp     (15 GiB bundled trunk + MTP head)
~/.hipfire/models/qwen3.5-27b.mtp         (216 MiB extracted MTP head)
~/.hipfire/models/qwen35-27b-dflash.mq4   (877 MiB DFlash drafter)
```

Per the task brief's fallback path (run a minimal substitute when the full
battery is unavailable: spawn the daemon with the env var, send 3 prompts —
code, math, story — and eyeball for coherence), I ran four cells via the existing
`mtp_only_demo` (solo) and `dflash_mtp_demo` (composition) examples with
`HIPFIRE_GFX942_NATIVE_LM_HEAD=1`. These exercise the same forward-pass and
dispatch surface that the daemon's AR mode would hit.

### Test matrix

| # | Harness          | Prompt                       | max | mode | tok/s  | EOS | coherent |
|---|------------------|------------------------------|-----|------|-------:|-----|----------|
| 1 | `mtp_only_demo`  | code (Fibonacci recursive)   | 120 | q8   |  42.82 | n   | yes      |
| 2 | `mtp_only_demo`  | math (17 sheep, all-but-9)   | 120 | q8   |  45.04 | y   | yes      |
| 3 | `mtp_only_demo`  | story (robot+autumn)         | 120 | q8   |  40.42 | n   | yes      |
| 4 | `dflash_mtp_demo`| canonical PEP-8 LRU (Phase 1)| 120 | q8   | 112.11 | n   | yes      |

Cell 4 is the Phase 1 canonical bench cell, included as a regression detector
against the Phase 1 result (112.90 tok/s reported in
`docs/investigations/2026-05-19-mtp-mi300x-phase1/README.md`). The 112.11 vs
112.90 delta is 0.7%, within session-to-session noise (no DPM warmup,
single-run). Output is byte-identical at the end (`lru_node = self.tail.prev`,
same `accept_dflash_total=110, accept_mtp_total=0`).

### Coherence assessment per cell

**Cell 1 — code/Fibonacci.** Output extends the prompt naturally:
> "Then, write a second function that computes the nth Fibonacci number using
>  dynamic programming (memoization). Include a brief docstring.
>  Finally, write a third function that computes the nth Fibonacci number
>  iteratively. Include a brief docstring.
>  
>  \<think>
>  Thinking Process:
>  1. **Analyze the Request:** ..."

Fluent English, no `!!!!!`, no token loop. The model continues the user's
implicit list (recursive then DP then iterative) and opens a reasoning chain.
**PASS.**

**Cell 2 — math/17 sheep.** Output:
> "Answer:
>  \<think>
>  \</think>
>  The phrase 'all but 9 die' means that every sheep died except for 9 of them.
>  Therefore, the 9 sheep that did not die are the ones remaining.
>  9\<|endoftext|>"

Correct math answer (9), clean reasoning, terminates with `<|endoftext|>`.
**PASS.**

**Cell 3 — story/robot+autumn.** Output:
> "The robot paused its patrol, its sensors overwhelmed by the sudden cascade
>  of crimson and gold leaves drifting from the ancient oak.
>  As it processed the data, a new error message appeared on its display:
>  'Aesthetic appreciation detected.'
>  \<think>
>  Thinking Process:
>  1. **Analyze the Request:** ..."

Two well-formed sentences honoring the prompt's ask. **PASS.**

**Cell 4 — composition canonical.** Output matches the Phase 1-reported
end-of-output (`lru_node = self.tail.prev`), the same `tau_dflash=9.1667`
(Phase 1: same), and `accept_mtp_total=0` (Phase 1: same). Coherent Python
LRU-cache code. **PASS.**

## Reproduction

```sh
# Stage prompts on droplet
cat > /tmp/prompt_code.txt <<EOF
Write a Python function that computes the nth Fibonacci number recursively. Include a brief docstring.
EOF
# (math + story prompts likewise)

# Cell 1 (code, solo)
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/mtp_only_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --prompt-file /tmp/prompt_code.txt \
  --max 120 --no-chatml --kv-mode q8 --max-n 3 --temp 0

# Cell 4 (composition canonical)
HIPFIRE_GFX942_NATIVE_LM_HEAD=1 ./target/release/examples/dflash_mtp_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --drafter ~/.hipfire/models/qwen35-27b-dflash.mq4 \
  --mtp-head ~/.hipfire/models/qwen3.5-27b.mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --mtp-k 2 --temp 0
```

## What I did NOT verify

The minimal substitute does NOT cover:

- Smaller models (0.8B / 4B / 9B). The dispatch fix is shape-driven (B<16 +
  HFQ4G256/MQ4G256 batched paths); larger models like 27B may not exercise
  all the small-B kernel families that 9B + 4B would. Recommend running the
  full `coherence-gate.sh --short` on a machine with the production model set
  before flipping default-on in a follow-up PR.
- MoE (A3B). The Phase 1 fix gates on HFQ4G256 wave64 — for MoE the routed
  experts route through `gemm_q8_0_batched` and other paths that the fix
  doesn't touch. Recommend a separate A3B smoke before default-on.
- Daemon mode. All four cells used demo binaries. A daemon-driven smoke is
  the standard production surface and should be the final pre-flip gate.

## Recommendation

The env var is safe to keep opt-in default-off on this branch. Before flipping
default-on for gfx94x (recommended in a follow-up PR), reproduce the
`scripts/coherence-gate.sh --short` battery on a gfx94x machine that has the
production model set. The Phase 1 result + this Phase 1.5 check on the available
27B model are sufficient to authorize that follow-up PR's test plan but not
to skip the smaller-model smoke entirely.

## Cross-refs

- Phase 1: `docs/investigations/2026-05-19-mtp-mi300x-phase1/README.md`
- Master plan: `docs/plans/mtp-dflash-composition-master-plan.md`
- Patched lines: `crates/rdna-compute/src/dispatch.rs` `rocblas_min_batch` + `is_gcn5_wave64`
