# DFlash overlap seam — seed-oracle hit-rate (Phase 2 go/no-go), 2026-07-20

The cheap go/no-go for the DFlash NPU-draft ‖ GPU-verify overlap
(`docs/plans/2026-07-20-dflash-overlap-seam-scope.md`, Phase 2). Grain B drafts
block N+1 during verify N seeded by a **predicted seed = this cycle's
`bonus_token`**. The prediction accuracy **bounds** the achievable overlap: every
mispredict forces a serial fallback. This measures that accuracy on the already-
instrumented seed oracle **before** any concurrency (phases 3–4) is built.

**Verdict up front: NO-GO. Stop at the serial NPU path (phase 1). Do not build
phases 3–4 as scoped.** The exact Grain-B seed predictor (TAIL_MATCH) is correct
in **2.0 %** of GPU-drafter cycles and **1.0 %** of NPU-drafter cycles. At that
rate the blended overlap throughput is **+0.9 % over the serial path that already
exists**. Even the unreachable single-position ceiling (ANYPOS, 23 %) yields only
+12 %. The failure is **structural**, not a drafter-quality deficit — see
Interpretation.

## Setup

- Machine nix1 (gfx1103 + npu1/aie2), branch `chaingun`, GPU lock held durably
  via `hipfire lock run <label> -- …` (survives the driver; verified free on
  exit).
- Target: `~/.hipfire/models/qwen3.5-9b-mq4.hfq` (the rebuilt 9B mq4; losslessness
  anchor `a099a2729d04…`, per the phase-0 brief — the original `02e621bd56b5`
  artifact is gone).
- Drafter: `~/.hipfire/drafts/Qwen3.5-9B.dflash.npu.oq4.25+.hfq` (Phase F ship
  format; closest GPU analog to the NPU body's W4A8).
- Harness: `crates/hipfire-runtime/examples/dflash_spec_demo.rs`. The oracle
  stats are **already surfaced** — `read_seed_oracle_stats()` drains into the
  `seed-oracle: …` stderr line at `dflash_spec_demo.rs:2695`. **No code change
  was needed for this measurement.**
- Config: `--no-adaptive-b --block-size 16 --temp 0.0 --seed 1234 --max 256`,
  separate process per prompt (no `--prompts-file` state leak), Phase F 8-prompt
  corpus.
- NPU path: `HIPFIRE_DFLASH_NPU_DRAFT=1` (phase 1, `0fa4a2972`), serial no-cache
  body (~185 ms/block); run on a 3-prompt τ-spanning subset.

TAIL_MATCH = `drafted[b-1] == bonus_token` (the exact Grain-B predictor:
draft's last-position argmax vs the target's bonus that becomes next cycle's
seed). ANYPOS_MATCH = `bonus_token ∈ drafted[1..b]` (upper bound of any single-
position-from-draft proxy). REJ_MATCH = `drafted[accept_len+1]` (0 by
construction — documented dead-end). Recorded at `speculative.rs:7493`.

## Results — GPU drafter (baseline), full corpus

| prompt | cycles | τ | full_accept | TAIL | ANYPOS |
|---|---|---|---|---|---|
| coherence_capital_france | 28 | 6.179 | 4 | 0.000 | 0.286 |
| coherence_sheep_reason | 46 | 4.696 | 1 | 0.000 | 0.130 |
| coherence_square_function | 32 | 3.375 | 0 | 0.000 | 0.281 |
| humaneval_0_has_close_elements | 41 | 5.415 | 4 | 0.000 | 0.195 |
| lru_cache_pep8_strict | 51 | 4.000 | 3 | 0.078 | 0.235 |
| merge_sort_thinking_off | 13 | 11.308 | 9 | 0.000 | 0.538 |
| trains-meet | 35 | 6.457 | 7 | 0.057 | 0.343 |
| coherence_lloyd_long | 52 | 3.923 | 3 | 0.000 | 0.115 |
| **overall (298 cycles)** | 298 | **5.03** | 31 | **0.020** | **0.228** |

TAIL is **0.000 on 6 of 8 prompts**; nonzero only on lru_cache (0.078) and
trains-meet (0.057). 6 tail hits / 298 cycles = **2.0 %**.

## Results — NPU drafter (the rate that bounds NPU overlap), τ-spanning subset

| prompt | cycles | τ | TAIL | ANYPOS |
|---|---|---|---|---|
| coherence_square_function | 40 | 2.500 | 0.000 | 0.275 |
| trains-meet | 48 | 4.417 | 0.021 | 0.229 |
| merge_sort_thinking_off | 15 | 9.600 | 0.000 | 0.133 |
| **subset (103 cycles)** | 103 | — | **0.010** | **0.233** |

The NPU drafter (lower-quality W4A8, so lower τ than the GPU oq4.25+ drafter)
does **not** have a higher seed-prediction rate — TAIL ≈ 1 %, ANYPOS ≈ 23 %,
statistically indistinguishable from the GPU baseline. This confirms the low rate
is **structural, not a function of drafter quality**.

One transient NPU dispatch timeout was seen on the first square_function NPU run
(`r14 dispatch: Ioctl(TimedOut)` at `dflash_body.rs:242`, empty output); a retry
succeeded and matched the GPU-drafter digest. This is a serial-NPU-path
reliability note, orthogonal to the oracle measurement.

## Losslessness — holds

Canonical gate (`"Explain how a four-stroke engine works."`, `--max 96`), 3
repeats each: **AR == GPU-drafter == NPU-drafter == `a099a2729d04d7dd2362d1676f868c6c`**,
9/9 identical. On the max-256 corpus the spec paths agree bit-identically with
each other (e.g. square_function GPU == NPU = `b33abd1f…`); the observed AR-vs-spec
md5 differences at max-256 are **EOS-boundary length artifacts** (a path stops one
token earlier/later around EOS), not correctness — prompts that run to the max
cutoff (trains-meet) match AR exactly. The drafter-independence invariant holds.

## Interpretation — why TAIL is structurally ~2 %, and the throughput verdict

**The bonus_token is by construction not predictable from the draft's own
outputs.** In every cycle the bonus is one of:

- **(a) partial reject** (≈70 % of cycles, per the accept-len distribution): the
  bonus is the target's **correction at the rejection boundary** — the exact token
  the draft got *wrong* (that is *why* the cycle rejected there). REJ_MATCH is 0 by
  construction; TAIL (a later position on the draft's already-diverged trajectory)
  is essentially random against it.
- **(b) full accept** (≈10 %, the second mode at accept_len=15): the bonus is the
  target's argmax **one position beyond** the draft's last output — a position the
  draft never computed. `drafted[b-1]` is a different position entirely.

So neither regime puts the bonus at a predictable position in the draft. The
task's hypothesis "high-τ prompts have high TAIL" is **refuted**: merge_sort, the
highest-τ prompt (11.3 GPU / 9.6 NPU, 9/13 full-accept), has **TAIL = 0.000** —
high τ means *more* full-accept cycles, i.e. more case (b), i.e. *worse* tail
prediction. The payoff is not concentrated where τ is good; it is structurally
absent everywhere. ANYPOS weakly tracks τ (merge_sort 0.538) but ANYPOS is not a
usable predictor — it only says the bonus appeared *somewhere* in the draft, not
at a position you could pick in advance.

**Blended throughput** at the measured hit-rate X (blended step
`= X·max(draft,verify) + (1−X)·(draft+verify)`; NPU draft 111.9 ms cached-warm,
verify wall ~100 ms, ~7.0 committed tok/cycle at corpus τ≈5):

| overlap fraction X | blended step | tok/s | vs serial | vs AR |
|---|---|---|---|---|
| serial, X=0 (**phase 1, already built**) | 211.9 ms | **33.2** | — | 1.90× |
| **TAIL = 2 % (the real predictor)** | 209.9 ms | **33.5** | **+0.9 %** | 1.92× |
| ANYPOS = 23 % (unreachable ceiling) | 189.1 ms | 37.2 | +12 % | 2.13× |
| perfect, X=1 (**the phase's promise**) | 111.9 ms | 62.8 | +89 % | 3.60× |
| GPU-only AR | — | 17.46 | — | — |

**The overlap machinery of phases 3–4 (draft-side snapshot/rewind, NPU worker
thread, cross-device race analysis) buys +0.9 % over the serial NPU path at the
measured seed accuracy.** The serial path (phase 1) already captures ~1.9× over
GPU-only AR — essentially all the available value — *without* any of that
machinery. Even the theoretical best single-position predictor (ANYPOS 23 %, which
no real predictor can reach, since you cannot know *which* drafted position will
turn out to be the bonus) yields +12 %, still nowhere near the +89 % the
`max(draft, verify)` model assumed.

## Recommendation

- **NO-GO on phases 3–4 as scoped.** Predicted-seed = bonus_token is the wrong
  speculation axis: the quantity being predicted is definitionally the token the
  draft could not produce. Do not build the draft-side rewind, the NPU worker
  thread, or the disjointness/fence analysis on this basis.
- **Ship / keep the serial NPU path (phase 1).** It already beats GPU-only AR by
  ~1.9× at τ≈5 with zero concurrency and structural losslessness.
- **The only thing that could rescue an overlap** is a seed predictor *not* built
  from the draft's own tokens (a cheap dedicated bonus-predictor, or hedging the
  top-K seed candidates). Hedging is a non-starter on Phoenix — the NPU draft is
  already at parity with the verify wall, so drafting K seeds ×K's the NPU cost and
  blows the `max(draft,verify)` budget. A dedicated bonus head is a redesign well
  beyond phases 3–4's scope and speculative. Either way, the scoped overlap is dead
  and the decision is made here, cheaply, as the plan intended.

## Reproduce

```
T=~/.hipfire/models/qwen3.5-9b-mq4.hfq
D=~/.hipfire/drafts/Qwen3.5-9B.dflash.npu.oq4.25+.hfq
# GPU baseline, per prompt (stderr carries the seed-oracle: line):
hipfire lock run oracle -- examples/dflash_spec_demo --target $T --draft $D \
  --prompt "$(cat benchmarks/prompts/<p>.txt)" \
  --no-adaptive-b --block-size 16 --temp 0.0 --seed 1234 --max 256
# NPU drafter: prefix HIPFIRE_DFLASH_NPU_DRAFT=1 (needs /tmp/dflash_w,
# /tmp/dflash_manifest_flash.json, ~/.hipfire/npu/r14_1x2x128_nb128).
```
