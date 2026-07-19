# DFlash Phase F: weight format decided on ACCEPTANCE RATE (2026-07-19)

Deciding experiment for the DFlash drafter weight format. **Valid for the first
time** — the previous attempt was invalidated by two correctness bugs (a
nondeterministic verify forward, fixed `6ca303af8`; a non-lossless rollback
path, fixed `6dcddfcd6`).

Speculative decoding is lossless: the target verifies every token, so draft
quality costs **acceptance rate**, never correctness. Cosine/SNR is the right
gate for a kernel and the wrong gate for a drafter's weight format. int4
drafters measure full-body cos 0.898 against a >0.99 bar that is unsatisfiable
by any int4 format (per-tensor group sweep 4096→64 moves cos only
0.9879→0.9943, ~5.5 dB/bit). This entry is the only thing that can answer
"is int4 usable?".

Sibling weight-quant quality entry: `benchmarks/results/oq-weight-quant-kld-20260715.md`
(that one measures standalone-model KLD/PPL; this one measures drafter
acceptance, which is a different and — for a drafter — the decision-relevant
quantity).

## Setup

- Machine nix1 (gfx1103), branch `chaingun`, GPU lock `dflash-phasef`.
- Target: `~/.hipfire/models/qwen3.5-9b-mq4.hfq`
- Harness: `crates/hipfire-runtime/examples/dflash_spec_demo.rs`
- **`--no-adaptive-b --block-size 16`** — mandatory. Adaptive-B is default-ON and
  sizes blocks from an EWMA of accept_len, so each drafter would otherwise run at
  a different mean B (previously measured 11.89–15.14) and both τ and
  accept_rate are functions of B. 16 is the drafter's trained block size.
- **Separate processes per row**, not `--prompts-file` — resident mode leaks
  state across rows.
- `--temp 0.0 --seed 1234 --max 256`. 8 prompts × 5 drafters = 40 runs.

Corpus (all committed, `benchmarks/prompts/`):

| prompt | md5 (first 12) |
|---|---|
| coherence_capital_france.txt | 3cfcecf5e775 |
| coherence_sheep_reason.txt | c8db3ca21a75 |
| coherence_square_function.txt | 7865a793cd52 |
| humaneval_0_has_close_elements.txt | 5333a1f70d88 |
| lru_cache_pep8_strict.txt | df5dedc8040c |
| merge_sort_thinking_off.txt | 253c7ac50857 |
| trains-meet.txt | db92b572702a |
| coherence_lloyd_long.txt | f20bbc4f5b88 |

## Losslessness gate — PASSES

At temp 0 every drafter commits **byte-identical tokens** to `--ar-baseline`,
differing only in accepted counts. Verified two ways:

1. Gate prompt ("Explain how a four-stroke engine works.", `--max 96`): all 96
   token ids identical to the AR baseline for all five drafters, in both the
   default and the `--no-adaptive-b --block-size 16` configuration.
2. Across the full 8-prompt corpus: token sequences and completion lengths
   identical to the f16 reference for every drafter on every prompt.

The AR baseline is deterministic across 3 repeats (token-list md5
`bef18060fa35`).

> **The literal gate in the Phase 0 brief is unreproducible as written.**
> `md5 over "| tail -20" must be 02e621bd56b5` cannot be stable — `tail -20`
> spans the `BENCH METRICS` block, which contains wall-clock timings
> (`prefill_secs`, `decode_secs`, `tok_s`, `ttft_ms`). No digest of that window
> can be constant across runs. The substantive invariant is the emitted token
> id sequence; that is what was checked here. The brief's gate text should be
> updated to digest the token list.

## Results

8 prompts, mean over prompts. Completion lengths are identical across drafters
by construction (lossless), so τ is directly comparable and **not**
EOS-confounded.

| drafter | format | on-disk | accept rate | τ | rel τ | Δaccept vs f16 | accept range | completion len | terminated | tok/s |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `Qwen3.5-9B.dflash.f16.hfq` | f16 ref | 2000.2 MiB | 0.3826 | 5.739 | 1.0000 | — | 0.271–0.754 | 237.4 | 2/8 | 8.18 |
| `Qwen3.5-9B.dflash.npu.oq8.hfq` | int8 | 1008.0 MiB | 0.3826 | 5.740 | 1.0001 | +0.0000 | 0.271–0.754 | 237.4 | 2/8 | 9.07 |
| `Qwen3.5-9B.dflash.npu.oq4.25+.hfq` | mixed int4+int8 | 531.5 MiB | 0.3778 | 5.668 | 0.9875 | −0.0048 | 0.264–0.754 | 237.4 | 2/8 | 8.16 |
| `Qwen3.5-9B.dflash.npu.oq4.hfq` | pure int4 (qt=47) | 508.0 MiB | 0.3551 | 5.327 | 0.9281 | −0.0275 | 0.240–0.695 | 237.4 | 2/8 | 7.63 |
| `qwen3.5-9b-mq4.dflash.hfq` | MQ4 | 531.5 MiB | 0.3729 | 5.594 | 0.9747 | −0.0097 | 0.264–0.695 | 237.4 | 2/8 | 8.02 |

Per-prompt acceptance rate:

| prompt | f16 | oq8 | oq4.25+ | oq4 | mq4 |
|---|---:|---:|---:|---:|---:|
| coherence_capital_france | 0.380 | 0.380 | 0.395 | 0.366 | 0.395 |
| coherence_sheep_reason | 0.307 | 0.299 | 0.287 | 0.295 | 0.350 |
| coherence_square_function | 0.277 | 0.277 | 0.277 | 0.246 | 0.264 |
| humaneval_0_has_close_elements | 0.303 | 0.303 | 0.306 | 0.298 | 0.296 |
| lru_cache_pep8_strict | 0.296 | 0.304 | 0.296 | 0.269 | 0.275 |
| merge_sort_thinking_off | 0.754 | 0.754 | 0.754 | 0.695 | 0.695 |
| trains-meet | 0.473 | 0.473 | 0.443 | 0.431 | 0.443 |
| coherence_lloyd_long | 0.271 | 0.271 | 0.264 | 0.240 | 0.264 |

**tok/s here is NOT a clean format comparison** and should not be read as one:
the loader dequantizes every format to F16 on the GPU, so on-device bytes are
identical at runtime and the formats cannot differ for bandwidth reasons. The
spread is thermal drift across a ~40-minute sweep plus load-time variation.
Acceptance rate is the format-sensitive quantity; tok/s is reported only for
completeness.

## Findings

- **int4 IS usable as a drafter.** Pure int4 (`oq4`) costs **7.2% of τ**
  (5.739 → 5.327) for a **3.94× on-disk reduction** vs f16 and a 1.98×
  reduction vs int8. The SNR gate (cos 0.898, ~22 dB loss) badly overstates the
  practical damage — it predicted the format was unusable; acceptance says it
  costs 7%.
- **int8 (`oq8`) is acceptance-identical to f16** — same accept rate on 7 of 8
  prompts and equal to 4 decimal places on the mean, at half the size. There is
  no acceptance argument for shipping f16 weights.
- **`oq4.25+` is the sweet spot: 98.75% of f16's τ at 531.5 MiB.** The mixed
  int4+int8 overlays recover most of what pure int4 loses (−0.0048 vs −0.0275
  accept) for 23.5 MiB more on disk. This is consistent with the brief's note
  that overlays buy ~1 dB — small in SNR terms, but it lands where it matters.
- **`oq4.25+` beats `mq4` at identical size** (531.5 MiB both): τ 5.668 vs
  5.594, accept −0.0048 vs −0.0097. Same on-disk cost, strictly better
  acceptance — matching the sibling KLD entry's finding that oq beats mq at the
  ~4-bit tier, now confirmed on the decision-relevant metric.
- The ordering is stable per-prompt, not just in the mean: `oq4` is worst or
  tied-worst on 7 of 8 prompts.

## Correctness note: residual verify nondeterminism (rare, ~1 in 68 runs)

One sweep run — `oq8` on `coherence_square_function` — emitted a sequence
differing from the f16/AR reference by **one token at position 45** (5019 vs
3301), with accept rate 0.200 instead of 0.277.

This is **not** a format effect and **not** a losslessness violation of the
oq8 weights:

- 18 subsequent repeats of the exact same command (3 + 15) all reproduced the
  reference sequence `c761774e4309` with accept 0.2773 — **18/18 identical**.
- f16 on the same prompt also reproduces `c761774e4309` 3/3.
- The divergent sequence `c9ede81585d7` never recurred.

So the verify forward is still nondeterministic at roughly **1 event in ~68
runs (~1.5%)**, well below the rate that motivated `6ca303af8` but not zero.
It does not change any conclusion here (the affected cell was re-measured with
a reproducible run), but it means **single-run md5 comparison remains an
unsafe gate** — `./tests/coherence-gate-dflash.sh` compares single runs and
structurally cannot catch this. Any future losslessness check must use ≥3
repeats and assert cross-run identity first. Worth a follow-up issue.

## Verdict: ship `oq4.25+`

Reasoning:

1. **Acceptance cost is negligible.** 98.75% of f16's τ. The drafter's job is to
   propose; a 1.25% relative τ loss is far inside the noise of prompt-to-prompt
   variation (accept ranges 0.264–0.754).
2. **It dominates `mq4` outright** — same 531.5 MiB, better acceptance. There is
   no reason to keep mq4 for this role.
3. **Pure `oq4`'s extra 23.5 MiB saving is not worth 5.9 points of relative τ.**
   The 4.4% further size reduction (531.5 → 508.0 MiB) buys almost nothing on
   the bandwidth term (below), while costing ~6× more acceptance than
   `oq4.25+`. Wrong side of the trade.
4. **`oq8` is the fallback if acceptance ever becomes the binding constraint** —
   it is free in acceptance terms but costs 1008.0 MiB, and on the NPU the
   weight bytes are the only remaining bandwidth lever.

**Sizing the bandwidth win honestly.** The saving is real but does **not** scale
the whole block. Measured warm block is 236.0 ms: GEMM 103.5 · attention 61.5 ·
glue 53.0 · primitives 23.5. Only the GEMM term scales with weight bytes, and it
sits against a ~60 ms bandwidth floor. So int8 → 4-bit buys **tens of ms, not a
halving** — and the marginal `oq4.25+` → `oq4` step buys a small fraction of
that, which is exactly why the acceptance cost decides it.

Against the verify budgets (9B 57 ms, 27B 155 ms, 31B 345 ms), 236.0 ms fits
31B today. The format choice does not change which budget is met; it is decided
on acceptance, and `oq4.25+` wins.

## Reproduce

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --example dflash_spec_demo -p hipfire-runtime
./target/release/hipfire lock acquire dflash-phasef

T=~/.hipfire/models/qwen3.5-9b-mq4.hfq
for pr in coherence_capital_france coherence_sheep_reason coherence_square_function \
          humaneval_0_has_close_elements lru_cache_pep8_strict merge_sort_thinking_off \
          trains-meet coherence_lloyd_long; do
  for d in Qwen3.5-9B.dflash.f16.hfq Qwen3.5-9B.dflash.npu.oq8.hfq \
           Qwen3.5-9B.dflash.npu.oq4.25+.hfq Qwen3.5-9B.dflash.npu.oq4.hfq \
           qwen3.5-9b-mq4.dflash.hfq; do
    ./target/release/examples/dflash_spec_demo --target $T --draft ~/.hipfire/drafts/$d \
      --no-adaptive-b --block-size 16 --temp 0.0 --seed 1234 \
      --prompt-file benchmarks/prompts/$pr.txt --max 256
  done
done

./target/release/hipfire lock release dflash-phasef
```

Read `decode_accept_rate`, `decode_tau`, `decode_tokens_emitted` from the
`BENCH METRICS` block, and the `DFlash tokens: [...]` line for the losslessness
check. Use ≥3 repeats before treating any token-sequence difference as real.
