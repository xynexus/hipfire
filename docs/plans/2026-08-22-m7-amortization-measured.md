# §M7 — the capacity thesis, measured

Status: measured 2026-08-22, nix1 / gfx1103.

§M7's exit is a measurement, and it does **not** depend on M4's module split
existing. The amortization ratio is a property of the routing distribution, not
of the executor that consumes it — so it can be measured today, and it was.

## Instrument

No new capture was written. Master already carries the whole thing:
`record_moe_router_selection` / `moe_router_histogram_active`
(`hipfire-dispatch/src/moe_telemetry.rs:96`, `:146`), called from the decode
top-K site (`qwen35/moe_decode.rs:1246`), accumulating per-layer
`router_topk_histogram` plus pairwise `topk_cooccurrence`, and dumped as
evidence JSON by `run.rs --evidence-dir`.

Three deliberately unrelated prompts (algorithms, poetry+translation, clinical
differential) on `Qwen3.6-35B-A3B--oq4`, kvarn KV, 96 generated tokens each:
**256 experts, top-8, 40 MoE layers, 14 560 routed tokens** (364 per layer).

Sanity check that the histogram means what it is being read to mean:
`sum(q_e)` at layer 0 = **8.0000**, exactly `k`. 250 of 256 experts are used at
least once.

## The number

For N sessions each decoding one token, with `q_e` the marginal probability that
expert `e` appears in a token's top-k at that layer:

```
E[distinct experts] = Σ_e (1 − (1 − q_e)^N)
ratio               = E[distinct] / (N × k)
```

| N | ratio | 1/ratio | min layer | max layer | distinct experts/layer |
|---|---|---|---|---|---|
| 1 | 1.0000 | 1.00× | 1.000 | 1.000 | 8.0 of 256 |
| 4 | 0.8650 | **1.16×** | 0.822 | 0.935 | 27.7 |
| 16 | 0.5749 | **1.74×** | 0.500 | 0.733 | 73.6 |
| 64 | 0.2582 | **3.87×** | 0.217 | 0.373 | 132.2 |
| 128 | 0.1518 | **6.59×** | 0.128 | 0.217 | 155.4 |

N=1 returning exactly 1.00 is not a result, it is the identity `Σ q_e = k`
falling out — it is there because if it had *not* come out at 1.00 the rest of
the table would be wrong.

## Is it falsified?

§M7 says: *falsified if there is no crossover below the N whose KV fits in
VRAM.*

**The VRAM ceiling on this box is N ≈ 129.** Measured, not estimated: the daemon
reports `runtime_session_bytes` growing 767.6 MiB per 4 sessions at max_seq
4096 under kvarn = **191.9 MiB/session**. Against 43 008 MiB total with the
18.2 GiB model resident, ~24.8 GiB remains → ~129 sessions.

So the entire measured range lies under the ceiling, and amortization inside it
is 1.16× → 6.59×. **Not falsified.**

## Two honest qualifications

**1. This is the weight-byte ratio, not a wall-clock crossover.** The second
half of §M7's exit — the N at which module-major beats layer-major *in time* —
cannot be measured while module-major does not exist. What can be said is where
the saving becomes convertible: the plan's own launch-trace measurement (§0.4
commentary) found width-1 MoE decode issues 1322 launches at ~8.5 µs each and is
**launch-bound, not bandwidth-bound**, while width-16 runs the same 1322
launches at 66 µs each and has "crossed into real work". Weight-byte savings buy
time only in the second regime. That places the useful crossover around N≈16,
where this measurement independently says the saving is 1.74×.

**2. The model assumes sessions route independently.** Correlation *beyond* the
marginals — different sessions favouring the same experts for reasons `q_e` does
not already capture — would produce MORE sharing, not less. So these figures are
a **lower bound** on amortization, which is the safe direction for a go/no-go.

## It corrects §0.4's estimate

§0.4 predicted "~1.05× at 8 slots and ~1.14× at 16". Measured is 1.74× at 16 —
well ahead. Two reasons, and neither is that the estimate was careless:

- §0.4 reasons about a **512**-expert model (`n_exp/k = 512/8 = 64`); this
  artifact has **256** (`n_exp/k = 32`), so sharing begins at half the N.
- The estimate treats routing as uniform. Real routing is not: 250 of 256
  experts are used, but unevenly, and non-uniformity raises collision
  probability. Skew helps module-major.

The prediction was right about the shape and conservative about the magnitude.
