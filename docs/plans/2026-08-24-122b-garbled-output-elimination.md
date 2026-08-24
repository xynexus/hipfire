# The 122B garbles, and it is NOT compact residency — what was eliminated

**Date:** 2026-08-24 · **Box:** halo, gfx1151, 124 GiB usable
**Artifacts:** `Qwen3.5-122B-A10B--oq4.25++fix.hfq` (68.62 GB), `122b-lmbf16.hfq`

## State

The 122B loads (82.01 GiB peak GTT, against ~137 expanded), runs without fault,
and emits structured-but-wrong text: `<think>` blocks, real tokens, broken
words. Deterministic across runs. It is NOT fixed.

Every hypothesis connecting this to compact-resident routed experts has been
eliminated by measurement. Recording them so the next person does not re-run
them.

## Eliminated

| hypothesis | test | result |
|---|---|---|
| lm_head (its known historical defect) | `122b-lmbf16.hfq`, Bf16Lut3 head | garbles identically |
| leftover Q8F16 tensors (the other known defect) | quant histogram | none present in either artifact |
| ~~dense compact residency~~ | ~~narrow compact to routed shapes, dense expanded~~ | **THAT TEST WAS INVALID — see below** |
| dense compact residency (properly) | real-weight parity on DeltaNet in_proj/out_proj, shared expert, lm_head | 10/10 PASS at ~1e-7 |
| routed compact, every layer | real-weight sweep, 48 layers x 2 projections x 16 experts | 96/96 PASS at ~1e-7 |
| router weights | quant histogram | BF16 — compact residency never touches them |
| KVarN batched prefill (known 57x-less-faithful path) | `HIPFIRE_KVARN_BATCHED_PREFILL=0` | byte-identical output |
| lowered vs hand forward executor | `HIPFIRE_FORWARD_LOWERED=0` | byte-identical output |
| compact kernel at this model's shapes | parity at gate_up [2048,3072] (ng=12) and down [3072,1024] | 21/21 PASS, all layouts |
| stride table contents | `HIPFIRE_MOE_FEED_DEBUG=1` dumps them in situ | exactly right: 226x136 + 30x0, matches the loader |
| representative dtype (expert 0 vs layer) | fixed decode AND prefill | output unchanged -- expert 0 was already compact |
| AWQ sidecar attachment differing by load path | census under forced expansion | identical either way (226 gate_up / 205 down) |
| grouped path-2 prefill (unvalidated kernel) | `HIPFIRE_MOE_GROUPED_GEMM=0` | output unchanged |
| indexed vs generic dispatch | `HIPFIRE_MOE_COMPACT_INDEXED=0` | both wrong, differently |
| **mixed-precision layers** | **forced-mix on a 35B with a known-good reference** | **BYTE-IDENTICAL** |

The last row is the strongest. `HIPFIRE_MOE_FORCE_EXPAND_EVERY_N=8` manufactures
a genuinely mixed layer (32 Oq8 + 224 compact, expert 0 among the Oq8) on a model
that has none, and the 35B-A3B stayed byte-identical to its expanded reference.
Mixed handling -- kernel, stride table, representative, dispatch -- is correct
end to end on a real model against a real reference.

## CORRECTION: the first "dense exonerated" result was invalid

The narrowing used `ONLY_M=2048,3072` with `ONLY_K=3072,1024`, and those lists are
**ANDed per tensor**. Every `[3072, 3072]` attention tensor matches both, so it
stayed compact. Dense was never actually expanded, and the conclusion drawn from
that run -- "garbles with dense expanded, therefore dense is exonerated" -- did
not follow. Dense is now exonerated properly, by direct real-weight parity
against an f64 oracle rather than by inference from a config I misread.

The lesson generalises: `ONLY_M`/`ONLY_K` cannot separate routed from dense on
this model at all. Routed `down` is `[3072, 1024]` and the shared expert's
`down_proj` is also `[3072, 1024]`. Shape is not an identity.

## The corruption is upstream of every dispatch choice

The 122B produces BYTE-IDENTICAL garbage across all of: lowered vs hand forward
executor, batched vs per-token prefill, indexed vs generic MoE dispatch, grouped
path-2 on and off, KVarN batched prefill on and off, and both lm_head variants.

Whatever is wrong is therefore not a dispatch selection, and not any of the
kernels those levers switch between. Combined with 106 real-weight parity checks
passing, it is not the compact weight decode either.

## Compact decode verified on the 122B's REAL weights

Added `parity_gemv_oq_compact_moe --hfq <artifact> <layer> <n_exp>`: reads real
routed-expert bytes straight out of the container and runs both decode paths plus
an oracle. **No model load**, so it works on a model too large to serve.

Layers 0, 1, 24, 47 of `oq4.25++fix`, 16 experts each: all PASS at ~1e-7, with
compact-kernel, production-expansion and oracle all agreeing.

It first reported FAIL on layers 0 and 1 (2.16e-1) and PASS on 24 and 47, which
looked like the smoking gun. It was the ORACLE: the harness's hand-rolled
f16->f32 flushed subnormals to zero, and real group scales go subnormal while
generated ones do not. The tell was `compact-vs-expanded = 2.3e-7` while BOTH
disagreed with the oracle by the same 1.7e-3 -- when the two independent
implementations agree and only the referee objects, the referee is wrong. Now
uses `hipfire_primitives::conv::f16_to_f32`.

Lesson worth keeping: an oracle that is only correct on the data the test
generates is not an oracle.

## The one anomaly, unexplained

Expanding a QUARTER of the 122B's compact experts
(`HIPFIRE_MOE_FORCE_EXPAND_EVERY_N=4`, 98.97 GiB) markedly improves the output:

> "The sky appears blue due to a phenomenon called Ray scattering.. When
> sunlight reaches Earth's atmosphere,, it interacts with gas"

against garbage for the same prompt at N=0. Deterministic both ways.

Expansion is NUMERICALLY LOSSLESS -- `oqplus_compact_to_moe_oq8_blocks` decodes
the same nibbles and bakes in the same overlay -- and the AWQ census is identical
across it. So identical weights and identical scales produce different output.

What DOES change is allocation SIZES, and therefore GTT layout. That is the
signature of an out-of-bounds read somewhere that lands on different neighbouring
data. It is not in the compact GEMVs: their addressing sums exactly to
`M*ng*block_stride` (nibble plane `M*ng*128` + side plane `M*ng*8`), and the Oq8
branch to `M*ng*260`, both verified.

Prime suspect for the next session: a consumer that sizes a routed expert from
its DTYPE rather than its actual buffer length. A reader assuming Oq8's 260 B
blocks would compute 6.39 MB for a tensor that is 3.34 MB inside a 4 MiB
allocation, and run 2.4 MB past the end into the neighbouring expert -- mapped
memory, so no fault, just wrong weights that shift when the layout shifts.

## Not reproducible on the 35B

The 35B-A3B is byte-identical under every configuration tried: all-compact,
all-expanded, forced-mix, indexed, generic, per-projection narrowing. Whatever
the 122B hits is specific to it -- 48 layers, dim 3072, mi 1024, an MTP head, and
1,288 natively-Oq8 routed experts.

## What cannot be settled on this box

The decisive A/B -- the same artifact loaded fully expanded -- needs ~148 GiB and
does not fit in 124. Until that runs somewhere larger, "was the 122B ever correct
on this tree?" is open: it could not load at all before compact residency, so the
89-commit merge that preceded this work has never been tested against it.

## Diagnostics left behind

- `HIPFIRE_MOE_FEED_DEBUG=1` — dumps the indexed branch's own inputs (||x||^2,
  activation head, per-expert stride tables with a zero count, and the per-layer
  compact/Oq8/AWQ census). Its most useful property is printing NOTHING when the
  branch is not the one executing, which is how the earlier feed bug was found.
- `HIPFIRE_MOE_FORCE_EXPAND_EVERY_N=<n>` — forces every nth expert through the
  expansion path, manufacturing a mixed layer on a uniform model so mixed
  handling can be tested against a known-good reference.
