# A no-op iu4 WMMA probe, and what it says about every compact GEMM

Both "ceiling" numbers in use were indirect: 99.2 TOPS came from a separate
synthetic probe, and ~70 TOPS came from ABLATING a real kernel — which still
carried its register pressure, loop bookkeeping and occupancy. Neither answers
what a kernel designer needs: at a given number of INDEPENDENT accumulator
chains, what can the matrix core actually sustain?

`kernels/src/wmma_iu4_noop_{w64,w32}.hip` are that probe: a dependency chain of
WMMA and nothing else. No global memory, no LDS, no address arithmetic,
loop-invariant operands. `examples/bench_wmma_noop` sweeps the chain count.

## Result 1 — the rate is ~105 TOPS and it is FLAT FROM ONE CHAIN

| chains | w64 TOPS | w32 TOPS |
|---|---|---|
| 1 | 101.7 | 103.5 |
| 2 | 101.5 | 106.9 |
| 4 | 102.1 | 106.9 |
| 8 | 105.9 | 108.1 |
| 16 | 108.0 | 104.4 |
| 32 | 104.4 | 102.9 |

A SINGLE accumulator chain saturates the matrix core. The instruction is fully
pipelined and exposes no latency worth hiding.

**This kills a design assumption.** Accumulator tiles are NOT for latency hiding;
one chain would do. They are for DATA REUSE — more output tiles per wave amortise
the operand loads over more MACs. Register budget should be spent on reuse, and
choosing a tiling to "keep enough WMMA in flight" is optimising a non-problem.

## Result 2 — wave32 and wave64 issue at the SAME rate

Within noise across the whole sweep. So wave64 is **not** faster per instruction,
and the +35% recorded for it elsewhere is not an issue-rate effect. Its real
advantage is register economy: `_w64` holds a 16x16 tile in 4 accumulator GPRs
against wave32's 8 (64 lanes x 4 = 256), so the same register budget buys twice
the output tiles — which is Result 1's lever, data reuse.

## Result 3 — one WMMA is exactly one VALU instruction

PMC on the probe: `SQ_INSTS_VALU / WMMA = 1.00` for every variant, LDS = 0. So
`SQ_INSTS_VALU` counts one instruction per WMMA and nothing else, which validates
subtracting the derived WMMA count from SQ_INSTS_VALU to get "non-WMMA VALU" —
the metric this session has been steering by.

It also means **WMMA and ordinary VALU share an issue port, so every non-WMMA
instruction displaces a WMMA.**

## The consequence: VALU/WMMA predicts fraction of peak

| kernel | VALU/WMMA | LDS/WMMA | measured % of ~105 TOPS |
|---|---|---|---|
| no-op probe | **1.00** | 0 | 100% |
| compact iu4 **wave64** | **2.50** | 0.36 | **51%** (53.4 TOPS) |
| compact iu4 wave32 | **4.56** | 1.06 | **32%** (33.6 TOPS) |

The ratio tracks the achieved fraction closely enough to use as a design target
(it under-predicts slightly, so the port is not perfectly serialised). Reaching
~80% of peak needs VALU/WMMA ~1.25, i.e. cutting non-WMMA VALU from 1.50 per WMMA
to ~0.25.

## Where the wave64 kernel's 1.50 non-WMMA VALU goes

Per group per wave it issues 256 WMMA (16 k-substeps x 16 tiles) and rescales 64
output elements at roughly (convert, multiply, multiply, add) each — about **256
VALU, i.e. ~1.0 of the 1.50**. The per-group rescale is two thirds of the
non-WMMA issue.

That agrees independently with the earlier measurement that this kernel reaches
85/85/75% of the pure integer core, which has no rescale at all. **The rescale is
the dominant remaining cost, and now the mechanism is known: it is not arithmetic
cost in the abstract, it is issue slots taken from WMMA.**
