# RESULT: GTT slab suballocator — 2026-09-02

**Step 2 is refuted by measurement: it would have DOUBLED GTT while making the
brief's own verification bar read perfect.** Step 3 is not worth building on
this evidence. The tool the brief rests on was measuring the wrong shape, and
that has been fixed so the next person cannot repeat the error.

## Step 2 — "round bulk slab requests up to 2 MiB" — measured, and it is harmful

Same workload (2000 expert pairs), `hipfire-rdna` example `gtt_granularity`,
reading the driver's own `mem_info_gtt_used`:

| arm | requested | **GTT consumed** | ratio |
|---|---|---|---|
| **A** as-is, interleaved `gate_up, down` | 3.113 GiB | **3.904 GiB** | 1.254x |
| **B** each request rounded up to 2 MiB | 7.812 GiB | **7.812 GiB** | **1.000x** |

Rounding up **doubles** consumption, 3.904 -> 7.812 GiB.

The trap is that it looks like a win. The brief's bar was *"consumed GTT
approaches the raw byte count"* — a **ratio** — and arm B scores a perfect
1.000x. It gets there by inflating the denominator. **A ratio cannot be the bar
for a change that alters what is requested.** The bar has to be absolute GTT
consumed for a fixed workload.

## Why: the driver already packs, and the tax is the block TAIL

The brief's premise — *"The rule is ALIGNMENT, not size ... any request that is
an exact multiple of 2 MiB costs exactly its size"* — is true but not the whole
mechanism. The driver **suballocates from 2 MiB blocks**, so sub-block requests
pack, and what a request costs depends on what is next to it:

| stream | consumed | note |
|---|---|---|
| `1114112` alone, uniform | 3.904 GiB | 1.881x — one per block |
| `557056` alone, uniform | 1.301 GiB | 1.254x — three per block |
| **`1114112, 557056` interleaved** | **3.904 GiB** | the down_proj rides **free** |
| + `417792` third (fits the leftover) | **3.904 GiB** | **also free** — +0.778 GiB of data, +0 GTT |
| + `524288` third (overflows) | 5.207 GiB | one size class larger, +1.3 GiB |

`1114112 + 557056 = 1671168`, which fits one 2 MiB block, so the pair costs one
block — and the block was already being paid for the `gate_up` alone. The
leftover is `2097152 - 1671168 = 425984 B`, and anything that fits it is free.

So the lever is **filling the block tail** — ordering and genuine
suballocation — and never rounding up, which throws the tail away by
construction.

## Step 3 — do not build it. The tax it targets is already gone.

`HIPFIRE_ALLOC_REPORT=1` on a real `Qwen3.6-35B-A3B--oq4.25++` load,
**after** the MoE expert stacking landed (`484af81bd`):

    hipMalloc: 35.36 GiB across 43729 allocations, 46 distinct sizes
        10.62 GiB      40 x 285212672 B   <- stacked gate_up, 136.000 x 2 MiB, EXACT
         5.31 GiB      40 x 142606336 B   <- stacked down,     68.000 x 2 MiB, EXACT
         0.95 GiB       1 x 1017118720 B  <- 485.000 x 2 MiB,  EXACT
        10.64 GiB   10250 x 1114112 B     <- TRANSIENT: per-expert, freed after the memcpy
         5.41 GiB   10420 x 557056 B      <- TRANSIENT: same

(The report is cumulative `hipMalloc`, not peak — the 10250/10420 rows are the
staging allocations the stacking path frees one at a time, which is why peak GTT
is 19.13 GiB, not 35.36.)

Summing rounding waste across every large allocation in the report: **0.193
GiB**. Peak GTT is 19.13 GiB against a 17.88 GiB payload — **1.070x**, and part
of that remainder is KV and scratch, not weights.

A slab suballocator for load-once weights would have to make
`GpuTensor::sub_offset` views a type-level ownership guarantee to avoid
reintroducing **#262** (an alias returned to the pool and came back as scratch
over live weights). That is real design work, and the measured prize on this
model is under 0.2 GiB. **Rung 1 of the ladder: it does not need to exist.**

## What was changed

`crates/hipfire-rdna/examples/gtt_granularity.rs` now accepts a
**comma-separated list** allocated round-robin, so it can measure an interleaved
stream instead of one size in a loop. The single-size mode is what produced the
brief's 1.881x/1.965x figures and the resulting 9x-optimistic prediction in the
MoE brief; the header now says so, with the numbers, so the next reader does not
re-derive it.

## When to revisit

If a future artifact's per-tensor sizes sum to a poor fit against 2 MiB — a
stream whose block tails are consistently large — the fix is **ordering/packing
to fill the tail** (arm D above: free), never rounding (arm B: 2x). Re-measure
with an interleaved run before assuming a tax exists.
