# FLM reproduction — results

Phase 2 milestone 4: rebuilding `layer.xclbin`'s decode GEMM from our own
mlir-aie/Peano source. **The bar is functional, not textual** — numerically
equivalent on real weights, and within ~10% of the throughput baseline. This
records where the reproduction matched and where it did not.

Working log with the failures in order: `docs/npu/flm-refe-log.md`. Dataflow
findings: `flm-layer-dataflow.md`, `flm-attn-dataflow.md`.

Artefacts:

| what | where |
|---|---|
| GEMV core body | `kernels/npu/flm_gemv_q4_1.cc` |
| container reader + references | `tools/npu/flm/q4nx.py` |
| correctness gate | `tools/npu/flm/gemv_verify.py` |
| throughput gate | `tools/npu/flm/gemv_bench.py` |

---

## 1. The weight format — reproduced exactly

FLM's `model.q4nx` is a plain safetensors container. Every streamed weight
tensor is declared `I8` with a second dimension of **5120 bytes**, and each
5120-byte row is **planar**:

```
[   0: 512]  256 x bf16  d   scales, all positive
[ 512:1024]  256 x bf16  m   mins,   all negative
[1024:5120]  4096 B          8192 packed 4-bit codes
```

`512 + 512 + 4096 = 5120` bytes for 8192 weights = **exactly 5.00 bits/weight**.
That confirms the bpw figure of `flm-layer-dataflow.md` §3 from the byte layout
itself rather than from dividing a manifest by a parameter count, and per-layer
bytes sum to 38,010,880 — the number that document derived independently.

8192 weights per row is four output rows at K=2048 or one at K=8192, which is
why q/k/v/o/gate/up and down all land on the same second dimension.

**Quantizer**: plain min/max asymmetric q4_1 (`d=(max-min)/15`, `m=min`).
`m/d` is −7.4 to −7.5 with std 1.35 across every tensor and `m + 7.5d` centres
on zero — the signature of min/max on symmetric zero-mean blocks. This
corroborates `Dequant::generate_dequant_q4_1_seq` from the symbol table.

**Block axis**: 32 **contiguous input dims** (K-major). Settled by per-block
scale spread, which separates the two orientations cleanly because outlier
channels make N-major spread much heavier:

| tensor | stored `d` p99/p1 | K-major | N-major |
|---|---|---|---|
| `q_proj` | **7.18** | **7.01** | 31.77 |
| `down_proj` | **2.49** | **2.42** | 2.51 |

This went **against** the ISA reading. `vextbcst.16` + `mac_elem_16` suggests
broadcasting one activation scalar into contiguous *output* weights, which
would make the layout N-major. The statistic says otherwise, and the statistic
is the direct evidence.

## 2. Where the reproduction did NOT match — FLM's weights are transformed

**FLM's stored codes are not a quantization of the published checkpoint.** Four
independent searches for the underlying float weights all returned the noise
floor, each against a control that discriminates cleanly:

| search | best score | control |
|---|---|---|
| `lm_head` block-scale fingerprint vs `embed_tokens` (tied, so the container carries its own ground truth) | r=0.56 | self 0.99996, best wrong row 0.55 |
| row sums (invariant to element order) | no better than the 1.9e-5 sum-distribution gap | — |
| N-major and strided fingerprints vs HF `consolidated.00.pth` | r≈0.5 | self 1.00, best other 0.30 |
| sorted 256-scale multiset per row (exact if blocks are contiguous-K and merely permuted in storage) | 3.3e-05 | self **0.000e+00**, best other 3.79e-05 |

The last was run against **both** the Instruct and the base checkpoint. Neither
matches.

Aggregate statistics match untransformed K-major weights to ~2% while every
per-block fingerprint fails. That pair of facts is consistent with a **mild
per-channel transform applied before quantization** — SmoothQuant/AWQ-style
scaling folded into the activation — which changes each block's range while
leaving the population of ranges nearly unchanged. A Hadamard control shows
rotations also leave these statistics nearly unchanged (`std/mean` 0.4297 vs
0.4580), so the evidence establishes *that* there is a transform, not *which*.

**Consequence, and it is the honest limit of this milestone.** The dequantized
weight matrix cannot be reconstructed from public weights, so an end-to-end
"same logits as FLM on the same prompt" comparison needs one fact we do not
have: which `(output row, k-block)` each of the 256 stored slots maps to. The
arithmetic does not depend on it — the block layout within a row is fully
decoded — but the output indexing does. **This is the gate on the final
equivalence check, and it is open.**

## 3. The core body — exact

`kernels/npu/flm_gemv_q4_1.cc`, verified on **real q4_1 data from FLM's own
container** (real scales, real mins, real codes):

| shape | vs bf16-faithful reference | vs exact float64 |
|---|---|---|
| K=2048, 8 rows, 1 tile | **1.9e-07** | 8.2e-03 (0.99% of \|out\|) |
| K=2048, 512 rows, 64 tiles | **3.3e-07** | 1.2e-02 (1.69% of \|out\|) |

The device reproduces a bf16-faithful emulation of the body to float32
round-off. The deviation from float64 is entirely the **format's** cost, which
the emulation reproduces exactly, and it is the same cost FLM pays — FLM
materialises `w = d*q + m` in bf16 before its own MAC.

Reporting only the float64 number would have read as a 1% failure. Both are
reported for that reason; `gemv_reference_bf16` is the gate,
`gemv_reference` is context.

### What the body does differently from FLM, deliberately

The dequant folds out of the inner loop. With `w = d*q + m`:

```
out[n] = sum_b ( d[n,b] * sum_t q[n,b,t]*a[b,t]  +  m[n,b] * sum_t a[b,t] )
```

The zero-point term becomes one scalar per block against an activation
block-sum **shared by every output row in the tile**, and the codes enter the
MAC as exact small integers. FLM instead spends a 42-op dequant chain
materialising bf16 weights. That chain is not reproduced, and does not need to
be: weight supply is **2.57 MACs/cycle/core** against the MAC unit's measured
**512**, so the body is built for correctness and for bytes, not for
arithmetic throughput.

That framing held for the *shape* of the body but not for its cost: at 16 cores
the reproduction turned out to be **compute-bound before it was
bandwidth-bound**, and closing that took the native int4 operand path plus a
wider tile (section 5). "The MAC unit is oversupplied" is about MAC issue
capacity; the unpack and rescale work around each MAC is not free, and it is
what actually set the rate.

This is also the cleaner provenance position: the arithmetic identity is ours,
derived from the format facts, not a transcription of FLM's schedule.

## 4. Ingress width — open thread 1, closed

Measured with `dispatch_bw_probe.py --full-elf --verify` at decode-realistic
totals:

| feed streams | 16 x 32 MiB = 512 MiB | 48 x 16 MiB = 768 MiB |
|---|---|---|
| 2 | 27.1 | 27.1 |
| **4** | **48.6** | **48.6** |
| 6 | — | 43.4 |
| 8 | 56.2 | 55.6 |
| 12 | — | 56.0 |

The 4-stream figure reproduces to three digits across two totals, one of which
(768 MiB) is FLM's actual per-token traffic of 772.3 MB.

- **4 streams deliver 48.6 GB/s, 5.2% above FLM's 46.2**, and FLM feeds its 16
  GEMM cores from 4 memtile columns. The hypothesis survives the sharper test.
- **8 streams give ~56 GB/s, +15%, and saturate there** (12 adds 0.4).
- This reproduces milestone 3's 56.34 GB/s at the same shape to **0.25%**,
  which incidentally bounds `flm serve` contention on these numbers as
  negligible.

Not proof — FLM's 46.2 includes compute and its streams are not these — but the
prediction the hypothesis makes is what the hardware does, at two totals, with
the bytes verified.

**Recorded rather than smoothed over**: 6 workers measures 43.4 GB/s, *below*
the 4-worker figure. 6 does not divide 8 columns evenly, so two columns carry
two streams and four carry one. Stream count is not the whole story;
**stream-to-column balance is.**

## 5. Throughput — 48.1 GB/s at 16 cores, 1.04x FLM. The bar is met.

`gemv_bench.py`, K=2048, 16 rows per weight tile, 116 tiles per core — so
16 cores carry **38.0 MB, exactly one llama decoder layer's weights** — in ONE
dispatch, with every point correctness-checked against the bf16 reference on
the bytes it actually streamed:

| cores | MB | GB/s | wall us | vs FLM 46.2 | max err |
|---|---|---|---|---|---|
| 8 | 19.0 | 24.4 | 778.9 | 0.53x | 6.0e-07 |
| **16** | **38.0** | **48.1** | **790.8** | **1.04x** | **6.0e-07** |

A repeat run measured 48.6 GB/s / 1.05x. **That is within ~10% of the baseline
— above it — so the milestone-4 throughput bar is met**, and it is 99% of the
48.6 GB/s that 4 feed streams deliver with *no* compute at all, i.e. the body is
now essentially free against the ingress it is fed through.

### How it got there — four changes, each measured

The first working version ran at 14.4 GB/s, 0.31x FLM, and was **compute-bound**:
wall time stayed flat at ~1315 us across an 8x change in cores and bytes while
GB/s doubled at every step. That is what identified the body, not the dataflow,
as the constraint — milestones 1-3 had already shown one command delivers
56 GB/s.

| change | GB/s | why |
|---|---|---|
| baseline (mask chain, 8 rows/tile, 8 cores) | 14.4 | — |
| **native uint4 operand** | 18.5 (+28%) | see below |
| **16 rows per tile** | 24.8 (+34%) | widest that fits L1 double-buffered |
| **16 cores, paired** | **48.1** (+94%) | the shim fan-out FLM uses |

**The native int4 operand path was the missing understanding.** FLM's GEMM cores
carry `vunpack:64` and `vups.4x:64` — the AIE2P instructions that widen a packed
int4 vector directly. The first body never reached them: it handed the hardware
`uint8` lanes and spent a generic `bit_and` -> `unpack` -> `to_float` chain doing
what those instructions do natively. Loading the codes as
`aie::vector<uint4, 64>` instead compiles to **75 instructions against 103** for
the identical loop — `vband` disappears and the widening rides the load as
`vldb.unpack` — and measured **+28% on hardware**, matching the −27%
instruction-count prediction almost exactly.

Nibbles are stored in plain element order for this. **That is a convenience,
not a requirement, and an earlier version of this document got it wrong.** Any
nibble order can be consumed for about one extra instruction using the AIE2P
shuffle network (the `T8_*`/`T16_*` transpose modes, exposed in aie_api as
`interleave_zip`/`interleave_unzip` and `shuffle_up`/`shuffle_down`):
llama.cpp's split form unpacks to lanes `[e0,e16,e1,e17,…]`, which
`interleave_unzip(lo, hi, 1)` separates in one op — **76 instructions against
this loop's 75**. The claim that the split form "costs 25 vector ops per 64
weights against 18" described an `aie::concat`-based gather, not the format.
This matters directly: FLM's own nibble order is not yet known, and the shuffle
network means matching it will be nearly free rather than a reason to repack.

**16 rows per tile is bounded by L1, not by taste.** At 16 rows the weight tile
is 20480 B, double-buffered 40960, plus the activation: 49 KB of the 64 KB tile
memory. 24 rows would need 68 KB.

**Pairing the cores is FLM's structure and it is what makes 16 cores fit.** One
shim stream feeds each pair and a memtile splits it in two; the pair's two
result streams are joined back into one before the shim. 16 private weight
streams plus an activation is 17 shim inputs against 8 columns x 2 channels, and
the placer rejects that outright:

```
no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels
```

This is exactly what `flm-layer-dataflow.md` predicted — `layer.xclbin`
packet-switches at the shim (24 of its 42 shim routes) precisely because many
DDR streams must be multiplexed onto few channels, and "a naive all-circuit
reproduction will not fit the channels". Splitting in the memtile is the other
way to spend the same budget, and it reproduces FLM's 16-streams-from-4-memtiles
and its N-split concatenating pair at the same time.

**Two things tried that do not help, recorded so they are not re-tried.** A
64-lane form — one MAC per 64 weights with the two block scales joined by
`aie::concat` instead of two of everything — compiles but is **78 instructions
against 75**, so it loses. And two
independent accumulators to break the MAC dependency chain does not compile — a
second 32-lane float accumulator makes the backend emit a 16-lane float add it
cannot legalize (`G_FADD <16 x s32>`), the same limit that forced the zero-point
term to share the dot accumulator. Marked `ponytail:` in the kernel.

**Where the remaining headroom is.** 8 -> 16 cores still scales at 1.97x, so
per-core compute has not stopped mattering; but at 48.1 GB/s the body is at
99-100% of the 4-stream ingress figure, and the fabric roof is 56.5. The next
lever is ingress width (open thread 1), not the kernel body.

### What this rate implies for future formats (phase 4)

Worth stating here because the phase-4 plan was written against "the MAC unit is
oversupplied 199x", which is a statement about **MAC issue capacity** and not
about the per-weight decode budget. Measured: **3.01 GB/s/core = 2.67
weights/cycle/core** at 5.00 bpw. The rate needed to saturate the 56.5 GB/s
fabric, by format and core count:

| bpw | 16 cores | 32 cores |
|---|---|---|
| 5.000 (q4_1 today) | 3.14 | 1.57 |
| 4.125 (oq4++) | 3.80 | 1.90 |
| 3.000 (QTIP-3) | 5.23 | **2.62** |
| 2.000 (QTIP-2) | 7.85 | **3.92** |

Fewer bits per weight means more weights per delivered byte, so the decode
requirement rises exactly as fast as the bandwidth win — they trade rather than
compound. On the full 32-core array a QTIP-3 decoder must be **no more expensive
per weight than this int4 unpack** (2.62 needed against 2.67 measured), and
QTIP-2 must be ~1.5x cheaper.

The counterweight: this body issues ~13 vector ops per 64 weights yet achieves
only 2.67 weights/cycle, an effective **0.54 ops/cycle** — it is latency- and
dependency-bound, not issue-bound, and the VLIW slots are largely idle
(corroborated by the second accumulator failing to legalize and unrolling not
helping). A sequential trellis decoder has **NROWS independent chains available
per tile** (16 output rows, no dependency between them), which is the structure
that fills idle slots. So the phase-4 gate should be measured with several rows
in flight rather than on one chain.

## 6. Traps found building this

Full detail in the log and in `tools/npu/flm/README.md`. Each cost real time.

1. **`iron.jit` does not hash `ExternalFunction(source_file=...)`.** Editing the
   kernel `.cc` silently reuses the cached xclbin and the run reports the *old*
   kernel's numbers. It presented as a fix that did nothing, and made two
   identical expressions in one probe disagree. Closure-captured shapes have the
   same hole. Pass both as design-level `source_files=` / `CompileTime[int]`.
2. **The default rounding mode truncates** — 13% error on a real row until
   `aie::set_rounding(conv_even)` is called, against 0.19% after.
3. **Scalar reductions mixing bf16 loads with a float accumulator miscompile**,
   silently dropping terms, while every operand reads back correct on its own.
4. **IRON's default worker stack is 1024 B**, and overflow is silent. Anything
   vector-loaded needs `alignas(64)`.
5. **AIE2P backend limits**: `aie::downshift` on uint8 segfaults the compiler;
   16-lane uint8 and 16-lane float ops fail to legalize.

The habit that found (1) and (3): **change the KIND of evidence.** (1) fell out
of comparing two expressions that must agree and did not; (3) fell out of
scoring the device's number against a list of candidate mis-pairings, where
"even blocks only" matched to six digits.
