# DFlash 9B host-glue threading A/B + latency percentiles (Phase 0 task #39)

nix1 / npu1 (Phoenix gfx1103 APU, 8-core Zen4, UMA). `dflash_body_native`
harness, W4A8 multicore, `--gemm multicore --cpu-primitives --ctx-cache --attn
flash --pipeline-glue`, l_ctx=32, B=16, `r14_1x2x128_nb128` (M_TILE=16), 12
blocks, warm-cached samples (blk>=2). rayon (already a dep) threads the host
glue behind opt-in flags; the serial path stays default and reproducible.

Flags added: `--thread-glue` (threads the int32→f32 rescale), `--thread-pack`
(adds the A-stripe pack; implies `--thread-glue`), `--thread-quant` (adds the
on-chain per-row quant — a measured regression, kept only to reproduce the null).
All three are **bit-identical** by disjoint-region construction: every threaded
run gives cos vs golden = 0.899149 and same-run `--cmp` parity cos =
1.000000000, max|Δ| = 0.000e0.

## Per-component A/B (steady warm, HIPFIRE_PIPE_TRACE, ms/block)

| component | where it runs | serial | threaded | verdict |
|---|---|---|---|---|
| quant | on the layer critical path | 1.7–2.0 | **3.5–4.1** | **REGRESSION** — rayon fork/join overhead exceeds the per-row work (16 rows × 20 GEMMs/block); hits the wall directly |
| rescale | in the NPU wait window | 6.3–6.9 | **5.4–6.1** | mild win (~1 ms); parallelizes cleanly over slots |
| pack | overlaps weight streaming (UMA) | **16–17** | **10–11** | win — see UMA note |

## UMA-contention verdict on threading the pack: it HELPS (refutes the worry)

The brief flagged that threading the pack might worsen UMA-bus contention (task
#30 saw the serial pack inflate ~14→20 ms under overlap). Measured here, the
opposite holds:

- serial pack, **no** pipeline (NPU idle, uncontended) = **10.5 ms**
- serial pack, **under** pipeline (contends with weight streaming) = **16 ms**  ← the UMA penalty is real
- 4-way stripe-parallel pack, under pipeline = **~11 ms**  ← recovers to ~uncontended

Threaded packing spreads the A-stripe memory movement across the 4 block-row
regions, so it overlaps weight streaming instead of serializing behind it. It
**mitigates** the UMA-contention inflation (16→11) rather than worsening it, even
though it discards the serial path's within-dispatch replication (so it does more
total byte work). Net: threading the pack is safe and helpful on Phoenix UMA.

## Block-wall A/B (pipeline, warm, ms) — p50 / p99 / spread

| config | wall(min) | p50 | p90 | p99 | max | spread |
|---|---|---|---|---|---|---|
| serial glue (baseline) | 83.8 | 84.9 | 87.1 | 88.6 | 88.6 | 5.8% |
| `--thread-glue` (rescale) | 82.3 | 84.0 | 84.8 | 86.2 | 86.2 | 4.7% |
| `--thread-glue --thread-pack` | **81.0** | **81.4** | 83.0 | **84.0** | 84.0 | 3.6% |
| `--thread-quant` | 85.8 | 87.5 | 88.0 | 88.9 | 88.9 | 3.6% |

Best config (glue + pack) shaves **~3.5 ms (~4%)** off the ~84 ms baseline →
**~81 ms**, and **tightens the tail** (p50→p99 4.4% vs baseline's 4.4%, overall
spread 5.8%→3.6%). Threading the quant alone REGRESSES ~2 ms, confirming the
component measurement.

## Why the wall win is small: the composed wall is NOT glue-bound

Under `--pipeline-glue`, pack + rescale already overlap NPU weight streaming, so
shaving them barely moves the wall. The pipeline trace shows the wall is gated by
`wait` (NPU weight streaming, ~38–43 ms) + serial `submit` ioctl (~6 ms) +
on-chain quant (~2 ms), none of which glue-threading touches. The ~3.5 ms win
comes from pack-threading removing the UMA-contention penalty inside the overlap
window, which slightly shortens the unhidden remainder and smooths jitter.

## Latency verdict: no tail pathology → CPU isolation NOT warranted

Across every config, p99 sits within ~4–5% of p50 and the max outlier within ~5%
of p50 (baseline spread 5.8%, best 3.6% — at/near the r135 ~3.4% noise floor).
There is **no tail-latency problem for cpuset/affinity/isolcpus to fix**;
threading in fact tightens the tail rather than needing isolation to. Do not
enable isolcpus for this workload on the strength of this data.

## Bottom line

Threading the host glue is a small, safe, bit-identical win (~3.5 ms / ~4% via
glue+pack) that mainly buys tail-tightening, not headline throughput — because
the pipelined wall is NPU-weight-streaming-bound, not glue-bound. The one
counter-intuitive result worth keeping: threading the A-pack MITIGATES Phoenix
UMA contention (16→11 ms under overlap) rather than worsening it. Threading the
on-chain quant is a net regression at this row count and is left off by default.
The real remaining wall lever stays the NPU weight-byte floor (fewer bits,
acceptance-gated) plus the ~6 ms serial submit ioctl — not host-glue threading.
