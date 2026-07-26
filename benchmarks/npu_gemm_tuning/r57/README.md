# R57 — bandwidth-first production kernel ladder

R57 grows the admitted EmbeddingGemma NPU kernels from R1/R56's trace-timed
external feed. It keeps each stage independently measurable and does not add
nibble decode or MMUL until the production DMA geometry is understood.

## Steps 0/1: `FEED_ONLY` and `PRODUCTION_DMA`

`production_dma_run.py` uses R1's minimal DCE-guard consumer while replacing
the synthetic transfer extent with the exact R34 resident attention buffer:

| component per active column | 16 KiB blocks | wire bytes |
|---|---:|---:|
| paired QKV | 45 | 737,280 |
| output projection | 72 | 1,179,648 |
| residual/norm state | 8 | 131,072 |
| total | 125 | 2,048,000 |

Four production columns transfer exactly 8,192,000 bytes, matching
`NpuEmbeddingLayerAttentionDenseW8::weight_bytes()`. Controls at one, two, and
eight columns retain the same 125-block per-column schedule; only four columns
is labeled `production_exact=1`.

Every result separates:

- `wire_bytes`: all DMA-requested bytes, including padding and replication;
- `nonpadding_bytes`: bytes in meaningful regions of physical blocks;
- `semantic_unique_bytes`: source values after removing cross-column and
  M-tile replication.

The four-column exact profile has 8,192,000 wire bytes, 8,159,360 non-padding
bytes, and 2,558,980 semantically unique bytes. The ratio is visible in the CSV
rather than being reported as useful model bandwidth.

Run the pure accounting tests without hardware:

```bash
cd benchmarks/npu_gemm_tuning/r57
python3 -m unittest -v test_profile.py
```

Run the locked AIE2P matrix:

```bash
benchmarks/npu_gemm_tuning/r57/run_r57.sh
```

Defaults are columns `1,2,4,8`, three fresh-process repetitions each, and output
`benchmarks/npu_gemm_tuning/results/r57-production-dma-20260712.csv`. Override
with `R57_COLUMNS`, `R57_REPEAT`, or `R57_OUTPUT`. Transient raw trace text/JSON
is kept under `~/.hipfire/r57-traces`; override with `R57_TRACE_DIR`.

The trace interval begins with the first compute-tile S2MM receive event and
ends with the last. Allocation, host initialization, BO synchronization, JIT,
and dispatch setup remain outside the bandwidth result, as in R1/R56.

## Step 2: `MEMTILE_STAGE`

This mode changes only the route to a one-to-one
shim -> memory-tile -> compute-tile forward. It retains the exact block schedule,
minimal guard consumer, and CSV schema from `PRODUCTION_DMA`, so any delta is
attributable to the memory-tile hop.

```bash
R57_MODE=MEMTILE_STAGE benchmarks/npu_gemm_tuning/r57/run_r57.sh
```

The default output is
`benchmarks/npu_gemm_tuning/results/r57-memtile-stage-20260712.csv`.

## Step 3: `MEMTILE_BROADCAST`

This mode retains one external stream per selected column and broadcasts each
memory-tile object to all four compute rows. External `wire_bytes` do not change;
`fanout=4`, `logical_semantic_bytes`, and `logical_semantic_gbs` expose reuse
without pretending those logical bytes crossed LPDDR.

```bash
R57_MODE=MEMTILE_BROADCAST benchmarks/npu_gemm_tuning/r57/run_r57.sh
```

The default output is
`benchmarks/npu_gemm_tuning/results/r57-memtile-broadcast-20260712.csv`.

## Step 4: `PRECONVERTED_LAYOUT`

This mode requires the exact four-column profile and an existing
`.rdna2.hfp` file. It validates the fixed R34 header and payload SHA-256, then
feeds the 8,192,000-byte payload through the unchanged step-3 broadcast graph.
There is no tensor-block conversion in Python, DMA, memory-tile, or AIE code.

```bash
R57_MODE=PRECONVERTED_LAYOUT \
R57_COLUMNS=4 \
R57_LAYOUT_FILE="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.rdna2.hfp" \
benchmarks/npu_gemm_tuning/r57/run_r57.sh
```

The default output is
`benchmarks/npu_gemm_tuning/results/r57-preconverted-layout-20260712.csv`.

## Measured result

Locked AIE2P runs on `halo` used three fresh processes per point, a measured
1.8 GHz H clock, XRT 2.25.0, amdxdna 2.25.0, and firmware 1.1.2.65. Median exact
four-column results are:

| accumulated mode | wire GB/s | unique payload GB/s | logical GB/s | receive busy |
|---|---:|---:|---:|---:|
| `PRODUCTION_DMA` | 43.577 | 13.613 | 13.613 | 0.761 |
| `MEMTILE_STAGE` | 43.536 | 13.599 | 13.599 | 0.759 |
| `MEMTILE_BROADCAST` | 43.198 | 13.494 | 53.976 | 0.754 |
| `PRECONVERTED_LAYOUT` | 43.251 | 13.511 | 54.042 | 0.755 |

The real preconverted layout retains 99.25% of direct production DMA and 99.88%
of synthetic broadcast wire bandwidth. All accumulated stages clear the plan's
85% feed-preservation gate. The eight-column controls remain at 55.8-56.1 GB/s,
matching R56's external-feed roof.

The first broadcast graph was rejected by the router because four independent
guard drains plus trace traffic targeted the same southbound channel. Joining
the guards in the memory tile also failed because its packet-switched DMA source
collided with trace collection. The accepted graph keeps one traced checksum
drain per column and gives the other three consumers a tile-local volatile DCE
guard. The weight broadcast itself is unchanged.

The real OQ4 validation created 24 files originally named
`EmbeddingGemma-300M.npu.oq4.layer-N.rdna2.hfp`, each containing a 128-byte
version/hash header followed by the exact 8,192,000-byte R34 stream payload. A
second complete resident-model load preserved file size, mtime, and SHA-256,
proving the cache-hit path did not silently repack or rewrite it.

R58 introduced a packed-W4 version-2 ABI. New dense R34 files use the explicit
`layer-N.attention-dense.rdna2.hfp` role to avoid colliding with packed
projection files such as `layer-N.qkv.oq4.whole-scaled.rdna2.hfp`. The original
R57 files remain valid inputs for reproducing the committed R57 rows.

This first file-layout proof caches R34's current dense-W8/BF16 execution
payload produced from the OQ4 source. It therefore validates one-time tensor
block ordering, but it is **not** the future packed-OQ4 nibble format. Step 5
must add a new versioned `.rdna2.hfp` encoding that retains packed low/high
nibbles through DMA and performs only representation-local nibble/lane swizzle
in the AIE core.

Durable rows:

- `../results/r57-production-dma-20260712.csv`
- `../results/r57-memtile-stage-20260712.csv`
- `../results/r57-memtile-broadcast-20260712.csv`
- `../results/r57-preconverted-layout-20260712.csv`
