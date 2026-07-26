# R56 — XDNA2 external-memory and cache-path characterization

This round extends R1's feed ceiling with cache-capacity and contention
controls. The primary metric remains the compute-tile receive-port event trace:
host allocation, JIT, BO synchronization, and dispatch setup are outside the
measured interval.

Raw results are in
[`../results/r56-feed-cache-20260712.csv`](../results/r56-feed-cache-20260712.csv).
The complete interpretation and manual audit live in
[`../../../docs/npu/npu-memory-bandwidth-cache-characterization.md`](../../../docs/npu/npu-memory-bandwidth-cache-characterization.md).

## Reproduction

Use the environment setup in `../r1/run_r1b.sh`, acquire the Hipfire lock, and
run `r1b_trace_run.py` or `r1b_cols_trace_run.py`. The cache-capacity sweep uses
eight columns reading the same region while varying `N_TILES`:

```bash
for nt in 16 64 256 1024 4096 8192 16384; do
  COLS=8 TRACE_COLS=1 DISTINCT=0 TILE_N=4096 N_TILES="$nt" \
    TRACE_SIZE=4194304 HCLK_MHZ=1800 python ../r1/r1b_cols_trace_run.py
done
```

Build the contention controls with:

```bash
c++ -O3 -std=c++20 -pthread cpu_dram_pressure.cc -o cpu_dram_pressure
hipcc -O3 --offload-arch=gfx1151 gpu_cache_pressure.hip -o gpu_cache_pressure
```

`gpu_cache_pressure 8 20` uses 16 MiB across its source and destination, below
the reported 32 MiB GPU MALL capacity. `gpu_cache_pressure 256 20` uses 512 MiB
and therefore exercises the external-memory path. Run the NPU trace during each
control and compare with an idle baseline. These controls are interference
tests, not direct cache-hit counters; shared power, clock, and fabric effects
must be reported as confounders.
