# R22 row-striped pack + direct-stream all-gather + W4 down

R22 parallelizes R21's serial 24-row activation pack across all eight compute
columns. Each core packs three rows with the exact R20 vector contract, keeps
its own 784-byte fragment, and reconstructs the complete 8 KiB activation
block through direct core streams before running the resident scaled W4 down
MMUL.

The stream topology is a physical ring, but the communication schedule does
not perform a closed ring exchange. A closed blocking ring stalls because the
AIE core streams have no cycle-breaking buffer. Instead, R22 performs eight
acyclic token broadcasts. For each owner, the fragment travels seven hops and
the predecessor receives without forwarding, deliberately leaving the cycle
open. Two adjacent columns share a six-row input FIFO so the memory tile uses
four activation outputs plus one weight output, within its DMA-channel limit.

```sh
bash benchmarks/npu_gemm_tuning/r22/r22_cache.sh
cargo run -p hipfire-xdna --release --example npu_pack_down_verify -- \
  ~/.hipfire/npu/embgemma_aie2p_ring_pack_down_w4_m256_k1152_n768 100
```

Three independent 100-iteration hardware runs report zero mismatches across
196,608 outputs, maximum absolute error `1.4e-6`, and dispatches of 0.9967,
0.9974, and 0.9743 ms (median 0.9967 ms). This is 2.59 times faster than R21's
2.5797 ms median. It remains a W4 combined pack/down projection: the group
input stream is host-arranged rather than fed directly from R18, and W8,
arbitrary mixed overlays, the rest of the FFN/model, and energy admission are
still open.
