# R72: graph-local Q handoff

R72 removes the first half of R71's external Q/KV round trip. Columns 0-3 pack
each query-head pair and send it over scalar core-to-core streams. Columns 4-7
cache the six query groups in the projection accumulator storage and consume
them during attention. Core streams do not allocate a third input DMA channel;
KV remains on R71's observable external path for this isolated bandwidth-first
rung.

The Q BO remains in the five-argument ABI but is intentionally unused by the
direct-Q graph. Projection stage, KV, and final attention remain externally
observable. Exact final attention parity against R71 is the functional gate.

Immutable weight ordering remains offline in `.rdna2.hfp`. The existing kernel
parameter is the correctness workaround. This experiment neither replaces that
parameter nor treats LDS/tile-memory avoidance as a correctness mechanism;
local-memory choices remain performance variables.

```bash
./build_r72.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The graph builds after unifying the 24-KiB query cache with the projection
accumulator and falling back from bank-aware to sequential tile-memory
allocation. Maximum linked core text is 15,248 bytes. Projection stage, KV, and
final attention match the isolated R70/R27 references byte-for-byte; the unused
external Q BO confirms that attention functionally consumes the streamed Q.

The direct-Q route is rejected for performance. Three fresh primed 100-command
runs measure 3.9288, 3.7749, and 3.9272 ms (median 3.9272 ms), 18.6% slower than
R71's 3.3118-ms median. Removing 393,216 external Q bytes did not offset scalar
stream synchronization and query-cache pressure. Do not extend this scalar
stream topology to K/V. The next R72 sub-rung should preserve burst/vector DMA
or reuse an existing graph-local FIFO while eliminating a boundary.

Durable rows: `../results/r72-direct-q-stream-20260713.csv`.
