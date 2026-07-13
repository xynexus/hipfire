# R73: adjacent-tile Q ObjectFIFO

R73 replaces R72's scalar per-word stream with one depth-one, 24-KiB
ObjectFIFO per adjacent producer/consumer pair. Even columns pack all six Q
groups into shared tile memory; neighboring odd columns acquire that buffer
once and run attention. The external Q BO remains unused, while KV retains
R71's externally observable path.

The existing kernel parameter is the correctness workaround. Shared tile
memory is deliberately used here and is judged only by correctness, capacity,
and measured performance; LDS/tile-memory avoidance is not a correctness rule.

```bash
./build_r73.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Build findings

The first producer-local cache variant exceeded the 16-KiB core-program limit
at 16,880-16,896 bytes. The adjacent ObjectFIFO graph then exceeded tile-memory
capacity with a 4-KiB producer stack. Reducing that stack to 2 KiB made the
same topology fit; producer core text is at most 14,912 bytes and consumer core
text is at most 14,352 bytes.

These are implementation-capacity findings, not a reason to avoid shared tile
memory.

## Result

Projection stage, external KV, and final attention match the isolated R70/R27
references byte-for-byte, and the Q BO is unused. Three fresh primed
100-command processes measure 3.6449, 3.7165, and 3.7205 ms (median 3.7165 ms).

R73 is rejected by the bandwidth-first speed ratchet. It is 5.4% faster than
R72's 3.9272-ms scalar-stream median, showing that coarse FIFO handoff recovers
some synchronization cost, but remains 12.2% slower than R71's 3.3118-ms
external-Q baseline. Do not extend this exact schedule to K/V. The next handoff
must overlap producer and consumer work or reuse an existing DMA path without
serializing all six Q groups behind a single depth-one buffer.

Durable rows: `../results/r73-adjacent-q-objectfifo-20260713.csv`.
