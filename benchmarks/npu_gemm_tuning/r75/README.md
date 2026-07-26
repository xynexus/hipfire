# R75: two-group attention task window

R75 preserves R71's kernels, tile mapping, observable Q/KV boundary, and byte
oracle. It changes only the runtime command stream: two groups' ordered Q then
KV input tasks and corresponding output tasks are started before any task is
awaited. Three windows cover groups 0-1, 2-3, and 4-5.

An initial six-group window exhausted static BD IDs at group 4. A four-group
window linked, but failed hardware parity with 392,405 of 393,216 Q bytes wrong.
The two-group window is the next correctness ratchet, not an assumed success.

This tests whether the per-group await/free barrier leaves command-processor or
DMA scheduling bubbles. It adds no tile buffers and changes no weight or tensor
layout. The added kernel parameter remains the correctness workaround.

```bash
./build_r75.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

Projection stage, Q, KV, and final attention match the R70/R27 references
byte-for-byte. Three fresh primed 100-command processes measure 3.2580, 3.2775,
and 3.3314 ms (median 3.2775 ms).

R75 passes the speed ratchet by 1.0% against R71's 3.3118-ms median. The gain is
small but isolated: kernels, core images, tile buffers, external traffic, and
math are unchanged; only the six per-group completion barriers become three
two-group windows. R75 is the admitted projection/pack/attention baseline for
the next scheduling rung.

Durable rows: `../results/r75-attention-window2-20260713.csv`.
