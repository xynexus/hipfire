# R76: three-group attention task window

R76 is the only untested scheduling point between R75's exact, admitted
two-group window and the four-group window that links but corrupts Q. It keeps
R71's kernels, tile mapping, buffers, traffic, and byte oracle unchanged while
starting three groups of ordered Q, KV, and output tasks before await/free.

```bash
./build_r76.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

Projection stage, Q, KV, and final attention match the R70/R27 references
byte-for-byte. Three fresh primed 100-command processes measure 3.4199, 3.2222,
and 3.2604 ms (median 3.2604 ms).

R76 passes the speed ratchet by 0.52% against R75's 3.2775-ms median and 1.55%
against R71's 3.3118 ms. Three groups is the maximum correct task window for
this graph: four groups link but corrupt Q, while six groups exhaust BD IDs.
R76 is the admitted schedule to carry into resident-weight integration.

Durable rows: `../results/r76-attention-window3-20260713.csv`.
