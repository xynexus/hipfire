# R78: odd-column R76 attention remap

R78 is a topology control before direct output projection. Even columns own
the existing Q/K/V pack functions and write the unchanged observable Q/KV
arguments. Adjacent odd columns run the unchanged R76 attention math and
three-group task window. No graph-local Q buffer or output-projection code is
present yet.

This preserves the R70/R27 byte oracle while establishing the odd→even
neighbor pairing required by the R32 output-projection handoff. Immutable
weights remain in offline `.rdna2.hfp` order; the added kernel parameter is the
separate correctness workaround.

```bash
./build_r78.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

Projection stage, Q, KV, and attention match the R70/R27 references
byte-for-byte. Odd attention cores link at no more than 15,888 bytes. Three
fresh primed 100-command processes measure 3.8331, 3.7729, and 3.7959 ms
(median 3.7959 ms).

R78 is rejected as a standalone scheduling result: it is 16.4% slower than
R76's 3.2604-ms median because even-only Q/K/V packing loses concurrency. It
also shows why R32 output projection cannot merely be appended: even cores
still carry compact-W4 projection and pack code, leaving too little of the
16-KiB program store for the output kernel.

The next capacity rung must replace those even-core projection routines with
R33-style paired projection on odd cores. Its pair-major compact-W4 block order
must be created once by the loader in a `.rdna2.hfp`; no kernel tensor-block
reorder is permitted.

Durable rows: `../results/r78-odd-attention-remap-20260713.csv`.
