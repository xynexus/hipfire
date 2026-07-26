# R81: paired compact projection plus external Q/K/V pack

R81 fuses R80's pair-major compact-W4 projection with the exact R66 Q/K/V pack
phase. Odd cores project both adjacent stripes and then drain the stage
broadcast. Even cores drain the original projection activation phase before
owning Q/K/V packing, keeping every broadcast lock in phase while linking no
QKV projection functions on even cores.

This rung stops at the observable Q/KV boundary. Attention and output
projection remain absent until stage, Q, and KV match R65/R66 byte-for-byte.

```bash
./build_r81.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The layer-0 oracle matches all 327,680 projection-stage BF16 values and every
Q/K/V byte. Three fresh 100-command processes measure 1.8370, 1.8179, and
1.8397 ms (median 1.8370 ms). Maximum odd/even core text is 14,032/10,912
bytes. R81 is capacity-admitted as the exact paired projection/pack boundary,
but it is 40.5% slower than R70's 1.3076-ms single-context boundary.

Durable rows: `../results/r81-paired-w4-projection-pack-20260713.csv`.
