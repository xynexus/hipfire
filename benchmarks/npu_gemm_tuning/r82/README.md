# R82: paired compact projection, pack, and attention

R82 appends unchanged R76 attention to R81's paired odd-core projection and
even-core Q/K/V packing. Odd cores reuse their projection weight FIFO for the
observable Q/KV attention input and retain the admitted three-group task
window. This is first a 16-KiB program-capacity test; hardware correctness is
required only if the graph links.

```bash
./build_r82.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

R82 is rejected at link/CDO generation before hardware execution. Odd columns
1/3 require 22,416 bytes of program text, columns 5 require 21,936 bytes, and
columns 7 require 20,592 bytes. The worst image exceeds the physical 16 KiB
program store by exactly 6,032 bytes; therefore R82 makes no correctness or
timing claim.

Symbol attribution shows that attention adds 8,384 bytes to the R81 image:
2,992 bytes of core-driver expansion, 4,256 bytes of attention functions, and
1,136 bytes of soft-float helpers. This is program capacity, not tile-memory
capacity. The separate kernel parameter remains the correctness workaround;
LDS/tile-memory avoidance is not implicated.

Durable rows: `../results/r82-program-capacity-20260713.csv`.
