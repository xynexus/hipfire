# R71: single-context W4 projection, QKV pack, and attention

R71 extends R70 with the established R30 BF16 bidirectional attention phase in
the same AIE2P graph and hardware context. Columns 4-7 execute attention after
their projection roles by reusing the completed 16-KiB weight FIFOs. Their Q/V
pack ownership moves to columns 0-3, which already link the Q/K pack functions.
Each attention column processes one token-row shard while its four core rows
own the four head pairs; the output DMA transposes those fragments back to the
canonical layout. No third compute-tile input channel or cross-context SHMEM
handoff is involved.

The DPU ABI remains at five data arguments. The 393,216-byte attention result
is appended to the existing 2,457,600-byte staging/result BO. Q and KV remain
separate observable arguments for correctness isolation at this rung. Compact
OQ4 weights retain their offline `.rdna2.hfp` order, and only local nibble/lane
work occurs in the projection kernel.

The kernel parameter that prevents the platform issue remains a separate
correctness requirement. LDS use or avoidance is not part of this R71
admission decision and remains an independently measured optimization choice.

```bash
./build_r71.sh
./build_r71_pack_control.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The first literal extension exceeded the compute tile's two input DMA
channels. Reusing the 16-KiB weight FIFO removed that channel, but linking
attention on columns 4-7 initially exceeded 16 KiB of program memory. Moving
all Q/V pack ownership to columns 0-3 leaves columns 4-7 below the limit;
maximum core text is 15,888 bytes.

The full-width graph matches the isolated R70 stage, Q, KV, and R27 attention
outputs byte-for-byte. Three fresh primed 100-command runs measure 3.5951,
3.2617, and 3.3118 ms (median 3.3118 ms). The redistributed pack-only control
is also exact at a 1.5446-ms median, while isolated R27 attention is 0.9141 ms.
The fused attention increment is therefore about 1.77 ms and is not admitted
for resident integration yet.

A two-column/two-row-wave fallback was exact but rejected at 5.8228 ms median.
Separating attention input and output onto different shim columns was rejected
at resource allocation because the additional memory-tile input path exceeded
the hardware channel limit. The next rung must reuse an existing memory-tile
path or eliminate the external packed-Q/KV round trip.

Durable rows: `../results/r71-single-context-projection-pack-attention-20260713.csv`.
