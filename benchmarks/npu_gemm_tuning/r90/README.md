# R90: local post-attention residual and RMS norms

R90 extends R89's local BF16 O staging with the complete post-attention tail:
post-attention RMS normalization, residual addition, and pre-FFN RMS
normalization. Each even core reuses 8 KiB of the final dead activation FIFO
object plus a 4 KiB tail as three block-aligned 8x256 BF16 tiles. Parameters
arrive as four R34-compatible 16 KiB records per active column and wave through
the existing O-weight FIFO; no sixth data argument or external tensor round
trip is added.

The initial 18-record projection drain uses its existing loop with a noinline
runtime bound to prevent Peano from expanding it back into 18 identical
acquire/release pairs. Q/K/V packing retains the byte-verified R89 order; the
rejected Q-pack loop rewrite is not used. Offline weight order, DMA-only
canonical scatter, and the independent kernel-parameter correctness workaround
are unchanged.

```bash
./build_r90.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

The first literal R90 image reached 16,784 bytes on an even core. A source-level
Q-pack loop linked at 14,976 bytes but changed the observable attention result
and was rejected. The admitted compaction changes only R81's pre-existing
18-record projection-drain loop from a compile-time to a noinline runtime bound;
all 12 Q-pack calls per even core and their original ordering remain intact.
Maximum even/odd text is 15,952/16,048 bytes. The shared build now rejects a
missing `final.xclbin` or `insts.bin` and any core above the 16 KiB program
limit, closing `aiecc`'s silent-success failure mode after CDO overflow.

The hardware oracle passes projection stage and KV byte-for-byte. Across all
196,608 BF16 tail values it reports global cosine 0.99995399, minimum token-row
cosine 0.99994058, maximum absolute error 0.09375, no non-finite values, and no
zeros. The tail gate therefore requires both global and minimum-row cosine at
least 0.9998 and maximum error at most 0.1. The tight row cosine distinguishes
the bounded row-scale error from the two AIE reciprocal-square-root operations
from a layout or parameter-record mismatch.

Three fresh 100-command processes measure 6.6044, 6.2915, and 6.3890 ms
(median 6.3890 ms), 0.6688 ms above R89. Admit R90 as the complete local
post-attention norm/residual/pre-FFN-norm correctness boundary, not as a speed
win or a full-layer result. The next rung must feed this resident normalized
activation directly into FFN without a host tensor round trip. The added kernel
parameter remains the independent platform-issue workaround; LDS use remains a
capacity/performance choice.

Durable rows: `../results/r90-residual-norm-tail-20260713.csv`.
