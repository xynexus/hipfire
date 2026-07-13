# R104: direct-X inline-normalized native-W4 FFN

R104 removes the host pre-FFN normalization bridge without adding a DMA route.
It prepasses the existing canonical direct-X input once per M block to compute
three BF16-row inverse RMS values, then reuses R99's input stream for gate/up.
The loader folds the immutable pre-FFN norm into the existing AWQ divisor, so
`X * inverse / (AWQ / pre_norm)` matches the R99 normalized-H quantization
boundary. The output remains R99's interleaved BF16x2 combined-row ABI and R100
continues to consume the same unchanged canonical X buffer.

This design follows the rejected R101/R102 row-state experiments. A separate
metadata FIFO exceeded the available DMA channels; the in-record alternative
either exceeded program memory or failed the hardware state/timeout oracle.

R104 became admissible after four source-level capacity changes:

- vectorize the `1 / 768` mean multiply so no scalar `__mulsf3` helper is linked;
- fuse inverse completion into the RMS scan;
- share one runtime-stride FWHT helper instead of four cloned templates; and
- hold one canonical `3 x 768` BF16 X object per core, scan it once, and select
  the K group at conversion time.

The last change also makes the physical input traffic match the logical
contract: one 442,368-byte padded canonical-X transfer, rather than a prepass
plus nine input replays. The normal `aiecc` build at `-O2` succeeds with all 32
core text sections exactly 16,384 bytes. The packaged artifact records
`input-dma=single-full-row-object` and `rms-epsilon=1e-6`.

The standalone hardware oracle reaches gate cosine `1.00000000`, final cosine
`0.99996707`, maximum absolute error `0.0737100`, and mean absolute error
`0.01499078`. A 100-command run with context recycling every seven commands
preserves the oracle at 6.5401 ms per M256 dispatch. The default layer-0 path
reaches FFN cosine `0.99991494`, tail cosine `0.99999844`, and completed-layer
cosine `0.99996658`. A default 24-layer run completes in 894.222 ms, or 286.3
input tok/s at 18.07 W and 15.8 tok/J. Paired R99/R104 trials show a 4.1%
latency improvement despite run-to-run package variance, so R104 is admitted as
the native-W4 default when its artifact is present.

Post-link Peano optimization produced smaller 15,824-15,840-byte images but
timed out and returned all-zero output on hardware; those compiler variants are
rejected. The added kernel parameter remains the independent platform
workaround. LDS/tile-memory placement, R97 fragment lifetime, and bounded
context recycling remain separate decisions or fixes.
