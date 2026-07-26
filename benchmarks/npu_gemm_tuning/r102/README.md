# R102: direct-X row-state native-W4 FFN

R102 consumes R101's canonical direct-X rows and their per-token inverse RMS.
The first vector conversion multiplies X by the inverse; the loader folds the
immutable pre-FFN norm into each layer's existing gate/up AWQ divisor. R25 then
receives the same normalized activation values as R99 without host readback,
normalization, or rewriting the shared input.

The output remains R99's 1,152-word combined-row interleaved BF16x2 ABI. The
R15 `rounding=floor` and `saturation=none` numerical controls remain enabled.
The separately added kernel parameter remains the platform-issue workaround;
LDS placement is independent.

The first depth-two build emitted a bank-allocation warning because the new
4,992-byte three-row object competed with two 15,872-byte weight objects.
Depth one removes that warning and builds at 16,064 bytes. R102 is nevertheless
not admitted or selected because neither tested R101 producer delivered a
correct, sustained row-state stream. This is a producer-boundary rejection,
not an R102 numerical admission.
