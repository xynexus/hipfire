# R70: single-context W4 projection and QKV pack

R70 fuses the admitted compact-W4 projection and inline headnorm/RoPE pack
phases into one AIE graph and one hardware context. The first attempt combined
R68 and R67 literally, but exceeded the compute tile's two input DMA channels.
The admitted build therefore reuses the projection activation FIFO for packing:
8-KiB activation records are padded once by the loader to the existing 10-KiB
inline record width, then the R65 projection and R66 pack phases execute in one
runtime sequence and reuse both input and output channels sequentially.

This is the direct response to R69: two full-array contexts were intermittent
and inflated both subcommands to about 2.7-2.9 ms. UG1079 documents graph-owned
streams or DMA buffers plus locks as the ordered producer/consumer mechanisms;
it does not define a cross-context SHMEM cache fence.

R70 performs no online immutable tensor-block conversion. Compact `.rdna2.hfp`
weights retain their offline order, activation padding is a loader operation,
and OQ4 nibble/lane handling remains local to the existing R15 projection
function. The inline pack schedule is a correctness-first single-context rung;
later optimization can recover R68's joined-input concurrency without adding a
third input channel.

```bash
./build_r70.sh
```

Acquire `hipfire lock` before compiling/running hardware artifacts.

## Result

The literal R68+R67 merge was rejected because four logical inputs exceeded
the tile's two input DMA channels. A first inline merge then exceeded 16 KiB of
program memory. R70 closes both constraints by reusing the activation channel,
splitting K and V ownership across columns, and replacing duplicate W4 init and
accumulate functions with one generic group function.

Three fresh primed 100-command processes pass the isolated R65 projection stage
byte-for-byte and the isolated R66 Q/KV outputs byte-for-byte. Times are 1.3076,
1.3108, and 1.3006 ms (median 1.3076 ms). Largest core text is 13,504 bytes.
As with the resident R26/R34 family, the first command is a discarded prime;
unprimed first-command Q output was intermittently incomplete.

This is about four times faster than R69's cross-context chain and about 12%
faster than the isolated R65+R66 median sum. It remains slower than R68's
isolated joined-stage sum, so the next graph should add attention in the same
context before optimizing joined-input concurrency. This timing is a QKV
projection/pack boundary, not full-model embedding throughput.

Durable rows: `../results/r70-single-context-projection-pack-20260713.csv`.
