# R77: destination-context resident R76 HFP weights

R77 replaces R76's benchmark-only extracted `weights.bin` with the real
layer-0 QKV `.rdna2.hfp`. `NpuEmbeddingQkvAttentionOpus` validates the HFP v2
descriptor, payload length, and payload SHA-256, allocates the weight BO from
the destination R76 hardware context, uploads it once, and reuses it across
commands. Mutable activations, stage/parameter tails, Q, and KV retain the R76
five-argument ABI and exact byte oracle.

The 192-byte HFP header is loader metadata; its 2,359,296-byte nibble-packed
payload is already in the R76 schedule order. No tensor-block conversion occurs
in the kernel or dispatch path. Local nibble decode and lane swizzle remain in
the AIE kernel. The added kernel parameter remains the separate correctness
workaround.

## Result

Projection stage, Q, KV, and attention match the raw-payload R70/R27 references
byte-for-byte. Three fresh primed 100-command processes measure 3.2753, 3.3165,
and 3.2137 ms (median 3.2753 ms). This is 0.46% above raw R76's 3.2604-ms
median and 1.1% below R71's 3.3118 ms; resident upload is outside the timed
loop and does not materially change dispatch latency.

R77 proves resident QKV/attention weights only. Output projection,
residual/norm tails, FFN, next-layer handoff, full-model tokens/s, and package
tokens/J remain outside this measurement.

Durable rows: `../results/r77-resident-hfp-r76-20260713.csv`.
