# R106: unit-RMS native-W4 consumer

R106 is byte-identical to the admitted R99 compute graph. Its manifest changes
the mutable input contract from fully learned-normalized H to R105's canonical
unit-RMS BF16. The loader folds immutable pre-FFN norm into the existing W4
activation divisor, preserving R99's quantization and output ABI without a new
kernel input or tensor reorder.

R105 and R106 are selected only as a pair. R15's rounding/saturation numerical
controls remain enabled. The separately added kernel parameter remains the
platform-issue workaround, independent of LDS placement.

## Integrated result

The pair is numerically valid. Layer 0 reports unit-RMS cosine `0.99999269`,
FFN cosine `0.99990930`, tail cosine `0.99999862`, and completed-layer cosine
`0.99996179`. A complete 24-layer M256 run reaches 901.432 ms, 284.0 input
tok/s, about 21 W package power, and 13.5 tok/J.

This regresses the admitted R99/R100 bridge baseline of 878.003 ms and 291.6
tok/s. The isolated R105 median is only 0.1295 ms, but cross-context cache
maintenance costs 2.35-4.14 ms per layer and preparation/output still remains
roughly 9-12 ms. R105/R106 is therefore rejected as an automatic selection and
is available only with `HIPFIRE_EMBED_UNIT_RMS_BRIDGE=1` (or an explicit R106
cache override). The next rung must fuse RMS preparation into the resident W4
context instead of adding another context boundary. Durable comparison rows
are in `../results/r106-unit-rms-layer-integration-20260713.csv`.
