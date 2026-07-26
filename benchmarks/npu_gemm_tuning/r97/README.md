# R97: inline canonical-BF16 resident W4 FFN

R97 replaces R25's externally materialized packed activation argument with
canonical BF16 pre-FFN-normalized rows. Multidimensional DMA gathers three rows
and one K group per core without host tensor reordering. Each core converts
BF16 to F32, applies the existing AWQ/FWHT/row quantizer, and enters the native
W4 gate/up, GeGLU, and down stages. Immutable weight block order remains an
offline/loader `.rdna2.hfp` contract; only OQ4 nibble/lane swizzle is local to
compute.

The first complete image corrupted its own down accumulator. R25 spills the
partial down result between gate N blocks through `saved`, `own`, and
`transit`; R97 had reused `own` and `transit` for the newly inlined gate
fragment exchange before restoring that partial. Two dedicated 784-byte gate
fragment buffers remove the alias. This is an R97 kernel state-lifetime bug and
fix, not the platform workaround and not an LDS-avoidance rule.

After the fix, the full hardware oracle reports gate cosine `1.00000000`, final
cosine `0.99998228`, maximum absolute error `0.2597733`, and mean absolute error
`0.03750710`, with no NaNs. The remaining difference from R95 is the admitted
native AIE activation preparation: one quantized value differs by one code and
scales differ only by normal floating-point rounding. Maximum core text is
15,456 bytes.

A fresh correctness dispatch measured 6.4095 ms, or 39,941 M256 rows/s, but
this is not sustained admission. R97 inherits both the added kernel parameter
that stops the platform issue and R15's required numerical settings
(`rounding=floor`, `saturation=none`). These are separate controls; neither is
an LDS-use or LDS-avoidance rule. A 20-command control still encountered the
separate known four-second command timeout cadence. Fresh command objects and
context recycling are timeout diagnostics or mitigations, not substitutes for
the kernel parameter and not LDS rules.

Bounded recycling is sufficient for sustained execution. Three independent
100-command runs with `HIPFIRE_R25_RECYCLE_EVERY=7` preserve the same full
oracle and measure 6.4974, 6.4844, and 6.3388 ms per command (median 6.4844 ms,
39,479 M256 rows/s). Admit sustained standalone R97 with that timeout
mitigation. This is a complete-FFN row-rate result, not full-layer or
end-to-end encoder input-token throughput.

Durable row: `results/r97-inline-canonical-gate-20260713.csv`.
