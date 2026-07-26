# R117: direct R113 compact full-K consumer, N32

R117 doubles R116's useful output width without increasing activation traffic.
Each K-tile activation load feeds four 8-column MMUL halves and produces one
N32 tile. All three K256 groups use the admitted prior-output staging fix.

The graph reads the same 589,824-byte padded R113 ABI with 199,680 unique chunk
bytes and materializes zero N-macro activation replicas. Its immutable N32
weight records are created offline/by the loader under the `.rdna2.hfp` layout;
the kernel performs no tensor-block reorder.

The added kernel parameter remains the platform workaround that stops the
platform issue. It is not LDS avoidance. R117's 1,024-byte prior-output staging
array is a separate local data-dependency mechanism.

The image builds at 3,192 bytes maximum core text. Hardware parity is zero
mismatches with `3e-9` maximum absolute error across both N16 halves. Eight
passing fresh 1,000-dispatch processes average 0.086916 ms, 9.82% faster than
R116's 0.096384 ms passing mean despite doing twice the useful N work. Two
other fresh contexts returned all-zero output before eight passes; retain the
same context-stability caveat.

R117 admits N32 activation-load reuse, not a context-stable runtime default or
full N1280 projection. The next topology stages all three compact chunks once
per core and streams multiple N32 weight/output records without replaying
activation DMA.

Durable rows: `../results/r117-direct-compact-fullk-n32-20260713.csv`.
