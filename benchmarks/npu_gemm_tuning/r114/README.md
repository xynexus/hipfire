# R114: rejected in-context R34 activation-prefix assembly

R114 tested whether R113's correct per-core diagnostic chunks could be assembled
into compact 24-token records inside the tail context. Runtime DMA assigns
canonical contiguous eight-token ranges to adjacent three-core shapes (and one
final two-core shape). Neighboring cores exchange chunks through shared
data-memory ObjectFIFOs and locks, not the already saturated stream switch. Each
packer emits split Q and scale planes through the existing completed-output
route. The compact 589,824-byte ABI materializes no 16 KiB padding and no
fivefold N-macro replicas. This changes dynamic activation scheduling only; it
does not reorder immutable tensor blocks.

Three rejected precursors are part of the result: logical-owner stream chains,
physical column-major stream chains, and neighbor-memory assembly with a new
shim output route all failed with `Unable to find a legal routing`. An attempt
to reuse the completed-output task with a zero destination stride was rejected
because DMA strides must be positive. Reusing the existing completed-output
route for split compact planes builds in about nine seconds, with maximum core
text of 11,200 bytes.

The build is not hardware-correct. A good tail dispatch still reaches the R113
tail oracle, but the compact pack reports 107,811 byte mismatches, maximum Q
delta 254, and maximum scale error 0.034057196. Errors are distributed across
all three K256 groups and all local-memory owner positions, so the compact
assembly/mapping remains incorrect; the evidence does not isolate the defect to
one predecessor chunk. R114 is rejected and must not be selected by runtime.

The useful design result is the compact boundary, not this implementation.
R113 already exposes correct per-core chunks in a 589,824-byte diagnostic ABI
(199,680 unique chunk bytes). The next resident R34 GEMM should consume those
chunks directly and reuse each one across five N-macros. It should not assemble
or materialize the canonical 2,949,120-byte replicated activation tensor. The
tail continues to return canonical completed BF16x2 rows, and immutable
`.rdna2.hfp` tensor blocks remain unchanged.

One fresh context also returned an all-zero completed tail before an immediate
repeat returned the distributed pack mismatch. Keep that as separate
context-transition evidence. The added kernel parameter is the platform
workaround that stops the platform issue. It is not LDS avoidance and is
independent of local-memory placement, R111 alignment, core-chain routing,
output mapping, and context lifetime.
