# R120: activation-once full-K N128

R120 parameterizes R119's admitted output schedule to four N32 blocks. Each
core still copies the three R113 activation chunks once, releases the input
FIFO, and reuses the 6,336-byte aligned local stage across every output block.
Activation DMA therefore remains one 589,824-byte pass containing 199,680
unique bytes; widening N does not materialize activation replicas.

One output task per stream uses an outer tiling dimension of four and
`repeat_count=3`. Output row strides scale with N while each produced N32
object remains 1,024 bytes per core. Immutable weight records are prepared
offline in `.rdna2.hfp` order; the kernel does not reorder tensor blocks.

The added kernel parameter remains the platform workaround that stops the
platform issue. The aligned local activation stage and repeated DMA schedule
are independent of that workaround and do not establish an LDS-avoidance rule.

Hardware parity passes all four blocks with zero mismatches and `7e-9` maximum
absolute error. Four passing fresh 1,000-dispatch contexts average 0.115102 ms;
six other fresh contexts return the known whole-output zero symptom. Admit the
N128 math and schedule, not context-stable production selection. Maximum core
text is 4,312 bytes. Durable rows:
`../results/r120-staged-fullk-n128-20260713.csv`.
