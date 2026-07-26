# R119: repeated output task for staged N64

R119 keeps R118 compute and activation staging byte-identical. It replaces two
explicit N32 output tasks with one descriptor carrying an outer tiling dimension
of two and a task `repeat_count` of one. This tests the scalable S2MM schedule
needed before increasing the output-block count toward N1280.

The added kernel parameter remains the platform workaround. This DMA scheduling
experiment is independent of LDS placement and of the 64-byte activation-stage
alignment fix.

Hardware parity is zero mismatches with `5e-9` maximum error. Eight passing
fresh 1,000-dispatch processes average 0.102308 ms, 3.54% below R118's two-task
mean. Two other contexts return the known all-zero result. The combined outer
dimension plus `repeat_count` schedule is admitted for scaling the N32 block
count, with the same context-stability caveat.

Durable rows: `../results/r119-repeat-output-task-20260713.csv`.
