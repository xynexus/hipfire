# R123 — core-stationary resident FFN weights

R123 tests whether one resident dense-W8 FFN dispatch can hold each 15,552-byte
weight record in the compute core while streaming all row macros through it.
The core keeps one f32 accumulator per M96 macro. M512 uses two three-macro
accumulator buffers to avoid a single 27 KiB cross-bank object. Output DMA uses
a rolling three-task window; queuing all six gate or twelve down outputs caused
an all-zero M512 result even though the image compiled.

The host uploads one 28-record gate/down sequence per column (3,483,648 bytes
total) rather than duplicating it per row macro. `r123_cache.sh 1` builds M256;
`r123_cache.sh 2` builds M512.

Hardware correctness passes both the absolute M256 oracle and the duplicated-
document M512 gate: both M512 documents are bit-exact with M256. Performance is
rejected, however: stationary M256/M512 measured 20.111/33.500 ms (1.20x row
throughput), while the existing replicated-weight path measured about
10.520/19.929 ms. The saved weight traffic does not repay the weight-major
control, accumulator, and activation-routing overhead.

Earlier object-FIFO attempts are also rejected: `iter_count` lowered to a
memtile repeat but delivered only the first sequence on hardware because the
source lock was produced once; `repeat_count` expanded BDs and exhausted the
24-BD memtile limit.

Durable rows: `../results/r123-weight-stationary-m-scaling-20260715.csv`.
