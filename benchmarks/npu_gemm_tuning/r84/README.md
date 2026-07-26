# R84: paired W4 projection through direct O projection

R84 grafts the proven R32/R33 direct-output seam onto R83. Odd cores finish
both attention lanes into one depth-three 4 KiB FIFO to the adjacent even core;
even cores retain Q/K/V packing and add the BF16 M8-by-256 output projection.
No attention tensor crosses the host boundary.

The five data arguments remain `%A,%W,%R,%Q,%KV`. The loader-side weight image
is the unchanged paired-QKV prefix followed by canonical direct-O blocks; no
tensor-block reorder occurs in the graph or kernel. `%R` preserves R83's
attention gap and appends canonical token-major F32 O output for the first
correctness ratchet.

```bash
./build_r84.sh
```

Acquire `hipfire lock` before building or running the hardware artifact.

## Result

R84 links and packages. Bank-aware placement warns on the 4 KiB odd-core Q
scratch, then the basic sequential allocator places the complete image; the
hardware correctness run confirms that placement is functional. The existing
kernel-parameter correctness workaround remains enabled and is independent of
this tile-memory placement choice.

The first output oracle exposed a topology-specific DMA transpose: R83 assigns
tokens as `active_column * 32 + core_row * 8`, whereas R32's older direct-O
scatter assumed `active_column * 8 + core_row * 32`. R84 now scatters the FIFO
directly into the canonical token-major destination with the former mapping.
This is a DMA destination layout only; weights remain loader-preconverted and
no tensor-block reorder was added to the graph or kernels.

The corrected hardware verifier passes the projection stage and KV byte for
byte, leaves external Q unused, and passes the fused attention-to-O numerical
oracle. Three fresh 100-command processes measure 5.9362, 5.8716, and 5.9856
ms (median 5.9362 ms). R84 is admitted as a correctness/capacity rung and
speed-rejected: adding direct O is 46.4% slower than R83's 4.0535 ms median.

Durable rows: `../results/r84-direct-attention-output-20260713.csv`.

## Latency attribution

Three controls retain the same R83 projection/attention math and paired
attention finish. Each uses a 64 KiB completion signal so command timing cannot
finish before the even cores:

- attention handoff/drain: 4.2988, 4.0951, 4.0730 ms; median 4.0951 ms;
- full O-weight stream/drain without MMUL: 4.6455, 4.7160, 4.8099 ms;
  median 4.7160 ms;
- full O MMUL plus F32 finish into tile-local scratch: 5.7587, 5.8797,
  5.7761 ms; median 5.7761 ms.

Relative to R83's 4.0535 ms and full R84's 5.9362 ms, the 1.8827-ms
increment partitions into 0.0416 ms adjacent handoff, 0.6209 ms O-weight
delivery/consumption, 1.0601 ms O MMUL plus finish, and 0.1601 ms canonical
output DMA. Compute/finish is the largest term. The controls do not replace or
alter the separate kernel-parameter correctness workaround.

Durable attribution rows:
`../results/r84-direct-output-attribution-20260713.csv`.
