# R92: same-DRM peer-context control

R92 keeps R91's shifted-stage GTT handoff and unchanged AIE images, but loads
the canonical-BF16 R35 FFN with `NpuKernel::load_peer`. Producer and consumer
therefore share one amdxdna DRM file and device heap while retaining separate
hardware contexts. This isolates separate file/heap ownership from the physical
context switch.

An attempted PRIME export of the producer's XDNA SHMEM BO is rejected by this
driver with `EINVAL`, so the accepted control retains R91's proven GTT dma-buf.
The buffer type and all kernel/weight/layout contracts are otherwise unchanged.

## Result

The full R91 tail and FFN oracles pass: exact KV, admitted residual/norm output,
FFN cosine 0.99989925, and FFN maximum error 0.0118408. Three fresh 100-command
processes measure 6.4109 ms producer, 9.7080 ms isolated FFN, and 22.1542 ms
alternating-chain medians. The remaining 6.0353 ms is 27.2% of the chain and is
statistically unchanged from R91's 6.0391 ms.

Reject same-DRM peer ownership as a sustained scheduling optimization. The
one-iteration 19.8480-ms result was a cold/sample artifact and is not used for
admission. The 11,555 M256 rows/s figure applies only to one layer boundary.
The context tax is intrinsic to alternating hardware contexts on this stack;
the next performance work must change phase partitioning or reduce the native
FFN phase, not change buffer ownership.

Durable rows: `../results/r92-peer-context-control-20260713.csv`.
