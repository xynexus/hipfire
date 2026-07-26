# R69: cross-context shared QKV handoff

R69 tests whether the admitted R68 W4 projection and R68 headnorm/RoPE pack
graphs can be chained through one 1,517,568-byte staging BO without a host copy.
The immutable Opus HFP order and local OQ4 nibble/lane handling are unchanged.

Three sharing mechanisms were tested on the installed amdxdna driver:

1. Importing one amdgpu GTT dma-buf into two independent XDNA contexts is not
   coherent after producer completion. The first unsynchronized run differed
   from the host-copy control in 1,531 Q bytes and 703 KV bytes. Producer BO
   sync, consumer BO sync, and dma-buf RW barriers did not repair the path.
2. PRIME-exporting an XDNA-owned SHMEM BO is unavailable. `strace` confirms
   `DRM_IOCTL_PRIME_HANDLE_TO_FD = -1 EINVAL` after successful SHMEM creation.
3. Duplicating one XDNA accel fd retains a shared DRM file description and GEM
   handle namespace. Peer contexts can submit the producer's original SHMEM
   handle directly. They must also allocate PDI/instruction BOs from the same
   device heap. A producer-side `SYNC_BO(TO_DEVICE)` then gives correct output
   in two of three fresh 100-command processes, but the third process failed the
   initial oracle by 384 Q bytes.

The two passing shared-DRM-file runs measured 5.4478 and 5.6600 ms per chain.
The cache-maintenance boundary itself was only 0.028-0.029 ms. Projection and
pack each inflated to roughly 2.7-2.9 ms versus their isolated R68 medians of
0.4946 and 0.3579 ms. Two full-array hardware contexts are therefore being
scheduled at a cost far larger than the kernels or the memory visibility sync.

Verdict: **reject the cross-context boundary**. It is intermittently incorrect
and its context scheduling destroys R68 throughput. R70 must keep projection
and headnorm/RoPE packing in one compiled AIE graph/hardware context so the
mutable handoff stays in graph-owned memory/FIFOs and no full-array context
switch occurs.

Inputs are generated under `~/.hipfire/npu/r69-chain-input/` by
`prepare_stage.py`; generated xclbins remain under `~/.hipfire/npu/`.
