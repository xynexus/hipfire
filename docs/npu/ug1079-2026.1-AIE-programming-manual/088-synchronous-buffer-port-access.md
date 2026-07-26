---
title: "Synchronous Buffer Port Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Synchronous-Buffer-Port-Access"
toc_id: 6jUs0VPTRimt29gTXEvf_A
content_id: iGRMsKaSGgA~lul28kxtbA
---

### Synchronous Buffer Port Access

A kernel reads from its input buffers and writes to its output buffers. By default, the synchronization that is required to wait for an input buffer of data is performed before entering the kernel. The synchronization that is required to provide an empty output buffer is also performed before entering the kernel. There is no synchronization needed for synchronous buffer ports within the kernel to read or write the individual samples of data after the kernel has started execution.

You can declared buffer port size using the `dimensions()` API or with kernel function prototype.

- **Option 1:** Configure with `dimensions()` API in graph.`connect netN(in.out[0], k.in[0]); dimensions(k.in[0])={INPUT_SAMPLE_SIZE};`
- **Option 2:** Configure using a kernel function prototype declared in the kernel header file referenced in the graph code.Graph code specifies connection. `connect netN(in.out[0], k.in[0]);` Kernel code specifies data type and buffer size. `void simple(input_buffer<int32, adf::extents<INPUT_SAMPLE_SIZE>> & in, output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out);`

In the following example:

- The kernel located in tile 1 uses a ping-pong buffer for writing
- The kernel located in tile 2, which is adjacent to tile 1, uses the same ping-pong buffer for reading

kernel 1

kernel 2

kernel 1

kernel 2

![agw1690984597227.png](../assets/088-01-agw1690984597227-png-208e6307cec5.png)

*Figure 1. Lock Mechanism For Synchronous Ping-pong Buffer Access*

The tiles `main` function handles the kernel's buffer lock mechanism. The kernel starts only when all input and output buffers are locked for reading and writing respectively. The minimum latency for a lock acquisition is seven clock cycles if the buffer is ready to be acquired. If another kernel locks the buffer, it stalls until it becomes available (shown in red in the preceding figure).

You can see in the diagram that the lock acquisition occurs alternatively on the ping then pong buffer. Ping or pong buffer selection is automatic.

When a synchronous buffer port is used, the lock on the output buffer of the kernel is released after the kernel execution has finished. The consumer kernel can then acquire the buffer, or DMA can transfer it to its destination (such as a PLIO).

**Note:** Important:
