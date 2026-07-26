---
title: "Buffer Allocation Control"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Allocation-Control"
toc_id: vrF0rfFdJbI9eQjj2DJbQg
content_id: _65OMzdiK_E6pdAGExIKpg
---

## Buffer Allocation Control

The AI Engine compiler allocates the desired number of buffers for each memory connection. There are several different cases.

- Lookup tables are always allocated as single buffers because they are expected to be read-only and private to a kernel. Synchronizing lookup table accesses does not require locks, because they are expected to be exclusive access.
- Buffer connections are usually assigned double buffers if the producer and consumer kernels are mapped to different processors or if the producer or the consumer is a DMA. This enables the two agents to operate in a pipelined manner using ping-pong synchronization with two locks. The AI Engine compiler generates this synchronization in the respective processor `main` functions.
- If the producer and consumer kernels are mapped to the same processor, the buffer connection is given only one buffer. No lock synchronization is required because the kernels execute sequentially.
- You can assign double buffers (default) to runtime parameter connections along with a selector word to choose which buffer to access next.

You can also assign single buffers to runtime parameter connections. Sometimes, with buffer connections, it is desirable to use only single buffer synchronization instead of double buffers. This approach helps when local data memory is limited and the performance impact of using a single buffer for data transfer is not critical. This can be achieved using the `single_buffer(port<T>&)` constraint.

```
single_buffer(first.in[0]); //For buffer input or RTP input
single_buffer(first.inout[0]); //For RTP output
```
