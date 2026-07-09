---
title: "FIFO Depth"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/FIFO-Depth"
toc_id: 9MJDHkY_OVSfE8Ji9vBI~Q
content_id: ailJVWsUm7AwwoVi3NNn5A
---

## FIFO Depth

The AI Engine architecture uses stream data extensively for:

- DMA-based I/O,
- Communicating between two AI Engines, and
- Communicating between the AI Engine and the programmable logic (PL)

This raises the potential for a resource deadlock when the data flow graph has reconvergent data paths. If the pipeline depth of one path is longer than the other, the producer kernel can stall. If this happens, it might not be able to push data into the shorter path because of back pressure. At the same time, the consumer kernel is waiting to receive data on the longer path due to the lack of data. If the order of data production and consumption between two data paths is different, a deadlock can occur. This can even happen between two kernels that are directly connected with two data paths. The following figure illustrates the paths.

*Figure 1. Producer and Consumer Kernels with Reconvergent Streams*

![tiu1513126478514.png](../assets/139-01-tiu1513126478514-png-8a4c46887fb1.png)

If the producer kernel is trying to push data on stream S1 and encounters back pressure while the consumer kernel is still trying to read data from stream S2, a deadlock occurs. You can fix this situation by creating more buffering in the paths that have back pressure in the source code. Do this by using a `fifo_depth` constraint on a connection.

```
p = kernel::create(producer);
c = kernel::create(consumer);
connect s1(p.out[0], c.in[0]);
connect s2(p.out[1], c.in[1]);
fifo_depth(s1) = 20;
fifo_depth(s2) = 10;
```

**Note:** `fifo_depth()`

AI Engines
