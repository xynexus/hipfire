---
title: "Stream Switch FIFO"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Stream-Switch-FIFO"
toc_id: J6QvFkykd9w0oAHnPmtIhA
content_id: 2AjgqS0B4iXGBoFqmys_Dw
---

### Stream Switch FIFO

The AI Engine has two 32-bit input AXI4-Stream interfaces and two 32-bit output AXI4-Stream interfaces. Each stream connects to a FIFO both on the input and output side. This allows the AI Engine to have a four word (128-bit) access per four cycles, or a one word (32-bit) access per cycle on a stream. A `fifo_depth()` constraint specification below 40 allocates FIFOs from the stream switch. The following is an example of a FIFO allocation on the stream switch requesting a `fifo_depth(32)`.

![azm1679329825475.png](../assets/140-01-azm1679329825475-png-5dc7dcfac952.png)

*Figure 1. FIFO Allocation on Stream Switch*
