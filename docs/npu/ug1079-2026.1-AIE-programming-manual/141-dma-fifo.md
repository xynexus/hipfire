---
title: "DMA FIFO"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/DMA-FIFO"
toc_id: P8Wo17oFzl5mpbbeCRLWLA
content_id: jfPRgufBcl1CgYyKGCLX_w
---

### DMA FIFO

A `fifo_depth()` constraint specification above 40 allocates FIFOs from memory, known as DMA FIFOs. The following is an example of a FIFO allocation for a request of `fifo_depth(128)` bytes which is allocated in memory.

![zts1679329672282.png](../assets/141-01-zts1679329672282-png-e0088c46e502.png)

*Figure 1. DMA FIFO Allocation*

**Note:** TLAST drops when data goes through DMA FIFO.

**Note:** Write to DMA FIFO must be continuous or multiple of 4 words (4 words = 128 bits).

You can also specify the type of FIFO allocated, whether stream switch or DMA, as well as their locations. See FIFO Location Constraints for more information.
