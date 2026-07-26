---
title: "AI Engine Tile DMA Performance"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-Tile-DMA-Performance"
toc_id: OP_ekcmcQGTC3tQ6IVFpiw
content_id: uwWzh9Whg7Z_1RSdRKkkYA
---

### AI Engine Tile DMA Performance

In high throughput use cases, the AI Engine and PL throughput can be close to maximum. When using a DMA FIFO, and the PL communicates with the DMA FIFO in an asynchronous PL to AI Engine clock relationship, the read side must occasionally wait for data due to nature of a single DMA FIFO. This can lead to slightly lower than 100% throughput on the AI Engine. Some of the recommended ways to avoid the small loss in throughput are as follows.

- Choose a `fifo_depth` constraint of less than or up to 40 at the AI Engine-PL boundaries on streaming connections with a slack of 40 or less.
- Add a small asynchronous FIFO in the PL to shift the alignment into the AI Engine clock domain.
- Use a synchronous PL clock to the AI Engine. Use a 128-bit AXI4-Stream interface from the PL and use a PL clock at integer multiples of the AI Engine frequency.
