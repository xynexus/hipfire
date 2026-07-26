---
title: "Data Communication via AI Engine Data Memory"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Communication-via-AI-Engine-Data-Memory"
toc_id: 6CGXqk2FK9UQx2p9ahqRWw
content_id: l9GP1GzSwg4_VnXMG~b03g
---

### Data Communication via AI Engine Data Memory

When multiple kernels fit in a single AI Engine, consecutive kernels can communicate through a shared buffer in either of the following:

- The AI Engine’s local data memory, or
- One of the three neighboring memories the AI Engine can access directly.

In this case, only a single buffer is needed because the kernels execute one after another in a round-robin fashion.

When kernels are in separate but neighboring AI Engines, they can communicate through the data memory module shared between the two neighboring AI Engine tiles that use ping-pong buffers. These buffers can be on separate memory banks to avoid access conflicts. The synchronization uses locks. The input and output buffers for the AI Engine kernel are ensured to be ready by the locks associated with the buffers. This type of communication saves routing resources and eliminates data transfer latency because DMA and AXI4-Stream interconnect are not required.
