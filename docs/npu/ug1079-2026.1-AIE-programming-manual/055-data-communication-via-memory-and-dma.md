---
title: "Data Communication via Memory and DMA"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Communication-via-Memory-and-DMA"
toc_id: ~98sIej6mfwTSeT~qHkFng
content_id: L3DD3v6OqbSXXRHlaEdmcw
---

### Data Communication via Memory and DMA

For non-neighboring AI Engines, establish similar communication using the DMA in the memory module associated with each AI Engine. Ping-pong buffers in each memory module are used and synchronization is carried out with locks. There is increased communication latency as well as memory resources in comparison to shared memory communication.

![rbj1606891786730.png](../assets/055-01-rbj1606891786730-png-f4735325117a.png)

*Figure 1. Data Communication via Memory and DMA*
