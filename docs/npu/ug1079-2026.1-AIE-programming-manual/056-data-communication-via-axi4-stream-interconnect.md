---
title: "Data Communication via AXI4-Stream Interconnect"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Communication-via-AXI4-Stream-Interconnect"
toc_id: wOyd8AM99wH29bFmEGlVCA
content_id: nCXcN~Pucn_ED8BrCM3tsQ
---

### Data Communication via AXI4-Stream Interconnect

AI Engines can directly communicate through the AXI4-Stream interconnect without any DMA and memory interaction. Data can transfer from one AI Engine to another or broadcast through the streaming interface. The data bandwidth of a streaming connection is 32-bit per cycle and built-in handshake and backpressure mechanisms are available.

The stream connection can be unicast or multicast.

**Note:** In multicast communication, data is sent to all the destination ports at the same time and only when all destinations are ready to receive data.
