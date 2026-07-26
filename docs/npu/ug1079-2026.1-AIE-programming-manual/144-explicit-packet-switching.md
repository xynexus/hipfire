---
title: "Explicit Packet Switching"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Explicit-Packet-Switching"
toc_id: hy8l_XlL3k7O8t2DjQSFjw
content_id: SAKAqr~tGzGXbmT4sC5sSw
---

## Explicit Packet Switching

Multiple AI Engine kernels can share a single processor and execute in an interleaved manner. Similarly, multiple stream connections can share a single physical channel. This mechanism is known as Packet Switching. The AI Engine architecture and compiler work together to provide a programming model where up to 32 stream connections can share the same physical channel.

Explicit Packet Switching allows fine-grain control over how packets are generated, distributed, and consumed in a graph computation. Explicit Packet Switching is typically recommended in cases where many low bandwidth streams from common PL source can be distributed to different AI Engine destinations. Similarly, many low bandwidth streams from different AI Engine sources to a common PL destination can also take advantage of this feature. The number of AI Engine to PL interface streams used is minimal because a single physical channel is shared between multiple streams.

A packet stream can be created from:

- one AI Engine kernel to multiple destination kernels, or
- multiple AI Engine kernels to a single destination kernel, or
- between multiple AI Engine kernels and multiple destination kernels.

This section describes graph constructs to create packet-switched streams explicitly in the graph, and provide multiple examples on packet switching use cases.
