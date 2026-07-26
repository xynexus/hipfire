---
title: "Buffer vs. Stream in Data Communication"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-vs.-Stream-in-Data-Communication"
toc_id: o7hAkxqtO79wY9dMDCI~IA
content_id: KiXU903NugNVZTi~NNaVsA
---

## Buffer vs. Stream in Data Communication

AI Engine kernels in the data flow graph operate on data streams that are infinitely long sequences of typed values. These data streams can be broken into separate blocks called buffers and processed by a kernel. Kernels consume input blocks of data and produce output blocks of data. An initialization function can be specified to run before the kernel starts processing input data. The kernel can read scalars or vectors from memory. However, the valid vector length for each read and write operation must be a multiple of 128 bits.

Buffers of input data and output data are locked for kernels before they are executed. Because the input data buffer needs to be filled with input data before the kernel can start, it increases latency compared to stream interface. The kernel can perform random access within a buffer of data. It can also specify a margin for algorithms that require some number of bytes from the previous buffer.

Kernels can also access data streams in a sample-by-sample fashion. Streams are used for continuous data and use blocking or non-blocking calls to read and write. Cascade streams only supports blocking access. The AI Engine supports two 32-bit stream input ports and two 32-bit stream output ports.

The valid vector length for reading or writing data streams is either 32 or 128 bits. Packet streams are useful when the number of independent data streams in the program exceeds the number of hardware stream channels or ports available.

A PLIO port attribute enables external stream connections that cross the AI Engine and PL boundary. You can connect the PLIO port to the AI Engine buffer via DMA S2MM or MM2S channels. Alternatively, connect it directly to AI Engine stream interfaces. Both of these connections (PL from/to buffer or stream) depend on the AI Engine's stream interface tiles (limit of 32 bits per cycle). However, for the buffer interface, the ping or pong buffer needs to be filled up before the kernel can start. Therefore, buffer interfaces from / to PL usually have a larger latency than a stream interface.

The following table summarizes the differences in buffer and stream connections between kernels.

| Connection | Margin | Packet Switching | Back Pressure | Lock | Max throughput by VLIW (per cycle) 2 | Multicast as a Source |
| --- | --- | --- | --- | --- | --- | --- |
| Buffer | Yes | Yes | Yes 1 | Yes | 2*256-bit load + 1*256-bit store | Yes |
| Stream | No | Yes | Yes | No | 2*32-bit read + 2*32-bit write | Yes |
| Buffer back pressure, acquired or not, occurs on the whole buffer of data. Considering all data are available at kernel boundary. |  |  |  |  |  |  |

Graph code is C++ and available in a separate file from kernel source files. The compiler places the AI Engine kernels into the AI Engine array, handling memory requirements and making all the necessary connections for data flow. You can place multiple kernels with low core usage into a single tile.
