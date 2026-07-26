---
title: "Memory and DMA Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Memory-and-DMA-Programming"
toc_id: 9X3HmCfDmpyoLb7iDrdyew
content_id: LsZ4W2B9bVs71PUNei3waA
---

# Memory and DMA Programming

The AI Engine processor can leverage local memory for efficient data access:

- **AI Engine memory:** The AI Engine processor has on-tile data memory, providing efficient access within the same processing unit. Additionally, the AI Engine processor can access data memory from neighboring tiles to the north, south, and west (or east). This access expands its reach and enables efficient data flow within a larger AI Engine array.

You can address this linearly within dedicated address ranges. Additionally, they support multidimensional addressing for more complex access patterns. Direct Memory Access (DMA) controllers manage these memories automatically. These DMAs offer various programmable features to optimize memory access, including:

| DMA Feature | Description | AI Engine Tile DMA 1 |
| --- | --- | --- |
| Maximum Addressing Dimension | Maximum dimension for accessing data available to the DMA depending on the type of memory. Depending on the application, you can access data in a uni-dimensional (1D), bi-dimensional (2D, as in a gray-scale image). | 2D |
| Packet-ID | Feature that is used by AXI4-Stream switch to drive packets to their destination based on the Packet ID. See Explicit Packet Switching. | Supported |
| Number of Buffer Descriptors | Lists the total number of buffer descriptors available in the memory. A Buffer Descriptor describes a DMA transfer. Each buffer descriptor contains all information needed for a DMA transfer. They are used by the DMA to specify the read/write access schemes to the memory. | 16 |
| Number of Locks | List the total number of locks available in the memory. These locks are used by the DMA to handle synchronization for Buffer Descriptor usage. They are used to organize read and write access to the memory. | 16 |
| Used to access local buffers. |  |  |

The AI Engine tiling parameters enable you to program the DMAs for the AI Engine Memory Module. Tiling parameters can be declared and defined in the ADF graph and connected to the appropriate kernel inputs or outputs. For more information on tiling parameters, see Tiling Parameters and Buffer Descriptors.
