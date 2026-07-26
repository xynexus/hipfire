---
title: "Interface Considerations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Interface-Considerations"
toc_id: vn5nSKZtr4UWtII6VkT2gg
content_id: KW1NpcTT7n6sXGC4JX50BA
---

# Interface Considerations

Single-kernel programming focuses on vectorization of algorithm in a single AI Engine. Multiple-kernel programming uses multiple AI Engine kernels with data flowing between them.

The ADF graph can contain a single kernel or multiple kernels interacting with PS, PL, and global memory. Each AI Engine kernel has a runtime ratio. This number is the ratio of the number of cycles taken by one kernel invocation (processing one data block) to the cycle budget. The cycle budget for an application is typically fixed according to the expected data throughput and the block size being processed. The runtime ratio is specified as a constraint for every AI Engine kernel in the ADF graph.

The AI Engine compiler allocates multiple kernels into a single AI Engine if the following conditions are met:

- Their combined total runtime ratio is less than one and multiple kernels fit in the AI Engine program memory, and
- If the total resource usage, like stream interface number, does not exceed the AI Engine tile limit.

Alternatively, the compiler can allocate them into multiple AI Engines.

**Note:** Important:

AI Engine

- two 32-bit AXI4-Stream inputs
- two 32-bit AXI4-Stream outputs
- one 384-bit cascade stream input
- one 384-bit cascade stream output
- two 256-bit data loads
- one 256-bit data store

To optimally use hardware resources, it is critical to understand the different methods available to do the following:

- Transfer data between the ADF graph and PS, PL, and global memory
- Transfer data between kernels
- Balance data movement
- Minimize memory or stream stalls

The following sections cover these methods in detail.
