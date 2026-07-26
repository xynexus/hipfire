---
title: "Design Analysis and Programming using Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Design-Analysis-and-Programming-using-Intrinsics"
toc_id: zIODppeHr4lmGbGUFTbfCQ
content_id: cWUOyX94FQUOtVv_mGUT2g
---

# Design Analysis and Programming using Intrinsics

**Note:** AMD

AI Engine

AI Engine

AI Engine

`fft_data_incr`

`cyclic_add`

AI Engine

AI Engines provide high compute density through large numbers of VLIW and SIMD compute units using connections through innovative memory and AXI4-Stream networks. When targeting an application on AI Engine, it is important to evaluate the compute needs of the AI Engine and data throughput requirements. For example, how the AI Engine interacts with PL kernels and external DDR memory. After meeting the compute and data throughput requirements for AI Engine, the next step involves divide and conquer methods. These map the algorithm into the AI Engine array.

For the divide and conquer step, you must understand vector processor architecture, memory structure, AXI4-Stream, and cascade stream interfaces. This step is often repeated. During each iteration, you optimize individual AI Engine kernels and construct and refine the graph. AI Engine tools simulate and debug AI Engine kernels and the graph. The graph is then integrated with PL kernels, GMIO, and PS to perform system level verification and performance tuning.

This chapter introduces the divide and conquer method for mapping the algorithm into data flow diagrams (DFD). Single and multiple kernel programming examples illustrate kernel partitioning by the compute and memory bound, single kernel vectorization and optimization, and streaming balancing between different kernels.

**Note:** `input_window`

`output_window`

`input_buffer`

`output_buffer`
