---
title: "Single Kernel Programming using Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Single-Kernel-Programming-using-Intrinsics"
toc_id: peLsXyD55R~zNedY12dVcQ
content_id: 1zKlTdVGgC~WDMBOrEqb0g
---

# Single Kernel Programming using Intrinsics

**Note:** AI Engine

Use intrinsics only when the design’s strict performance requirements exceed what the AI Engine API supports. For example, the AI Engine API does not currently support functionality provided by some intrinsics such as, `fft_data_incr` and `cyclic_add`. AI Engine APIs support and abstract the main permute use cases, but they do not cover not all permute capabilities. Using intrinsics can allow you to close the performance gap required by your design.

An AI Engine kernel is a C/C++ program. It uses native C/C++ language and specialized intrinsic functions that target the VLIW scalar and vector processors. The AI Engine kernel code is compiled using the AI Engine compiler that is included in the AMD Vitis™ core development kit. The AI Engine compiler compiles the kernels to produce ELF files that run on the AI Engine processors.

For more information on intrinsic functions, see the AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)). The first few sections of this chapter cover AI Engine compiler and simulator.

AI Engine supports specialized data types and intrinsic functions for vector programming. By restructuring the scalar application code with these intrinsic functions and vector data types as needed, you can implement the vectorized application code. The AI Engine compiler manages:

- Mapping intrinsic functions to operations
- Vector or scalar register allocation and data movement
- Automatic scheduling
- Generation of microcode that is efficiently packed in VLIW instructions

The following sections introduce the data types supported and registers available for use by the AI Engine kernel. The vector intrinsic functions that initialize, load, and store, as well as operate on the vector registers using the appropriate data types are also described.

To maximize AI Engine performance, single-kernel programming should aim to fully utilize the vector processor’s theoretical maximum. Vectorization of the algorithm is important, but managing the vector registers, memory access, and software pipelining are also required.

Try to make data for the next operation available while the current one runs, because the vector processor can execute one operation per clock cycle. Optimizations using software pipelining in loops is available using pragmas. For instance, when the inner loop has sequential or loop carried dependencies it might be possible to unroll an outer loop and compute multiple values in parallel. The following sections also cover these concepts.
