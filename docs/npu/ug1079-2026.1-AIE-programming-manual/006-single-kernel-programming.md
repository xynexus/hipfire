---
title: "Single Kernel Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Single-Kernel-Programming"
toc_id: 0BNqeYD3jluI6rBGBldIYg
content_id: I50HKhRRWxdhxL~E0ZEdMw
---

# Single Kernel Programming

An AI Engine kernel is a C++ program written using specialized intrinsic functions. These intrinsic functions target the different functional units of the AI Engine processor, like the VLIW vector and scalar unit. The AI Engine kernel code is compiled using the AI Engine compiler, which is included in the AMD Vitis™ core development kit. The AI Engine compiler converts kernel code into ELF files that run on the AI Engine processors.

The AI Engine supports specialized data types and functions for vector processing. By restructuring some scalar application code with these API functions and vector data types, one can create fast and efficient vectorized code. The compiler takes care of mapping functions to operations, performing register allocation and data movement, scheduling and generation of microcode. The compiler packs the microcode efficiently into VLIW instructions.

The following chapters introduce the data types supported and registers available for use by the AI Engine kernel. Additionally, they describe the vector API functions that initialize, load, store, and operate on the vector registers using the appropriate data types.

For highest performance on the AI Engine, the primary goal of single kernel programming is to ensure that the vector processor approaches its theoretical maximum. Vectorization of the algorithm is important, but managing the vector registers, memory access, and software pipelining is also required.

Try to make the data for the new operation available while the current one runs because the vector processor can execute an operation every clock cycle. Optimizations using software pipelining in loops are available using pragmas. For instance, when the inner loop has sequential or loop carried dependencies it might be possible to unroll an outer loop and compute multiple values in parallel. The following sections discuss these concepts.
