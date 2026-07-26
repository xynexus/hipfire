---
title: "Matrix Multiplication"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Matrix-Multiplication"
toc_id: tz38cJk3mWC9Cox_a2kCVA
content_id: E0jXKKjMnM79wps~xh9umg
---

## Matrix Multiplication

The AI Engine API provides a `aie::mmul` class template for a vector-based matrix multiplication. Multiple intermediate matrix multiplication results are accumulated to give the final result. For more details on the supported matrix multiplication shapes (`M*K*N`) and data types, see [Matrix Multiplication](https://www.xilinx.com/htmldocs/xilinx2024_2/aiengine_api/aie_api/doc/group__group__mmul.html) in the AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

The `aie::mmul` operations `mul` and `mac` accept row-major format data for the vector-based matrix multiplication. Then for the `mac` operation of `aie::mmul`, arrange the data by `M*K` or `K*N`. This data shuffling can be done either in the PL or AI Engine.

This section gives an example of `A(64 * 64) x B(64 * 64)` matrix multiplication. The data type is `int8 x int8`. The matrix multiplication shape `4*16*8` is chosen for `aie::mmul` operations.

The input data, in row-major format is input to the matrix multiplication kernel as `4*16` matrix and `16*8` matrix. Prior to the matrix multiplication kernel, the input data is shuffled.

For example, before shuffling, matrix `A(64 * 64)` is stored in memory with `a0`, `a1`, …, `a63`, `a64`,…, `a4096` in order. The `aie::mmul` operations uses shapes `4*16`. The matrix `A` is partitioned into smaller matrix sized `4*16`. For the smaller matrix `A00`, `a0` to `a15`, `a64` to `a79`, `a128` to `a143`, and `a192` to `a207` should be fetched sequentially for `aie::mmul`. So, the purpose of the data shuffle is to put `a0` to `a15`, `a64` to `a79`, `a128` to `a143`, and `a192` to `a207` into continuous storage for the matrix multiplication kernel. The following figure shows data shuffling.

![qhw1665632468641.png](../assets/068-01-qhw1665632468641-png-12c63c813c3b.png)

*Figure 1. Data Shuffling for Matrix A*

Similarly, the output data is shuffled. The following figure shows the graph of the design.

![mxb1665765790710.png](../assets/068-02-mxb1665765790710-png-7de9fa5978a2.png)

*Figure 2. Matrix Multiplication Kernel Graph*
