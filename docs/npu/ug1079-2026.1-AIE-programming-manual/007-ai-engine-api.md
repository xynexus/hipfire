---
title: "AI Engine API"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-API"
toc_id: vKHq1xrf9yxeuWRMl0Tnjw
content_id: 3zKYQ1S_JrgkVq1mRsK88Q
---

## AI Engine API

The AI Engine API is a portable programming interface for AI Engine kernel programming. This API interface targets current and future AI Engine architectures. For AI Engine API documentation, see the AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

The interface provides parameterizable data types that enable generic programming and also implements the most common operations in a uniform way across different data types. It is an easier programming interface than using intrinsic functions and is implemented as a C++ header-only library that translates into optimized intrinsic functions.

For advanced users who require programming with intrinsics, refer to AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)).

The AI Engine API user guide is organized as follows:

- **API Reference:** Basic Types Memory Initialization Arithmetic Comparison Reduction Reshaping Floating-point Conversion Elementary Functions Matrix Multiplication Fast Fourier Transform (FFT) Special Multiplications Operator Overloading Interoperability with Adaptive Data Flow (ADF) Graph Abstractions

![ihq1631655071544.png](../assets/007-01-ihq1631655071544-png-dc7668b71d6e.png)

*Figure 1. AI Engine API User Guide*
