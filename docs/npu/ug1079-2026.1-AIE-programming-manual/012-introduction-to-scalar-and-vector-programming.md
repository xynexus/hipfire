---
title: "Introduction to Scalar and Vector Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Introduction-to-Scalar-and-Vector-Programming"
toc_id: 4_G1mqp_hD42QbH3SwPGHg
content_id: djiaRl8mcreliJRqolvTqA
---

# Introduction to Scalar and Vector Programming

This section provides an overview of the key elements of kernel programming for scalar and vector processing elements. The following sections describe each element and optimization skills.

The following example demonstrates a `for` loop iterating through 512 `int32` elements. Each loop iteration pulls one element from each input buffer, multiplies them together and places the product in the output buffer. The `scalar_mul` kernel operates on two input buffers `input_buffer<int32>` and updates an output buffer `output_buffer<int32>`.

Iterators read and write to the buffers outside the kernel. The code sample below does not use any vector registers and will be implemented on the scalar engine.

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
using namespace adf;
void scalar_mul(input_buffer<int32>& __restrict data1,
                input_buffer<int32>& __restrict data2,
                output_buffer<int32>& __restrict out) {
  auto inIter1=aie::begin(data1);
  auto inIter2=aie::begin(data2);
  auto outIter=aie::begin(out);
  for(int i=0;i<512;i++) {
    int32 a=*inIter1++;
    int32 b=*inIter2++;
    int32 c=a*b;
    *outIter++=c;
  }
}
```

The following example is a vectorized version for the same kernel and will be implemented on the vector processor.

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
using namespace adf;
void vect_mul(input_buffer<int32>& __restrict data1,
              input_buffer<int32>& __restrict data2,
              output_buffer<int32>& __restrict out) {
  //iterator for vector of 8 elements
  auto inIter1=aie::begin_vector<8>(data1);
  auto inIter2=aie::begin_vector<8>(data2);
  auto outIter=aie::begin_vector<8>(out);
  for(int i=0;i<512/8;i++) chess_prepare_for_pipelining {
    //vector of 8 elements
    auto va=*inIter1++;
    auto vb=*inIter2++;

    //element-by-element multiplication
    auto vt=aie::mul(va,vb);
    *outIter++=vt.to_vector<int32>(0);
  }
}
```

The iterators returns a vector of eight `int32` and stores them in variables named `va` and `vb`. These two variables are vector type variables and they are passed to the API function `aie::mul`. The result of the `aie::mul` function is stored in `vt`, which is an accumulator with data type `aie::accum<acc80,8>`. The accumulator is then converted by the shift-round-saturate function `to_vector` to a variable of `aie::vector<int32,8>` type. The result is then written to the output buffer. The following sections contain additional details on the data types supported by the AI Engine.

The `__restrict` keyword used on the input and output parameters of the functions, allows for more aggressive compiler optimization by explicitly stating independence between data. For more information, see Restrict Keyword.

`chess_prepare_for_pipelining` is a compiler pragma that explicitly directs kernel compiler to achieve optimized pipeline for the loop.

The scalar version of this example function needs 1045 cycles, while the vectorized and optimized version needs only 88 cycles. That means that there is more than ten times speedup for the vectorized version of the kernel. Vector processing itself gives 8x the throughput for int32 multiplication. However, with loop optimizations, vector processing can achieve more than 10x.

To calculate the maximum performance for a given datapath, it is necessary to multiply the number of multiply-accumulates (MACs) per instruction with the clock frequency of the AI Engine kernel. For example, with 32-bit input vectors X and Z, the vector processor can achieve 8 MACs per instruction. Using the clock frequency for the slowest speed grade device results in:

The sections that follow describe in detail the various data types that can be used, registers available, and also the kinds of optimizations that can be achieved on the AI Engine using concepts like software pipelining in loops and keywords like `__restrict`.

Related reference

Software Pipelining of Loops
