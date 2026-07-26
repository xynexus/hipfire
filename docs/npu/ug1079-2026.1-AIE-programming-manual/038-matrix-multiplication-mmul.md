---
title: "Matrix Multiplication - mmul"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Matrix-Multiplication-mmul"
toc_id: 81yQuhAUYvWBVcB_K7G1Hw
content_id: 8vLFWge_s2mbSHDHhFDmBQ
---

## Matrix Multiplication - mmul

The AI Engine API encapsulates the matrix multiplication functionality in the `aie::mmul` class template. This class template uses the matrix multiplication shape `(M×K×N)`, the data types, and optionally the requested accumulation precision as parameters. For the supported shapes, see [Matrix Multiplication](https://www.xilinx.com/htmldocs/xilinx2024_2/aiengine_api/aie_api/doc/group__group__mmul.html) in AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

It defines one function for the initial multiplication (`mul`) and one function for multiply-add (`mac`). You can initialize `aie::mmul` objects from vectors or accumulators. This lets you use them in chained computations where partial results are sent over the cascade.

The resulting class defines a function that performs the multiplication and a result data type that converts to an accumulator or vector. The function interprets the input vectors as matrices as described by the shape parameters.

The following is a sample code to compute a C(2x64) = A(2x8) * B(8x64) matrix multiplication, using 2*4*8 mode of `mmul`. One iteration of the loop does: C0(2x8) = A0(2x4) * B0(4x8) + A1(2x4) * B1(4x8), where:

- A0 is left half of A
- A1 is right half of A
- B0 is upper left 4x8 matrix of B
- B1 is lower left 4x8 matrix of B
- C0 is leftmost 2x8 matrix of C

Store all matrix data in row-major format in memory. Matrix A is read into a vector, per instructions, thus requires data filtering for `mmul`. B0 and B1 are read a row (eight elements) at a time. Four rows are combined for `mmul`. The indexes of two rows of C0 need to be calculated and two rows of C0 are written to memory separately.

**Note:** `mmul`

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "aie_api/utils.hpp"

// For element mmul
const int M=2;
const int K=4;
const int N=8;

// Total matrix sizes
const int rowA=2;
const int colA=8;
const int colB=64;
const int SHIFT_BITS=0;
using namespace adf;
using MMUL = aie::mmul<M, K, N, int16, int16>;
void matmul_mmul(input_buffer<int16>& __restrict data0,
    input_buffer<int16>& __restrict data1, output_buffer<int16>& __restrict out){
  auto pa=aie::begin_vector<MMUL::size_A*2>(data0);
  aie::vector<int16,MMUL::size_A*2> va=*pa;

  // select left half matrix of A into va0
  aie::vector<int16,MMUL::size_A> va0=aie::filter_even(va,4);

  // select right half matrix of A into va1
  aie::vector<int16,MMUL::size_A> va1=aie::filter_odd(va,4);

  auto pb0=aie::begin_vector<8>(data1);
  auto pb1=pb0+32;
  aie::vector<int16,N> vb0_[4];
  aie::vector<int16,N> vb1_[4];
  aie::vector<int16,MMUL::size_C> vc;
  auto pc=aie::begin_vector<8>(out);
  for(int i=0;i<colB/N;i++)
  chess_prepare_for_pipelining
  {
    for(int j=0;j<4;j++){
      vb0_[j]=*pb0;
      pb0+=8;
      vb1_[j]=*pb1;
      pb1+=8;
    }
    MMUL m;
    m.mul(va0,aie::concat(vb0_[0],vb0_[1],vb0_[2],vb0_[3]));
    m.mac(va1,aie::concat(vb1_[0],vb1_[1],vb1_[2],vb1_[3]));
    vc=m.to_vector<int16>(SHIFT_BITS);//right shift SHIFT_BITS
    *pc=vc.extract<8>(0);
    pc+=8;
    *pc=vc.extract<8>(1);
    pc-=7;
    pb0-=31;
    pb1-=31;
  }
}
```
