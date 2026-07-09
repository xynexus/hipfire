---
title: "Introduction to Scalar and Vector Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Introduction-to-Scalar-and-Vector-Programming"
toc_id: yfwoMQthLaaq8W0h9dHSUw
content_id: Etj5EdJDWe3TB2Z_ZrztRg
---

## Introduction to Scalar and Vector Programming

This section provides an overview of the key elements of kernel programming for scalar and vector processing elements. The following sections include details about each element and optimization skills.

The following example uses only the scalar engine. It demonstrates a for loop iterating through 512 int32 elements. Each loop iteration performs a single multiply of int32 a and int32 b storing the result in c and writing it to an output buffer. The scalar_mul kernel processes two input blocks (buffer) of data `input_buffer<int32>` and produces an output buffer of data `output_buffer<int32>`.

You can access buffers through scalar and vector iterators. For additional details on the buffer APIs, see Streaming Data API.

```
void scalar_mul(input_buffer<int32> & data1,
			input_buffer<int32> & data2,
			output_buffer<int32> & out){
        auto pin1 = aie::begin(data1);
        auto pin2 = aie::begin(data2);
        auto pout = aie::begin(out);
	for(int i=0;i<512;i++)
	{
		int32 a=*pin1++;
		int32 b=*pin2++;
		int32 c=a*b;
		*pout++ = c;
	}
}
```

The following example is a vectorized version for the same kernel.

```
void vect_mul(input_buffer<int32> & __restrict data1,
			input_buffer<int32> & __restrict data2,
			output_buffer<int32> & __restrict out){
        auto pin1 = aie::begin_vector<8>(data1);
        auto pin2 = aie::begin_vector<8>(data2);
        auto pout = aie::begin_vector<8>(out);
	for(int i=0;i<64;i++)
	chess_prepare_for_pipelining
	{
		aie::vector<int32,8> va=*pin1++;
		aie::vector<int32,8> vb=*pin2++;
		aie::accum<acc80,8> vt=mul(va,vb);
		aie::vector<int32,8> vc=srs(vt,0);
		*pout++ = vc;
	}
}
```

Note the data types `vector<int32,8>` and `accum<acc80,8>` are used in the previous kernel code. The buffer API `begin_vector<8>` returns an iterator that will iterate over vectors of 8 int32s and stores them in variables named `va` and `vb`. These two variables are vector type variables and they are passed to the intrinsic function `mul` which outputs `vt` which is a `accum<acc80,8>` data type. The `accum<acc80,8>` type is reduced by a shift round saturate function `srs` that allows a `vector<int32,8>` type, variable vc, to be returned and then written to the output buffer. You can find additional details about supported AI Engine data types in the following sections.

The `__restrict` keyword used on the input and output parameters of the `vect_mul` function, allows for more aggressive compiler optimization by explicitly stating independence between data.

`chess_prepare_for_pipelining` is a compiler pragma that directs kernel compiler to achieve optimized pipeline for the loop.

The scalar version of this example function takes 1055 cycles while the vectorized version takes only 99 cycles. As you can see there is more than 10 times speedup for vectorized version of the kernel. Vector processing itself gives 8x the throughput for int32 multiplication but has a higher latency and does not get 8x the throughput overall. However, with the loop optimizations done, it can get close to 10x. The following sections describe in detail the various data types and available registers. The sections also explain AI Engine optimizations using concepts such as software pipelining in loops and keywords like `__restrict`.
