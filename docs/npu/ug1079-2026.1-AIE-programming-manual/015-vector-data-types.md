---
title: "Vector Data Types"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Data-Types"
toc_id: aefEWYfr31cKts4V33ZCzA
content_id: 0CjrjVgnedsHfabMHQV37Q
---

## Vector Data Types

The two main vector types offered by the AI Engine API are vectors (`aie::vector`) and accumulators (`aie::accum`).

#### Vector

A vector represents a collection of same-type elements mapped transparently to the corresponding vector registers supported on AI Engine architectures. Vectors are parameterized by the element type and count. Any combination that defines a 128 b/256 b/512 b/1024 b vector is supported. The default is 512 b.

| Vector Types | Sizes 1 |
| --- | --- |
| int8 | 16/32/64/128 |
| int16 | 8/16/32/64 |
| int32 | 4/8/16/32 |
| uint8 | 16/32/64/128 |
| float | 4/8/16/32 |
| cint16 | 4/8/16/32 |
| cint32 | 2/4/8/16 |
| cfloat | 2/4/8/16 |
| The integer in bold is the native vector size for the data type supported in the AI Engine. For example, aie::broadcast((cfloat){1,1}) is equivalent to aie::broadcast<cfloat,8>((cfloat){1,1}). Specifying <cfloat,8> is optional because it is the native vector size for the cfloat data type. |  |

For example, `aie::vector<int32,16>` is a 16 element vector of integers with 32 bits. Each vector element is called a lane. Using the smallest bit width necessary can improve performance by making good use of registers.

![hjd1607630994973.png](../assets/015-01-hjd1607630994973-png-26a2446b70cd.png)

`real`

`imag`

```
cint8 ctmp={1,2};

// print real and imag values
printf("real=%d imag=%d\n",ctmp.real,ctmp.imag);

// store real and imag values
printf("ctmp mem storage=%llx\n",*(long long*)&ctmp);
cint32 ctmp2={3,4};
int32 *p_ctmp2=reinterpret_cast<int32*>(&ctmp2);

// print "real=3" and "imag=4"
printf("real=%d imag=%d\n",p_ctmp2[0],p_ctmp2[1]);
```

`aie::vector` and `aie::accum` have member functions to do type casting, data extraction and insertion, and indexing. These operations are covered in following sections.

*Figure 1. aie::vector<int32,16>*

#### Accumulator

An accumulator represents a collection of elements of the same class, typically obtained as a result of a multiplication operation, which is transparently mapped to the corresponding accumulator registers supported on each architecture. Accumulators commonly provide a large number of bits, allowing users to perform long chains of operations whose intermediate results might exceed the range of regular vector types. Accumulators are parameterized by the element type, and the number of elements. The native accumulation bits define the minimum number of bits and the AI Engine API maps different types to the nearest native accumulator type that supports the requirement. For example, `acc40` maps to `acc48` for the AI Engine architecture.

The following table lists the native accumulator data types that AI Engine API kernel coding supports. For a complete list, see [Basic Types](https://www.xilinx.com/htmldocs/xilinx2024_2/aiengine_api/aie_api/doc/group__group__basic__types.html) in the AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)). For the kernel cascade interface and graph support, see Stream Data Types.

| Type | acc48 | cacc48 | acc80 | cacc80 | accfloat | caccfloat |
| --- | --- | --- | --- | --- | --- | --- |
| Native accumulation bits | 48 | 80 | 32 |  |  |  |
