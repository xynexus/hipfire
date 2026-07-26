---
title: "API Operation Examples"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/API-Operation-Examples"
toc_id: SIJsl21hhsKkfgRpdc0Qbg
content_id: MYaX~eLzqDP0MnjXB7SgrQ
---

## API Operation Examples

The following example takes two vectors with reals in `rva` and imaginary in `rvb` (with type `aie::vector<int32,8>`) and creates a new complex vector, using the offsets to interleave the values as required.

```
aie::vector<int32,8> rva,rvb;
auto rv=aie::interleave_zip(rva,rvb,1);
aie::vector<cint32,8> cv=aie::concat(rv.first.cast_to<cint32>(),rv.second.cast_to<cint32>());
```

The following example shows how to extract real and imaginary portion of a vector `cv` with type `aie::vector<cint32,8>`.

```
aie::vector<cint32,8> cv;
aie::vector<int32,16> re_im=cv.cast_to<int32>();
aie::vector<int32,8> re=aie::filter_even(re_im,1);
aie::vector<int32,8> im=aie::filter_odd(re_im,1);
```

You can use `aie::broadcast` to set every element of a vector to a given value. The following example shows how to set all eight elements in a vector to a constant value.

```
// set all elements to 100
aie::vector<int32,8> v1=aie::broadcast<int32,8>(100);
```

The following example shows how to use `aie::broadcast` to set multiple values repeatedly in the vector.

```
alignas(aie::vector_decl_align) int16 init_data[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

// set 0 1 2 3 repeatedly into buff
aie::vector<int16,32> buff=aie::broadcast<cint32,8>(*(cint32*)init_data).cast_to<int16>();
```

The following example shows how to multiply each element in `rva` by the first element in `rvb`. This is efficient for a vector multiplied by constant value.

```
aie::vector<int32,8> rva,rvb;
aie::accum<acc80,8> acc = aie::mul(rva,rvb[0]);
```

The following examples show how to multiply each element in `rva` by its corresponding element in `rvb`.

```
aie::vector<int32,8> va,vb;
aie::accum<acc80,8> acc=aie::mul(va,vb);
```

The following examples show how to perform matrix multiplication for int8 x int8 data types with `mmul` intrinsic. They assume that data storage is in row-major format.

```
// Z_{2x8} * X_{8x8} = A_{2x8}
aie::vector<int8,16> Z;
aie::vector<int8,64> X;
aie::mmul<2,8,8,int8,int8> m;
m.mul(Z,X);

// Z_{4x8} * X_{8x4} = A_{4x4}
aie::vector<int8,32> Z;
aie::vector<int8,32> X;
aie::mmul<4,8,4,int8,int8> m;
m.mul(Z,X);
```

For more information about vector lane permutations, refer to the AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)).
