---
title: "Data Permute and MAC Examples"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Permute-and-MAC-Examples"
toc_id: pT1opN2nO6U2tH3wGoBLPw
content_id: DNRLVdh41PebMsOxVkVrtA
---

### Data Permute and MAC Examples

The following example takes two vectors with reals in `rva` and imaginary in `rvb` (with type `v8int32`). The example creates a new complex vector, using the offsets to interleave the values as required.

```
v8cint32 cv = as_v8cint32(select16(0xaaaa, concat(rva, rvb),
		0, 0x03020100, 0x07060504,  8, 0x30201000, 0x70605040));
```

The following example shows how to extract real and imaginary portion of a vector `cv` with type `v8cint32`.

```
v16int32 re_im  = shuffle16(as_v16int32(cv), 0, 0xECA86420, 0xFDB97531);
v8int32 re = ext_w(re_im, 0);
v8int32 im = ext_w(re_im, 1);
```

Shuffle intrinsic functions can be used to reorder the elements in a vector or set all elements to the same value. Some intrinsic functions operate only on larger registers but it is easy to use them for smaller registers. The following example shows how to implement a function to set all four elements in a vector to a constant value.

```
v4int32 v2 = ext_v(shuffle16(xset_v(0, v1), 0 ,0, 0), 0);
```

The following example shows how to multiply each element in `rva` by the first element in `rvb`. This is efficient for a vector multiplied by constant value.

```
v8acc80 acc = lmul8(concat(rva,undef_v8int32()),0,0x76543210,rvb,0,0x00);
```

The following examples show how to multiply each element in `rva` by its corresponding element in `rvb`.

```
acc = lmul8(concat(rva, undef_v8int32()),0,0x76543210,rvb,0,0x76543210);
acc = lmul8(upd_w(undef_v16int32(),0,rva),0,0x76543210,rvb,0,0x76543210);
```

The following examples show how to do matrix multiplication for int8 x int8 data types with `mul` intrinsic, assuming that data storage is row based.

```
//Z_{2x8} * X_{8x8} = A_{2x8}
mul16(Xbuff, 0, 0x11101110, 16, 0x3120, Zbuff, 0, 0x44440000, 2, 0x3210);
//Z_{4x8} * X_{8x4} = A_{4x4}
mul16(Xbuff, 0, 0x00000000, 8, 0x3120, Zbuff, 0, 0xCC884400, 2, 0x3210);
```

If the kernel has multiple `mul` or `mac` intrinsics, try to keep the `xoffsets` and `zoffsets` parameters constant across uses. Also vary the `xtsart` and `zstart` parameters. This helps prevent configuration register spills on stack.

For more information about vector lane permutations, refer to the AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)).
