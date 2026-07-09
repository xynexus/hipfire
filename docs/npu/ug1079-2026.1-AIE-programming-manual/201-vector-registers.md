---
title: "Vector Registers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Registers"
toc_id: gyNe7Lng0AQeDjRBoFBk~Q
content_id: aG2LGApj8vV_9SI9sd6DgQ
---

## Vector Registers

All vector intrinsic functions require the operands to be present in the AI Engine vector registers. The following table shows the set of vector registers and how smaller registers are combined to form large registers.

| 128-bit | 256-bit | 512-bit | 1024-bit |  |
| --- | --- | --- | --- | --- |
| vrl0 | wr0 | xa | ya | N/A |
| vrh0 |  |  |  |  |
| vrl1 | wr1 |  |  |  |
| vrh1 |  |  |  |  |
| vrl2 | wr2 | xb | yd (msbs) |  |
| vrh2 |  |  |  |  |
| vrl3 | wr3 |  |  |  |
| vrh3 |  |  |  |  |
| vcl0 | wc0 | xc | N/A | N/A |
| vch0 |  |  |  |  |
| vcl1 | wc1 |  |  |  |
| vch1 |  |  |  |  |
| vdl0 | wd0 | xd | N/A | yd (lsbs) |
| vdh0 |  |  |  |  |
| vdl1 | wd1 |  |  |  |
| vdh1 |  |  |  |  |

The underlying basic hardware registers are 128-bit wide and prefixed with the letter v. Two v registers can be grouped to form a 256-bit register prefixed with w. wr, wc, and wd registers are grouped in pairs to form 512-bit registers (xa, xb, xc, and xd). xa and xb form the 1024-bit wide ya register, while xd and xb form the 1024-bit wide yd register. This means the xb register is shared between ya and yd registers. xb contains the most significant bits (MSBs) for both ya and yd registers.

The vector register name can be used with the `chess_storage` directive to force vector data to be stored in a particular vector register. For example:

```
aie::vector<int32,8> chess_storage(wr0) bufA;
aie::vector<int32,8> chess_storage(WR) bufB;
```

When upper case is used in the `chess_storage` directive, it means register files (for example, any of the four wr registers), whereas lower case in the directive means just a particular register (for example, wr0 in the previous code example) will be chosen.

Vector registers are a valuable resource. If the compiler runs out of available vector registers during code generation, then it generates code to spill the register contents into local memory and read the contents back when needed. This consumes extra clock cycles.

The name of the vector register used by the kernel during its execution is shown for vector load/store and other vector-based instructions in the kernel microcode. This microcode is available in the disassembly view in Vitis IDE. For additional details on Vitis IDE usage, see Using Vitis Unified IDE and Reports.

Many intrinsic functions only accept specific vector data types but sometimes not all values from the vector are required. For example, certain intrinsic functions only accept 512-bit vectors. If the kernel code has smaller sized data, one technique that can help is to use the `concat()` intrinsic to concatenate this smaller sized data with an undefined vector (a vector with its type defined, but not initialized).

For example, the `lmul8` intrinsic only accepts a `v16int32` or `v32int32` vector for its `xbuff` parameter. The intrinsic prototype is:

```
v8acc80 lmul8	(	aie::vector<int32,16> 	xbuff,
	int 	         xstart,
	unsigned int 	xoffsets,
	aie::vector<int32,8> 	     zbuff,
	int 	         zstart,
	unsigned int 	zoffsets
)
```

The `xbuff` parameter expects a 16 element vector (v16int32). In the following example, there is an eight element vector (`aie::vector<int32,8>`) rva. The `concat()` intrinsic is used to upgrade it to a 16 element vector. After concatenation, the lower half of the 16 element vector has the contents of rva. The upper half of the 16 element vector is uninitialized due to concatenation with the undefined `aie::vector<int32,8>` vector.

```
int32 a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
aie::vector<int32,8> rva = *((aie::vector<int32,8>*)a);
acc = lmul8(concat(rva,aie::zeros<int32,8>()),0,0x76543210,rvb,0,0x76543210);
```

For more information about how vector-based intrinsic functions work, refer to Vector Register Lane Permutations.
