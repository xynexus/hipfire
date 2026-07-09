---
title: "Casting and Datatype Conversion"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Casting-and-Datatype-Conversion"
toc_id: ExR4le47xvZy3XVhpD8SeA
content_id: kHC_ucRG_0cchw3LyDn8ww
---

## Casting and Datatype Conversion

Casting intrinsic functions (`as_[Type]()`) allow casting between vector types or scalar types of the same size. The casting can work on accumulator vector types too. Generally, using the smallest data type possible reduces register spillage and improve performance. For example, if a 48-bit accumulator (acc48) meets the design requirements, use that instead of a larger 80-bit accumulator (acc80).

**Note:** `acc80`

You can also use standard C casts. This works identically in almost all cases, as shown in the following example.

```
aie::vector<int16,8> iv;
aie::vector<cint16,4> cv=as_v4cint16(iv);
aie::vector<cint16,4> cv2=*(aie::vector<cint16,4>*)&iv;
aie::accum<acc80,8> cas_iv;
aie::accum<acc48,8> cas_cv=as_v8cacc48(cas_iv);
```

There is hardware support built-in for floating-point to fixed-point (`float2fix()`) and fixed-point to floating-point (`fix2float()`) conversions. For example, the fixed-point square root, inverse square root, and inverse are implemented with floating-point precision and the `fix2float()` and `float2fix()` conversions are used before and after the function.

**Note:** AI Engine

Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine))

This example uses the scalar engine because the square root and inverse functions are not vectorizable. You can verify this by looking at the function prototype's input data types:

```
float _sqrtf(float a) //scalar float operation
int sqrt(int a,...) //scalar integer operation
```

Note that the input data types are scalar types (int) and not vector types (vint).

either the vector or scalar engines can handle the conversion functions (`fix2float`, `float2fix`), depending on the function called. Note the difference in data return type and data argument types:

```
float fix2float(int n,...) //Uses the scalar engine
v8float fix2float(aie::vector<int32,8> ivec,...) //Uses the vector engine
```

**Note:** `float2fix`

`float2fix_safe`

`float2fix_fast`

`float2fix_safe`

`FLOAT2FIX_FAST`

`float2fix`

`float2fix_fast`
