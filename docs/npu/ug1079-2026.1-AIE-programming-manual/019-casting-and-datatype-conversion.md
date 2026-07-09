---
title: "Casting and Datatype Conversion"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Casting-and-Datatype-Conversion"
toc_id: WCZENG3jt8_BE3HJHpuWMA
content_id: t519Wjs0nN9DULCD8DXmaw
---

## Casting and Datatype Conversion

Casting functions (`aie::vector_cast<DstT>(const Vec& v)` and `aie::vector.cast_to<DstT>()`) allow value casting between vector types with the same size in bits. Accumulator vector types have the casting function `aie::accum.cast_to<DstT>()`. Generally, using the smallest data type possible reduces register spillage and improve performance. For example, if a 48-bit accumulator (acc48) meets the design requirements, use that instead of a larger 80-bit accumulator (acc80).

**Note:** `acc80`

```
aie::vector<int16,8> iv;
aie::vector<cint16,4> cv=iv.cast_to<cint16>();
aie::vector<cint16,4> cv2=aie::vector_cast<cint16>(iv);
aie::accum<cacc48,4> acc=aie::mul(cv,cv2);
aie::accum<acc64,4> acc2=acc.cast_to<acc64>();
```

You can also use standard C++ casts, but the recommended ways of reading vectors from a buffer are as follows:

- Use `aie::load_v` and increment the scalar pointer by the number of elements in the vector.
- Using vector iterators.

Additional details about `aie::load_v` and iterators are covered in the following sections.

```
int16 coeff_buffer[16];

// cast to int32 and load
aie::vector<int32,8> coeff=aie::load_v<8>((int32*)coeff_buffer);

// create vector<int16,8> iterator
auto it = aie::begin_vector<8>(coeff_buffer);

// read first vector<int16,8>
aie::vector<int16,8> vec0=*it++;

// read second vector<int16,8>
aie::vector<int16,8> vec1=*it;
```

The API supports floating-point to fixed-point (`to_fixed()`) and fixed-point to floating-point (`to_float()`) conversions. The conversion functions (`to_float()` and `to_fixed()`) can be handled by either the vector or scalar engines depending on the function called.

**Note:** AI Engine

Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine))

```
int a=48;
float f1=aie::to_float(a);

// first argument is the value of f1
// second argument is the position of input decimal point
float f2=aie::to_float(a,2);

int b1=aie::to_fixed(f1);

// first argument is the value of f1
// second argument is the position of output decimal point
int b2=aie::to_fixed(f1,2);
aie::vector<float,32> fv;
aie::vector<int32,32> iv=aie::to_fixed<int32>(fv,2);
```

The vector engine offers two implementations of the `to_fixed()` functions: `safe`, which is the default, and `fast`. The `safe` implementation offers more strict data type checks. For the `fast` implementation, you can run the `v++ -c --mode aie` command with the `--fastmath` option, which lets `to_fixed()` choose the `fast` implementation.

The scalar engine offers three implementations for the `to_fixed()` function for floating point scalar operations:

| Compiler Option | Description |
| --- | --- |
| --fastmath | The to_fixed and floating-point comparison has two implementations. Fast to_fixed can have wrong results if the shift amount is greater than 1 while the safe version requires more cycles to complete. Fast floating-point comparison gives wrong results with +0 and -0 while the safe version is correct but takes more cycles. |
| --fast-floats | Floating point scalar operations, like add, subtract, multiply, and compare, can either be mapped on vector floating point or on softfloat lib. By default, softfloat lib implementation is chosen (--fast-floats=false). This takes quite a few cycles because it is emulated, but the vector floating-point processor can be used at the same time. |
| --fast-nonlinearfloats | Floating point non-linear scalar operations, like sine/cosine, sqrt, and inv, can either be mapped on scalar non-linear function or on runtime lib (math.c). By default, runtime lib implementation is chosen (--fast-nonlinearfloats=false) which takes quite a few cycles because it is emulated. |

You can move data from vector to accumulator using `aie::accum.from_vector()` or from accumulator to vector using `aie::accum.to_vector<DstT>()` with shifting and rounding. The following example shows this.

```
aie::vector<int16,16> v;
aie::accum<acc32,16> acc;
acc.from_vector(v, 0);

aie::accum<acc32,16> acc2;
aie::vector<int16,16> v2;
v2 = acc2.to_vector<int16>(15);
```

- `aie::pack`: Returns a vector by converting each element into half number of bits.
- `aie::unpack`: Returns a vector by converting each element into twice the number of bits.

```
aie::vector<int16,16> data;
aie::print(data,true,"data=");
aie::vector<int8,16> data_smaller=data.pack();
aie::vector<int16,16> data_larger=data_smaller.unpack();
aie::print(data_smaller,true,"smaller data=");
aie::print(data_larger,true,"larger data=");
//Example output:
//data=0 1 2 -32768 -4 -5 -6 32767 3 4 126 130 -8 -9 -300 0
//smaller data=0 1 2 0 -4 -5 -6 -1 3 4 126 -126 -8 -9 -44 0
//larger data=0 1 2 0 -4 -5 -6 -1 3 4 126 -126 -8 -9 -44 0
```
