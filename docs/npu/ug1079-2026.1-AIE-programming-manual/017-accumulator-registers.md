---
title: "Accumulator Registers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Accumulator-Registers"
toc_id: XuSk6riWgPRzVXtjlC1ihA
content_id: eTZczaFgvDMzEHMzXPaudQ
---

## Accumulator Registers

The accumulation registers are 384 bits wide and can be viewed as eight vector lanes of 48 bits each. The idea is to have 32-bit multiplication results and accumulate over those results without bit overflows. The 16 guard bits allow up to 216 accumulations. The output of fixed-point vector MAC and MUL intrinsic functions is stored in the accumulator registers. The following table shows the set of accumulator registers and how smaller registers are combined to form large registers.

| 384-bit | 768-bit |
| --- | --- |
| aml0 | bm0 |
| amh0 |  |
| aml1 | bm1 |
| amh1 |  |
| aml2 | bm2 |
| amh2 |  |
| aml3 | bm3 |
| amh3 |  |

The accumulator registers are prefixed with the letters 'am'. Two of them are aliased to form a 768-bit register that is prefixed with 'bm'. The `to_vector` operation moves a value from an accumulator register to a vector register with any required shifting and rounding.

```
aie::accum<acc48,8> acc;

// shift right 10 bits from accumulator
aie::vector<int32,8> res=acc.to_vector<int32>(10);register to vector register
```

The `from_vector` operation is used to move a value from a vector register to an accumulator register with upshifting.

```
aie::vector<int32,8> v;
aie::accum<acc48,8> acc;

// shift left 10 bits
// from vector register to accumulator register
acc.from_vector(v, 10);
aie::print(acc,true,"acc value=");
```

Besides `from_vector()` and `to_vector()` functions, `aie::accum` class has the following member functions similar to `aie::vector`:

- **`insert()`:** Updates the contents of a region of the accumulator using the values in the given native subaccumulator and returns a reference to the updated accumulator.
- **`grow()`:** Returns a copy of the current accumulator in a larger accumulator. The value of the new elements is undefined.
- **`extract()`:** Returns a subaccumulator with the contents of a region of the accumulator.
- **`cast_to()`:** Reinterprets the current accumulator as an accumulator of the given type. The number of elements is automatically computed by the function.

```
int32 data[8]={1,2,3,4,5,6,7,8};
aie::vector<int32,8> v=aie::load_v<8>(data);
aie::accum<acc48,8> acc;

// shift left 0 bits
acc.from_vector(v, 0);

aie::accum<acc48,16> acc2=acc.grow<16>();
aie::print(acc2,true,"acc2 value=");


acc2.insert(1,acc);
aie::print(acc2,true,"acc2 value=");


// extract lower part, and cast to cacc48
aie::accum<cacc48,4> cacc1=acc2.extract<8>(0).cast_to<cacc48>();
aie::print(cacc1,true,"cacc1 value=");
```

```
aie::vector<float,8> readincr_v<8>(input_cascade<accfloat> * str);
aie::vector<cfloat,4> readincr_v<4>(input_cascade<caccfloat> * str);
void writeincr(output_cascade<accfloat>* str, aie::vector<float,8> value);
void writeincr(output_cascade<caccfloat>* str, aie::vector<cfloat,4> value);
```

For additional details on stream APIs, see Streaming Data API. For additional details on buffers, see Input and Output Buffers.
