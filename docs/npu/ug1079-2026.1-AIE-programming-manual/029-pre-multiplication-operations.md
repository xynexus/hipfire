---
title: "Pre-Multiplication Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Pre-Multiplication-Operations"
toc_id: vFD1dMI1p1teK6ZVN~48Eg
content_id: ZFWerLogVUcyK4tQiJ75AQ
---

### Pre-Multiplication Operations

AI Engine architectures offer multiplication instructions that can perform additional operations on the input arguments. Instead of adding one variant for each possible combination, the AI Engine API offers types that can wrap an existing vector, accumulator of element reference and be passed into the multiplication function. Then the API merges the operations into a single instruction or apply the operation on the vector before the multiplication, depending on the hardware support.

The pre-multiplication operations are special empty operations that simply return the original objects they wrap. Pre-multiplication operations include the following:

- `op_abs`
- `op_add`
- `op_conj`
- `op_max`
- `op_min`
- `op_none`
- `op_sub`
- `op_sign`
- `op_zero`

The following example performs an element-wise multiplication of the absolute of vector `a` and the conjugate of vector `b`.

```
aie::accum<cacc48,16> foo(aie::vector<int16,16> a, aie::vector<cint16,16> b){
  aie::accum<cacc48,16> ret;
  ret = aie::mul(aie::op_abs(a), aie::op_conj(b));
  return ret;
}
```

The following code performs accumulation with dynamic zeroization of the accumulator and dynamic sign of the data vectors:

```
alignas(aie::vector_decl_align) int16 data[16]={0,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};
aie::vector<int16,16> a=aie::load_v<16>(data);
aie::vector<int16,16> b=aie::load_v<16>(data);
aie::accum<acc48,16> ret;
bool is_zero=true;
bool is_sign=true;
ret = aie::mac(aie::op_zero(ret,is_zero),aie::op_sign(a,is_sign), aie::op_sign(b,is_sign));

// print the "ret" values
aie::print(ret,true,"ret=");
```
