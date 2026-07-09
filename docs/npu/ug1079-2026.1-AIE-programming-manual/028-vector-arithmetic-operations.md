---
title: "Vector Arithmetic Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Arithmetic-Operations"
toc_id: 9jXuDIS39XDR88R04F6_OQ
content_id: i4IHY8f~_DXsZ4z68t8rcQ
---

## Vector Arithmetic Operations

The AI Engine API supports basic arithmetic operations on two vectors, or on a scalar and a vector (operation on the scalar and each element of the vector). It also supports addition or subtraction of a scalar or a vector on an accumulator. Additionally, it supports multiply-accumulate (MAC). These operations include the following:

- **`aie::mul`:** Returns an accumulator with:The element-wise multiplication of two vectors, or The product of a vector and a scalar value.
- **`aie::negmul`:** Returns an accumulator with:The negative of the element-wise multiplication of two vectors, or The negative of the product of a vector and a scalar value.
- **`aie::mac`:** Multiply-add on vectors (or scalar) and accumulator.
- **`aie::msc`:** Multiply-sub on vectors (or scalar) and accumulator.
- **`aie::add`:** Returns a vector with:The element-wise addition of two vectors, or Adds a scalar value to each component of a vector, or Adds scalar or vector on accumulator.
- **`aie::sub`:** Returns a vector with:The element-wise subtraction of two vectors, or Subtracts a scalar value from each component of a vector, or Subtract scalar or vector on accumulator.
- **`aie::saturating_add`:** Returns a vector with:The element-wise addition of two vectors, or adds a scalar value to each component of a vector. `aie::saturating_add` supports saturation mode.
- **`aie::saturating_sub`:** Returns a vector with:The element-wise subtraction of two vectors, or Subtract a scalar value from each component of a vector `aie::saturating_sub` supports saturation mode.

The vectors and accumulator must have the same size and their types must be compatible. For example:

```
aie::vector<int32,8> va,vb;
aie::accum<acc64,8> vm=aie::mul(va,vb);
aie::accum<acc64,8> vm2=aie::mul((int32)10,vb);
aie::vector<int32,8> vsub=aie::sub(va,vb);
aie::vector<int32,8> vadd=aie::add(va,vb);

// vsub2[i]=va[i]-10
aie::vector<int32,8> vsub2=aie::sub(va,(int32)10);

// vsub2[i]=10+va[i]
aie::vector<int32,8> vadd2=aie::add((int32)10,va);

aie::accum<acc64,8> vsub_acc=aie::sub(vm,(int32)10);
aie::accum<acc64,8> vsub_acc2=aie::sub(vm,va);
aie::accum<acc64,8> vadd_acc=aie::add(vm,(int32)10);
aie::accum<acc64,8> vadd_acc2=aie::add(vm,vb);

aie::accum<acc64,8> vmac=aie::mac(vm,va,vb);
aie::accum<acc64,8> vmsc=aie::msc(vm,va,vb);

// scalar and vector can switch placement
aie::accum<acc64,8> vmac2=aie::mac(vm,va,(int32)10);

// scalar and vector can switch placement
aie::accum<acc64,8> vmsc2=aie::msc(vm,(int32)10,vb);
```

`aie::add`

`aie::saturating_add`

```
aie::tile::current().set_saturation(aie::saturation_mode::saturate);

aie::vector<int16, 16> v1 = aie::broadcast<int16, 16>(20000);
aie::vector<int16, 16> v2 = aie::broadcast<int16, 16>(20000);
aie::vector<int16, 16> result1 = aie::add(v1, v2);
printf("vector + vector = %d\n", result1.get(0));
//output: vector + vector = -25536

aie::vector<int16, 16> result_sat = aie::saturating_add(v1, v2);
printf("vector + vector saturate= %d\n", result_sat.get(0));
//output: vector + vector saturate= 32767
```

The AI Engine API supports arithmetic operations on a vector or accumulation of element-wise square, including the following:

- **`aie::abs`:** Computes the absolute value for each element in the given vector.
- **`aie::abs_square`:** Computes the absolute square of each element in the given complex vector.
- **`aie::conj`:** Computes the conjugate for each element in the given vector of complex elements.
- **`aie::neg`:** For vectors with signed types, returns a vector whose elements are the same as in the given vector but with the sign flipped. If the input type is unsigned, the input vector is returned.
- **`aie::mul_square`:** Returns an accumulator of the requested type with the element-wise square of the input vector.
- **`aie::mac_square`:** Returns an accumulator with the addition of the given accumulator and the element-wise square of the input vector.
- **`aie::msc_square`:** Returns an accumulator with the subtraction of the given accumulator and the element-wise square of the input vector.

The vector and the accumulator must have the same size and their types must be compatible. Following is an example:

```
aie::vector<int16,16> va;
aie::vector<cint16,16> ca;
aie::vector<int16,16> va_abs=aie::abs(va);
aie::vector<int32,16> ca_abs=aie::abs_square(ca);
aie::vector<cint16,16> ca_conj=aie::conj(ca);
aie::vector<int16,16> va_neg=aie::neg(va);
aie::accum<acc32,16> va_sq=aie::mul_square(va);

aie::vector<int32,8> vc,vd;
aie::accum<acc64,8> vm=aie::mul(vc,vd);

// vmac3[i]=vm[i]+vc[i]*vc[i];
aie::accum<acc64,8> vmac3=aie::mac_square(vm,vc);

// vmsc3[i]=vm[i]-vd[i]*vd[i];
aie::accum<acc64,8> vmsc3=aie::msc_square(vm,vd);
```

Operands can also be supported pre-multiplication operations. On some AI Engine architectures certain operations can be collapsed with the multiplication into a single instruction. Following is an example:

```
aie::vector<cint16,16> ca,cb;
aie::accum<cacc48,16> acc=aie::mul(aie::op_conj(ca),aie::op_conj(cb));
```

For details about pre-multiplication operations, see Pre-Multiplication Operations.
