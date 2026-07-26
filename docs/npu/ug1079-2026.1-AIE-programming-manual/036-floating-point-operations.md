---
title: "Floating-Point Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Floating-Point-Operations"
toc_id: G1JBUe_rWUGCnGaaKoPWQA
content_id: flJTzjMACA~orJlx3Wf06Q
---

## Floating-Point Operations

The scalar unit floating-point hardware support includes square root, inverse square root, inverse, absolute value, minimum, and maximum. It supports other floating-point operations through emulation. The `softfloat` library must be linked in for test benches and kernel code using emulation. Use the single-precision float version for math library functions. For example, use `expf()` instead of `exp()`.

The AI Engine vector unit provides eight lanes of single-precision floating-point multiplication and accumulation. The unit reuses the vector register files and permute network of the fixed-point data path. You can perform only one fixed-point or floating-point vector instruction per cycle.

Floating-point MACs have a two-cycle latency. Use two accumulators in a ping-pong manner to improve performance, and let the compiler schedule a MAC each cycle.

```
aie::accum<accfloat,8> acc1=aie::zeros<accfloat,8>();
aie::accum<accfloat,8> acc2=aie::zeros<accfloat,8>();
aie::vector<float,8> va,vb;
auto ita=aie::begin_vector<8>(data1);
auto itb=aie::begin_vector<8>(data2);
auto ito=aie::begin(out);
for(int i=0;i<32;i++)
chess_prepare_for_pipelining
{
  va=*ita++;
  vb=*itb++;
  acc1=aie::mac(acc1,va,vb);
  va=*ita++;
  vb=*itb++;
  acc2=aie::mac(acc2,va,vb);
}
auto acc=aie::add(acc1,acc2);
auto sum=aie::reduce_add(acc.to_vector<float>(0));
*ito=(float)sum;
```

There is a scalar float divide function `aie::div`, but there is no divide vector function at this time. However, you can implement vector division using an inverse and multiply, as shown in the following example.

```
aie::vector<float,8> vf_div,vf1,vf1_inv,vf2;
vf1_inv=aie::inv(vf1);
vf_div=aie::mul(vf1_inv,vf2);
```

The following API functions support operations on a scalar or all elements of a vector:

- `aie::inv`
- `aie::sqrt`
- `aie::invsqrt`
- `aie::sin`
- `aie::cos`
- `aie::sincos`: Same as sin and cos, but performs both operations and returns a `std::pair` of vectors of result values. The first vector contains the sine values, the second contains the cosine values
- `aie::sincos_complex`: Similar to `sincos`, but returns both values as real and imaginary parts of a complex number. The real part contains `cos`, and the imaginary part contains `sin`.

For `aie::sin`, `aie::cos`, and `aie::sincos`, the input can either be a float value in radians or an integer. The floating-point range is [-Pi, Pi]. Integer values are handled as a fixed-point input value in Q1.31 format scaled with 1/Pi (input value 2^31 corresponds to Pi). This case uses only the upper 20 bits of the input value. According to input type, the returned value is either a float or a signed Q0.15 fixed-point format.

```
alignas(aie::vector_decl_align) static int16 dds_stored [16]={...};
aie::vector<cint16,8> dds=aie::load_v<8>((cint16*)dds_stored);
int32 phase_in;
auto [sin_,cos_] = aie::sincos(phase_in << 14) ;
cint16 scvalues={cos_,sin_};
dds.push(scvalues);
```
