---
title: "Data Reshaping"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Reshaping"
toc_id: YTpOcEriPL3PnFmkWFZPfA
content_id: qAKLYkrQsk1jQZ27To5y4A
---

## Data Reshaping

The AI Engine API provides operations to do the following:

- Change the location of the elements within a vector
- Extract subvector from larger vector, or
- Combine the elements from two or more vectors

The following APIs retrieve half of the vector by specific patterns:

- **`aie::filter_even`:** Returns a vector of half the size by: out = { v[0:step-1], v[2*step:3*step-1], ... }
- **`aie::filter_odd`:** Returns a vector of half the size by: out = { v[step:2*step-1], v[3*step:4*step-1], ... }
- **`aie::filter_odd`:** Note: The parameter `step` must be a power of two.

```
aie::vector<int32,16> xbuff;

// first parameter is the vector value
// second paramter is the step value
aie::vector<int32,8> result_e=aie::filter_even(xbuff,1);
aie::vector<int32,8> result_o=aie::filter_odd(xbuff,4);
```

The following figures show how elements are chosen by `aie::filter_even` and `aie::filter_odd`.

![ajl1634590991843.png](../assets/033-01-ajl1634590991843-png-f2c8df8c39ec.png)

*Figure 1. aie::filter_even*

![owq1634591366518.png](../assets/033-02-owq1634591366518-png-8ec54bab1583.png)

*Figure 2. aie::filter_odd*

`aie::select` can be used to choose elements from two vectors (or a scalar and a vector) to form a new vector. `aie::mask` is used for specifying the element to choose.

- **`aie::select`:** Choose elements by each bit of mask value: out[0] = {mask[0] == 0? a[0] : b[0], …}

```
aie::vector<int32,8> va,vb;
aie::mask<8> msk_value;

// vc[i]=(mask_value[i]==0)?va[i]:vb[i]
aie::vector<int32,8> vc=aie::select(va,vb,msk_value);
```

The AI Engine API provides the following functions to shift vector elements by a specific number but maintain the vector size and element values:

- **`aie::shuffle_down`:** Shift elements down by n: out = {v[n], v[n+1], …, v[Elems-1], undefined[0],…,undefined[n-1]}. Higher elements of output(out[n],out[n+1],…) are undefined. "Elems" is the vector size.
- **`aie::shuffle_up`:** Shift elements up by n: out = {undefined[0],…,undefined[n-1],v[0], v[1], …}. Lower elements of output(out[0],…,out[n-1]) are undefined.
- **`aie::shuffle_down_rotate`:** Rotate elements down by n: out = {v[n], v[n+1], …,v[Elems-1], v[0], …,v[n-1]}.
- **`aie::shuffle_up_rotate`:** Rotate elements up by n: out = {v[Elems-n], …, v[Elems-1], v[0], …,v[n-1]}.
- **`aie::shuffle_down_fill`:** Shift elements down by n and fill with elements from a second vector: out = {v[n], v[n+1], …, v[Elems-1], fill[0],…,fill[n-1]}. Higher elements of output(out[n],out[n+1],…) are filled with elements from second vector "fill"(fill[0],…,fill[n-1]).
- **`aie::shuffle_up_fill`:** Shift elements up by n and fill with elements from a second vector: out = {fill[Elems-n],…,fill[Elems-1],v[0], v[1], …}. Lower elements of output(out[0],…,out[n-1]) are filled with elements from second vector "fill"(fill[Elems-n],…,fill[Elems-1]).

```
aie::vector<int32,8> v,fill;

// v_d[0:4]=v[3:7]
// (v_d[5],v_d[6],v_d[7] are undefined)

auto v_d=aie::shuffle_down(v,3);

// v_u[3:7]=v[0:4]
// (v_u[0],v_u[1],v_u[2] are undefined)
auto v_u=aie::shuffle_up(v,3);

// vd_r[0:4]=v[3:7],vd_r[5:7]=v[0:2]
auto vd_r=aie::shuffle_down_rotate(v,3);

// vu_r[3:7]=v[0:4],vu_r[0:2]=v[5:7]
auto vu_r=aie::shuffle_up_rotate(v,3);

// v_d_fill[0:4]=v[3:7]
// v_d_fill[5:7]=fill[0:2]
auto v_d_fill=aie::shuffle_down_fill(v,fill,3);

// v_u_fill[3:7]=v[0:4]
// v_u_fill[0:2]=fill[5:7]
auto v_u_fill=aie::shuffle_up_fill(v,fill,3);
```

Reverse a vector using the following:

- **`aie::reverse`:** Reverse elements by: out = {v[Elems-1], …, v[1], v[0]}

```
// v_rev[0:7]=v[7:0]
aie::vector<int32,8> v_rev=aie::reverse(v);
```

The AI Engine API provides functions to shuffle two input vectors and combine them into the larger output vectors. The two vectors must be the same type and size. The return type is `std::pair`:

- **`aie::interleave_zip`:** Select and combine two vectors by this pattern and return a `std::pair` of vectors: out = { v1[0:step-1], v2[0:step-1], v1[step:2*step-1], v2[step:2*step-1], ...}
- **`aie::interleave_unzip`:** Select and combine two vectors by this pattern and return a `std::pair` of vectors: out = { v1[0:step-1], v1[2*step:3*step-1], ..., v2[0:step-1], v2[2*step:3*step-1], ..., v1[step:2*step-1], v1[3*step:4*step-1], ..., v2[step:2*step-1], v2[3*step:4*step-1], ... } Note: The parameter `step` must be a power of two.

```
alignas(aie::vector_decl_align) int32 data[16]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
aie::vector<int32,8> rva=aie::load_v<8>(data);
aie::vector<int32,8> rvb=aie::load_v<8>(data+8);
std::pair<aie::vector<int32,8>,aie::vector<int32,8>> rv=aie::interleave_zip(rva,rvb,4);
std::pair<aie::vector<int32,8>,aie::vector<int32,8>> rv2=aie::interleave_unzip(rva,rvb,2);

// rv.first=1 2 3 4 9 10 11 12
aie::print(rv.first,true,"rv.first=");

// rv.second=5 6 7 8 13 14 15 16
aie::print(rv.second,true,"rv.second=");

// rv2.first=1 2 5 6 9 10 13 14
aie::print(rv2.first,true,"rv2.first=");

// rv2.second=3 4 7 8 11 12 15 16
aie::print(rv2.second,true,"rv2.second=");
```

`rva`

`rvb`

`aie::vector<int32,8>`

```
aie::vector<int32,8> rva,rvb;
auto rv=aie::interleave_zip(rva,rvb,1);
aie::vector<cint32,8> cv=aie::concat(rv.first.cast_to<cint32>(),rv.second.cast_to<cint32>());

auto [uva,uvb]=aie::interleave_unzip(rv.first,rv.second,1);

// return value as true
bool ret=aie::equal(rva,uva) && aie::equal(rvb,uvb);
```

AI Engine

- `aie::transpose`

```
alignas(aie::vector_decl_align) int16 data[16]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
aie::vector<int16,16> va=aie::load_v<16>(data);

// Row=4 & Col=4
auto va_t=aie::transpose(va,4,4);

// print va_t
// va_t=1 5 9 13 2 6 10 14 3 7 11 15 4 8 12 16
aie::print(va_t,true,"va_t=");
```

AI Engine

- `aie::real`: Gather and return the real part of a complex value or vector.
- `aie::imag`: Gather and return the imaginary part of a complex value or vector.

```
alignas(aie::vector_decl_align) int16 data[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
cint16 c1={1,2};
aie::vector<cint16,8> vc1=aie::load_v<8>((cint16*)data);
int16 c1_re=aie::real(c1);
aie::vector<int16,8> vc1_imag=aie::imag(vc1);
printf("c1_re=%d\n",c1_re);
//Example output: c1_re=1

aie::print(vc1_imag,true,"vc1_imag=");
//Example output: vc1_imag=2 4 6 8 10 12 14 16
```
