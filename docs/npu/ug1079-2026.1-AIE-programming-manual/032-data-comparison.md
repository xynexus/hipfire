---
title: "Data Comparison"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Comparison"
toc_id: ~SrKRh8eNbM3aYW7MrjFpg
content_id: bP84OCmRGwkpOB15V~IBTA
---

## Data Comparison

AI Engine API provides vector comparison operations, including:

- `aie::eq`
- `aie::neq`
- `aie::le`
- `aie::lt`
- `aie::ge`
- `aie::gt`

The vector comparison operations compare two vectors element by element or compare one scalar with one vector, and return a special type `aie::mask`. Each bit of `aie::mask` corresponds to one elementary comparison result.

```
aie::vector<int32,16> v1,v2;

// compare each element
// true if v1[i]<v2[i]
aie::mask<16> msk_lt=aie::lt(v1,v2);

// set bit 0 to be true
msk_lt.set(0);

// set bit 1 to be false
msk_lt.clear(1);

/*element-wise selection
 *select v2[i] if msk_lt[i] is true
 */select v1[i] if msk_lt[i] is false;
aie::vector<int32,16> v_s=aie::select(v1,v2,msk_lt);
```

The following API compares two input vectors, or one input vector and one scalar value element by element, and returns the difference between them if the first input element is larger than the second input element. Otherwise, it returns 0 for that element:

- `aie::maxdiff`

```
aie::vector<int16,16> v1,v2;

// vc[i]=(v1[i]>v2[i])?(v1[i]-v2[i]):0
auto vc=aie::maxdiff(v1,v2);

// vc[i]=(v1[i]>1)?(v1[i]-1):0
auto vc2=aie::maxdiff(v1,(int16)1);

// vc[i]=(2>v2[i])?(2>v2[i]):0
auto vc3=aie::maxdiff((int16)2,v2);
```

The following APIs compare all the elements of the two input vectors and return a `bool` value:

- **`aie::equal`:** Returns whether all the elements of the two input vectors are equal. The vectors must have the same type and size.
- **`aie::not_equal`:** Returns whether some elements in the two input vectors are not equal. The vectors must have the same type and size.

The following APIs are provided to choose the max or min value of two vectors (or one scalar and one vector) element by element. The type of the scalar and vectors must be same:

- `aie::max`
- `aie::min`

```
// vc[i]=max(v1[i], v2[i])
aie::vector<int32,16> vc=aie::max(v1,v2);
```
