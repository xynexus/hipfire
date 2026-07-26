---
title: "Initialization"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Initialization"
toc_id: _52aTQvvhHbGvYESFQv7MQ
content_id: Nh0Uc383XfGI0~NwiT66Lw
---

### Initialization

You can initialize vector registers with elements undefined as all zeros, using data from local memory or using part of the values set from another register, or a combination of the above.

```
// contents undefined
aie::vector<int32,8> uv;

// all 0's
aie::vector<int32,8> nv = aie::zeros<int32,8>();

// all values are 100
aie::vector<int32,8> bv = aie::broadcast<int32,8>(100);

// concatenate vectors into larger vector
aie::vector<int32,16> cv=aie::concat(nv,bv);

/*re-interpret "reals" as "aie::vector<int32,8>*" pointer and load value from it
 */
aie::vector<int32,8> iv = *(reinterpret_cast<aie::vector<int32,8>*>(reals));

// load value pointed by "reals"
aie::vector<int32,8> iv2 = aie::load_v<8>((int32*)reals);

// create a new 512-bit vector with lower 256-bit set with "iv"
aie::vector<int32,16> sv = iv.grow<16>(0);
```

The following list describes the functions used in the previous example:

- **`aie::zeros`:** Creates vector with all elements 0.
- **`aie::broadcast`:** Sets all elements of vector to a value.
- **`aie::concat`:** Concatenates multiple vectors or accumulators with same type and size into a larger vector or accumulator.

The vector and accumulator classes have the member function `grow()` that creates a larger vector where only one part is initialized with the subvector (`iv` in above example) and the other parts are undefined. Here the function parameter `idx` indicates the location of the subvector within the output vector.

The `static` keyword applies to the vector data type as well. When applied, the default value is zero when not initialized and the value is kept between graph run iterations.
