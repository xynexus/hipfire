---
title: "Vector Registers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Registers"
toc_id: pVB56s_gVduYtPKGVlaGYg
content_id: z_45NWxakx1_1vMRsXJeMw
---

## Vector Registers

All vector intrinsic functions require the operands to be present in the AI Engine vector registers. The following table shows the set of vector registers and how smaller registers are combined to form larger registers.

| 128-bit | 256-bit | 512-bit | 1024-bit |  |
| --- | --- | --- | --- | --- |
| vrl0 | wr0 | xa | ya | N/A |
| vrh0 |  |  |  |  |
| vrl1 | wr1 |  |  |  |
| vrh1 |  |  |  |  |
| vrl2 | wr2 | xb | yd (MSBs) |  |
| vrh2 |  |  |  |  |
| vrl3 | wr3 |  |  |  |
| vrh3 |  |  |  |  |
| vcl0 | wc0 | xc | N/A | N/A |
| vch0 |  |  |  |  |
| vcl1 | wc1 |  |  |  |
| vch1 |  |  |  |  |
| vdl0 | wd0 | xd | N/A | yd (LSBs) |
| vdh0 |  |  |  |  |
| vdl1 | wd1 |  |  |  |
| vdh1 |  |  |  |  |

The underlying basic hardware registers are 128-bit wide and prefixed with the letter `v`. Two `v` registers can be grouped to form a 256-bit register prefixed with `w`. `wr`, `wc`, and `wd` registers are grouped in pairs to form 512-bit registers (`xa`, `xb`, `xc`, and `xd`). `xa` and `xb` form the 1024-bit wide `ya` register, while `xb` and `xd` form the 1024-bit wide `yd` register. This means the `xb` register is shared between the `ya` and `yd` registers. `xb` contains the most significant bits (MSBs) for both `ya` and `yd` registers.

The vector register name can be used with the `chess_storage` directive to force vector data to be stored in a particular vector register. For example:

```
aie::vector<int32,8> chess_storage(wr0) bufA;
aie::vector<int32,8> chess_storage(WR) bufB;
```

When upper case is used in the `chess_storage` directive, it means register files (for example, any of the four `wr` registers). Lower case in the `chess_storage` directive means the specified register (for example, `wr0` in the previous code example) is used.

This Chess directive can be replaced with a C++ compliant directive: `[[chess::storage(<reg>)]]`.

**Note:** `chess_storage`

Vector registers are a valuable resource. If the compiler runs out of available vector registers during code generation, then it generates code to spill the register contents into local memory and read the contents back when needed. This consumes extra clock cycles.

The name of the vector register used by the kernel during its execution is shown for vector load/store and other vector-based instructions in the kernel microcode. This microcode is available in the disassembly view in Vitis IDE. For additional details on Vitis IDE usage, see Using Vitis Unified IDE and Reports.

The `aie::vector` has member functions to support multiple operations on vectors. Some common operations include:

- **`insert()`:** Updates the contents of a region of the vector using the subvector and returns a reference to the updated vector.
- **`grow()`:** Returns a copy of the current vector in a larger vector. The contents of the new elements are undefined.
- **`grow_replicate()`:** The vector is replicated multiple times in the returned larger vector.
- **`extract()`:** Returns a subvector with the contents of a region of the vector.
- **`push()`:** Shifts all elements in the vector up and writes the given value into the first position of the vector (the element in the last position of the vector is lost).
- **`cast_to()`:** Reinterprets the current vector as a vector of the given type. The number of elements is automatically computed by the function.
- **`set()`:** Updates the value of the element on the given index.
- **`get()`:** Returns the value of the element on the given index.
- **`operator[]`:** Returns a constant or non-constant reference object to the element on the given index.

```
aie::vector<int16,16> wv;
aie::vector<int16,8> vv0,vv1;

// insert content of vv0 to lower half of wv
wv.insert(0,vv0);

// insert content of vv1 to higher half of wv
wv.insert(1,vv1);

// grow() returns a vector of size 32
// returned vector is assigned to xv
// lower 16 values in xv is assgined the values from wv
aie::vector<int16,32> xv=wv.grow<32>(0);

// wv is replicated 4 times using grow_replicate()
// vector of size 64 is assigned to xv2
aie::vector<int16,64> xv2=wv.grow_replicate<64>();

int a = 100;
aie::vector<int32,4> v1,v2;

// set 0th element to a
v1[0]=a;

// another method to set 0th element to a
v1.set(a,0);
a=v1.get(0);

// operator[] is preferred
// Element extraction may be merged
// with the underlying operation with no cost in cycles
auto v3 = aie::add(v1[3], v2);

// Element extraction and add in different cycles
v3 = aie::add(v1.get(3), v2);

// cast wv to complex type
aie::vector<cint16,8> cv=wv.cast_to<cint16>();

// extract higher half from cv
aie::vector<cint16,4> cv0=cv.extract<4>(/*idx=*/1);
```

**Note:** The updates replace the content of a part of the vector register. If a vector operation tries to access the updated content, the compiler re-arranges the operations to ensure it operates on correct data. This can impact the performance of the kernel.

`aie::vector`

`push`

```
aie::vector<int32,4> v1;
v1.push(100);
```

When defining and implementing a template function for an element type that is templated, standard C++ syntax requires that any variable of the template type calling its member functions be prepended `template` to the member function name. For example:

```
template<typename ELEMENT_TYPE> void func_test(){
  aie::vector<ELEMENT_TYPE,8> wv;
  aie::vector<ELEMENT_TYPE,16> xv=wv.template grow<16>(0);
  aie::vector<ELEMENT_TYPE,8> wv2=xv.template extract<8>(1);
}
```
