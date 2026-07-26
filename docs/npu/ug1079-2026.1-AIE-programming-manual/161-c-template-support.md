---
title: "C++ Template Support"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/C-Template-Support"
toc_id: o15wkJTHBWHtHwu_yyvAHA
content_id: jvyQMqSsxd8RATM3o4eykw
---

## C++ Template Support

A template is a powerful tool in C++. By passing the data type as a parameter, you eliminate the need to rewrite code to support different data types. Templates are expanded at compile time, like macros. The difference is that the compiler performs type checking before template expansion.

The source code contains template functions and class definitions, but the compiled code can contain multiple copies of same function or class. Type parameters, non-type parameters, default arguments, scalar parameters, and template parameters can be passed to a template, where the compiler instantiates the function or class accordingly.

- Support for general C++ template features.
- Supported data types (T) and connection types between kernels: `int8` `uint8` `int16` `uint16` `cint16` `int32` `uint32` `cint32` `int64` `uint64` `float` `float16` `bfloat8` `float8` `mx9`

  - `int8`
  - `uint8`
  - `int16`
  - `uint16`
  - `cint16`
  - `int32`
  - `uint32`
  - `cint32`
  - `int64`
  - `uint64`
  - `float`
  - `float16`
  - `bfloat8`
  - `float8`
  - `mx9`
- The compiler does not support pre-compiled headers for template kernels.

**Note:** Important:

`acc48`

`cacc48`

#### Function Templates

Function template source code defines a generic function fr use with different data types. Example function template:

```
// add.h

template<typename ELEMENT_TYPE, int FACTOR, size_t NUM_SAMPLES> void add(adf::input_buffer<ELEMENT_TYPE> &in, adf::output_buffer<ELEMENT_TYPE> &out);
```

```
// add.cpp

#include <aie_api/aie.hpp>
template<typename ELEMENT_TYPE, int FACTOR, size_t NUM_SAMPLES> void add(adf::input_buffer<ELEMENT_TYPE> &in, adf::output_buffer<ELEMENT_TYPE> &out){
  auto inIter=aie::begin(in);
  auto outIter=aie::begin(out);
  for (int i=0; i<NUM_SAMPLES; i++){
    ELEMENT_TYPE value = *inIter++;
    value += FACTOR;
    *outIter++=value;
  }
}
```

```
// graph.h
mygraph()
{
    k[0] = adf::kernel::create(add<int32, 6, 8>);
    k[1] = adf::kernel::create(add<int16, 3, 8>);
    for (int i=0; i<NUM_KERNELS; i++)
    {
        adf::runtime<adf::ratio>(k[i]) = 0.3;
        adf::source(k[i]) = "src/add.cpp";
    }

    adf::connect(in[0], k[0].in[0]);
    adf::connect(k[0].out[0], out[0]);

    adf::connect(in[1], k[1].in[0]);
    adf::connect(k[1].out[0], out[1]);

    adf::dimensions(k[0].in[0])={8};
    adf::dimensions(k[0].out[0])={8};
    adf::dimensions(k[1].in[0])={8};
    adf::dimensions(k[1].out[0])={8};
}
```

where:

- add.h defines a template `add()` function.
- add.cpp defines the code for the template `add()` function.
- graph.h uses the template `add()` function within `mygraph` class.

#### Class Templates

Like function templates, class templates are useful when a class defines an object that is independent of a specific data type. Example class template:

```
// fir.h
...
template<size_t NUM_COEFFS, typename ELEMENT_TYPE> class FIR
{
private:
    ELEMENT_TYPE (&coeffs)[NUM_COEFFS];
    ELEMENT_TYPE tapDelayLine[NUM_COEFFS];
    uint32 numSamples;

public:
    FIR(ELEMENT_TYPE(&coefficients)[NUM_COEFFS], uint32 samples);

    void filter(input_buffer<ELEMENT_TYPE> &in, output_buffer<ELEMENT_TYPE> &out);

    //user needs to write this function to register necessary info
    static void registerKernelClass()
    {
        REGISTER_FUNCTION(FIR::filter);
        REGISTER_PARAMETER(coeffs);
    }
}
```

```
// fir.cpp
...
template<size_t NUM_COEFFS, typename ELEMENT_TYPE> FIR<NUM_COEFFS, ELEMENT_TYPE>::FIR(ELEMENT_TYPE(&coefficients)[NUM_COEFFS], uint32 samples):coeffs(coefficients)
{
    ...
}

template<size_t NUM_COEFFS, typename ELEMENT_TYPE> void FIR<NUM_COEFFS,ELEMENT_TYPE>::filter(adf::input_buffer<ELEMENT_TYPE> &in, adf::output_buffer<ELEMENT_TYPE> &out)
{
    ...
}
```

```
// graph.h
...
mygraph()
{
    k1 = adf::kernel::create_object<FIR<12, int32>>(std::vector<int>({ 180, 89, -80, -391, -720, -834, -478, 505, 2063, 3896, 5535, 6504 }), 8);
    adf::runtime<adf::ratio>(k1) = 0.1;
    adf::source(k1) = "src/fir.cpp";
    adf::headers(k1) = { "src/fir.h" };

    k2 = adf::kernel::create_object<FIR<15, int32>>(std::vector<int>({ -21, -249, 319, -78, -511, 977, -610, -844, 2574, -2754, -1066, 18539, 0, 0, 0 }), 8);
    adf::runtime<ratio>(k2) = 0.1;
    adf::source(k2) = "src/fir.cpp";
    adf::headers(k2) = { "src/fir.h" };
    ...
}
```

where:

- fir.h defines a class template where class `FIR` is declared.
- fir.cpp contains class `FIR` implementation and the class `FIR` member function `filter` implementation.
- graph.h demonstrates the template class `FIR` instantiation within the `mygraph` class.
