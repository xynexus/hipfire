---
title: "AI Engine API Overview"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-API-Overview"
toc_id: 4ANNBacchQU7FuJhtzUleQ
content_id: QfaaIRdlw8hOKEg8dfL9_A
---

## AI Engine API Overview

The AI Engine API is a portable programming interface for AI Engine accelerators. The AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)) details which API is supported only on a specific architecture. The APIs are implemented as a C++ header-only library that provides types and operations for translation into efficient low-level intrinsics. The API also provides higher-level abstractions such as iterators.

Usually, kernel source code requires the following two header files:

- **`aie_api/aie.hpp`:** AI Engine main entry point.
- **`aie_api/aie_adf.hpp`:** Graph buffer and stream interfaces.

When profiling is enabled, the AI Engine API provides a helper file that prints `aie::vector` and `aie::mask` values in simulation.

- `aie_api/utils.hpp`: `aie::print` and `aie::print_matrix` functions are provided.

To support operator overloading on some operations, include header file `aie_api/operators.hpp` and use namespace `aie::operators`. For additional information, see Operator Overloading.

A typical example of using the API for an AI Engine kernel is as follows.

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
using namespace adf;
void vec_incr(input_buffer<int32>& data, output_buffer<int32>& out) {

  //set all elements to 1
  aie::vector<int32,16> vec1=aie::broadcast(1);

  auto inIter=aie::begin_vector<16>(data);
  auto outIter=aie::begin_vector<16>(out);

  for(int i=0;i<16;i++) chess_prepare_for_pipelining {
    aie::vector<int32,16> vdata=*inIter++;

    //print vector in a line
    aie::print(vdata,true,"vdata=");

    //print vdata as a 2x8 matrix
    aie::print_matrix(vdata,8,"vdata matrix=");

    //increment each element in vdata by 1
    aie::vector<int32,16> vresult=aie::add(vdata,vec1);

    *outIter++=vresult;
  }
}
```

```
vdata=0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
vdata matrix=0 1 2 3 4 5 6 7
             8 9 10 11 12 13 14 15
vdata=16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
vdata matrix=16 17 18 19 20 21 22 23
             24 25 26 27 28 29 30 31
vdata=32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
vdata matrix=32 33 34 35 36 37 38 39
             40 41 42 43 44 45 46 47
vdata=48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
vdata matrix=48 49 50 51 52 53 54 55
             56 57 58 59 60 61 62 63
vdata=64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
vdata matrix=64 65 66 67 68 69 70 71
             72 73 74 75 76 77 78 79

...

vdata=208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
vdata matrix=208 209 210 211 212 213 214 215
             216 217 218 219 220 221 222 223
vdata=224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
vdata matrix=224 225 226 227 228 229 230 231
             232 233 234 235 236 237 238 239
vdata=240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255
vdata matrix=240 241 242 243 244 245 246 247
             248 249 250 251 252 253 254 255
```

The following sections cover vector data type and operations.
