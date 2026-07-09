---
title: "Prepare the Kernels"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Prepare-the-Kernels"
toc_id: IUb9nP62plIhnjVaZbjAkQ
content_id: Wo1JGHGJTF407WxK6yRo9Q
---

## Prepare the Kernels

Kernels are computation functions that form the fundamental building blocks of the data flow graph specifications. Kernels are declared as ordinary C/C++ functions that return `void` and can use special data types as arguments. Define each kernel in its own source file. This organization is useful for reusability and faster compilation. Furthermore, the kernel source files must include all relevant header files to allow for independent compilation.

**Note:** AI Engine

AI Engine

- `#include "aie_api/aie.hpp"`

Declare all kernel function prototypes in a header file (for example, `kernels.h`) used in the graph. An example is as follows.

```
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

void simple(adf::input_buffer<int16> & __restrict in,
            adf::output_buffer<int16> & __restrict out);

#endif
```

The `#ifndef` and `#endif` ensure the include file loads only once. This is good C/C++ practice.

The kernels.cc file is the implementation file for the simple function. The kernel implementation uses two `int` 16 variables `in_t` and `out_t`. These variables point to the underlying values of the input and output buffer respectively using the `data()` member function.

```
/* A simple kernel */
#include <adf.h>
#include "kernels.h"

void simple(adf::input_buffer<int16> & __restrict in,
            adf::output_buffer<int16> & __restrict out)
{
    int16* __restrict in_t = in.data();
    int16* __restrict out_t = out.data();
    for (unsigned i=0; i<NUM_SAMPLES; i++) {
        *out_t++ = 1 + *in_t++;
    }
}
```

Related concepts

Input and Output Buffers

Streaming Data API
