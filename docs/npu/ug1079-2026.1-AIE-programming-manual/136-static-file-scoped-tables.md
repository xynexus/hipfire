---
title: "Static File-Scoped Tables"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Static-File-Scoped-Tables"
toc_id: SjVBvlMhWjQB3RhIG2duHg
content_id: BMMr~2TCz3LrE8kYrOX0Zw
---

### Static File-Scoped Tables

Kernel functions can use private, read-only data structures. These structures are accessed as file-scoped variables. The compiler allocates a limited amount of static heap space for such data. As an example, consider the following header file (user_parameter.h).

```
#ifndef USER_PARAMETER_H
#define USER_PARAMETER_H

#include <adf.h>
static int16 lutarray[8] = {1,2,3,4,5,6,0,0} ;

#endif
```

This header file can be included in the kernel source file and the lookup table can be accessed inside a kernel function directly. The `static` modifier ensures that the array definition is local to this file. The AI Engine compiler allocates this array in static heap space for the processor where this kernel is used.

```
#include <aie_api/aie.hpp>
#include "user_parameter.h"

void simple_lut(adf::input_buffer<int16> &in, adf::output_buffer<int16> &out){
  aie::vector<int16,32> sbuff;
  aie::vector<int16,8> coeffs=aie::load_v<8>((int16*)lutarray);
  auto inIter=aie::begin_vector<32>(in);
  sbuff=*inIter++;
  auto acc = aie::sliding_mul<8,16>(coeffs, 0, sbuff, 0);
  auto outIter=aie::begin_vector<8>(out);
  *outIter++=acc.to_vector<int16>(0);
}
```
