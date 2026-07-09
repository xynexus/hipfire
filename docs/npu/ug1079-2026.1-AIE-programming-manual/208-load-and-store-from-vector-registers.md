---
title: "Load and Store from Vector Registers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-from-Vector-Registers"
toc_id: FiffqklpdgcPSHG1NL9kHA
content_id: _i_BYDmAGgkTEmx0sWSMSw
---

#### Load and Store from Vector Registers

The compiler supports standard pointer de-referencing and pointer arithmetic for vectors. Post increment of the pointer is the most efficient form for scheduling. Loading vector registers does not require special intrinsic functions.

```
v8int32 * ptr_coeff_buffer = (v8int32 *)ptr_kernel_coeff;
v8int32 kernel_vec0 = *ptr_coeff_buffer++; // 1st 8 values (0 .. 7)
v8int32 kernel_vec1 = *ptr_coeff_buffer;   // 2nd 8 values (8 .. 15)
```
