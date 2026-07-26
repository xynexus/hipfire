---
title: "Kernel Coding Bounds"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Coding-Bounds"
toc_id: ~P3PdykAxfe~keBMYviCRA
content_id: eTwygHvM2m9WjR77XYjiUw
---

### Kernel Coding Bounds

This example requires a total of 1024 int16 x int16 multiplications to compute a 128 output value. Given that the AI Engine can perform 32 16-bit multiplications per cycle, the compute bound for the kernel is as follows.

```
Compute bound = 32 cycles / invocation
```

The vector register can store matrix B because it is only 16*16-bit =256 bits. It does not need to be fetched from the AI Engine data memory or tile interface for each MAC operation. Considering the data “a” needed for computation, there are total 64*8*2=1024 bytes to be fetched from memory. Given that AI Engine allows two 256 bits (32 bytes) loads per cycle, the memory bound for the kernel is as follows.

```
Memory bound = 1024 / (2*32) = 16 cycles / invocation
```

The compute bound is larger than the memory bound. Hence the purpose of vectorization can be to achieve the theoretical limit of MAC operations in the vector processor.
