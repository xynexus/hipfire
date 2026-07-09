---
title: "Kernel Coding Bounds"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Coding-Bounds"
toc_id: WXV55vIR6pCmQt_3x0espA
content_id: pXQ6dcp_MQLYiiWYN_lOOg
---

### Kernel Coding Bounds

This example requires a total of 16 int16 x int16 multiplications per output value. Matrix C consists of 64 values, so requires a total of 16 * 64 = 1024 multiplications to complete one matrix multiplication. 32 16-bit multiplications can be performed per cycle in an AI Engine. Therefore, the minimum number of cycles required for the matrix multiplication is 1024/32 = 32. The summation of the individual terms comes without additional cycle requirements because the addition can be performed together with the multiplication in a MAC operation. Hence the compute bound for the kernel is:

Next, analyze the memory accesses bound for the kernel. If it is going to fully use the vector unit MAC performance, 32 16-bit multiplications are performed per cycle. Vector b can be stored in the vector register because it is only 16*16-bit =256 bits. It does not need to be fetched from the AI Engine data memory or tile interface for each MAC operation. Considering data “a” needed for computation, it needs 32*16-bit = 512 bits data per cycle. The stream interface only supports 2*32 bit per cycle and hence fetching data from memory can be considered. It allows two 256 bits loads per cycle which matches the MAC performance. Thus, if two 256 bits loads are performed each cycle, the memory bound for the kernel is:

Note that compute bound and memory bound are the theoretical limits of the kernel realization. It does not take into account the function overhead outside the main computation loop. When the kernel forms only part of the graph, it can be relieved due to bandwidth limitation of other kernels or lower system performance requirements.
