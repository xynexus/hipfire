---
title: "Design Analysis"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Design-Analysis"
toc_id: Tjzj0owl5WelkKAC~zLu~A
content_id: VhDiZ7KDYPTyQ5td4y3Juw
---

### Design Analysis

The following equation describes the finite impulse response (FIR) filter. x denotes the input, C denotes the coefficients, y denotes the output, and N denotes the length of the filter.

![eul1606806208916.png](../assets/064-01-eul1606806208916-png-8ed34e2e4b8c.png)

Following is an example of a 32-tap filter.

![igx1606887920100.png](../assets/064-02-igx1606887920100-png-c7875cf4dea6.png)

Each output takes 32 multiplications. If you use `cint16` for data and coefficient types, the kernel needs four cycles to compute a sample. Each AI Engine performs eight MAC operations per cycle. If data is streaming from one stream port (32 bits), one data can produce one output (in the middle of processing).

So, the design is compute bound. You can split the kernel into four cascaded kernels to process one sample per cycle.
