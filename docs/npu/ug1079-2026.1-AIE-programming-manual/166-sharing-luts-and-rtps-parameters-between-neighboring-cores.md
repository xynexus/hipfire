---
title: "Sharing LUTs and RTPs Parameters Between Neighboring Cores"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Sharing-LUTs-and-RTPs-Parameters-Between-Neighboring-Cores"
toc_id: ID4dN~sDUIVlZWD~~GF3KQ
content_id: ZckmPCoCn4drAl47cD_Zqg
---

## Sharing LUTs and RTPs Parameters Between Neighboring Cores

This specialized graph construct enables efficient sharing of parameters between neighboring cores, optimizing memory usage and reducing duplication of constant data across multiple kernels.

- Allow kernels to share parameters (for example, lookup tables (LUTs) and runtime parameters (RTPs)) between neighboring cores instead of maintaining individual copies of data, thereby reducing memory overhead in designs with multiple kernels.
- The sharing of LUTs and RTPs is limited to read-only or constant values and is supported on AIE and AIE-ML architectures.
- You must define LUT arrays and RTPs in kernel source files and use the `adf::share_parameter` construct to facilitate sharing between neighboring cores.
