---
title: "Runtime Parameter Specification"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Runtime-Parameter-Specification"
toc_id: W2ehVT0dovngIVjMHi4oxQ
content_id: sW8zffa9Z3M_e5AfmKLIYA
---

## Runtime Parameter Specification

Using runtime parameters (RTP) is another way to pass data to the kernels. The following two execution models are supported for RTPs:

1. Asynchronous parameters can be changed at any time by a controlling processor such as the Arm® processor. The system reads them each time it invokes a kernel. This means the parameter update happens between kernel executions. It does not require a specific update pattern. For example, these types of parameters can serve as filter coefficients that change infrequently.
2. Synchronous parameters (triggering parameters) block a kernel from running until a controlling processor, such as the Arm® processor, writes them. Upon a write, the kernel reads the new updated value and executes once. After completion, it is blocked from executing until the parameter is updated again. This allows a different type of execution model from the normal streaming model. This can be useful for certain updating operations where blocking synchronization is important.

It is very important to understand that the RTP interaction between AI Engine kernels only happens in kernel execution boundaries. This means that you can read the RTP output of the source kernel only when the source kernel has completed its current iteration.

**Note:** AI Engine

For more information about runtime parameter usage, refer to the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).
