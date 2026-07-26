---
title: "Runtime Parameter Specification"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Runtime-Parameter-Specification"
toc_id: ohBu7YI_IA~JSVVO0soGYw
content_id: cM0kyslkAOaPaUlw_rlfzQ
---

## Runtime Parameter Specification

The data flow graphs shown until now are defined completely statically. However, in real situations you might need to modify the behavior of the graph based on some dynamic condition or event. The data being processed might require modification. For example modifying a mode of operation or a new coefficient table. Alternatively, the modification might be in the control flow of the graph such as conditional execution, or dynamically reconfiguring a graph with another graph. Runtime parameters (RTP) are useful in such situations. Either the kernels or the graphs can be defined to execute with parameters. Additional graph APIs allow you to update or read parameter values while the graph is running.

The system supports two types of runtime parameters:

- **Asynchronous (Sticky) Parameters:** You can change these parameters at any time using a controlling processor such as the Processing System (PS). After a complete update, they are read each time a kernel invokes. You can use these parameters as filter coefficients that change infrequently.
- **Synchronous (Triggering) Parameters:** A kernel that requires a triggering parameter does not execute until a controlling processor writes the required parameters. Upon a write, the kernel executes once, reading the new updated value. After completion, the kernel is blocked from executing until the parameter is updated again. This allows a different type of execution model from the normal streaming model. This can be useful for updating operations where blocking synchronization is important.

Runtime parameters can either be scalar values or array values. A `graph.update()` API is used for RTP update.
