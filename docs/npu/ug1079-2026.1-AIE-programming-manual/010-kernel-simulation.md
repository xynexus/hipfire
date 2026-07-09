---
title: "Kernel Simulation"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Simulation"
toc_id: ph25vrk1sTejxJvmMbRmmw
content_id: S_MXBnFbFbp6sosYinMVzQ
---

## Kernel Simulation

To simulate an AI Engine kernel you need to write an AI Engine graph application, consisting of a data-flow graph specification written in C++. The graph contains the AI Engine kernel. Test bench data provides the input(s) to the kernel. The data output(s) from the kernel can be captured as the simulation output and compared against golden reference data. You can compile and execute this specification using the AI Engine compiler (`v++ -c --mode aie`). Use the `aiesimulator` to simulate the application. For additional details on the `aiesimulator`, see [Simulating an AI Engine Graph Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=yql1512608436352.html) in the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).
