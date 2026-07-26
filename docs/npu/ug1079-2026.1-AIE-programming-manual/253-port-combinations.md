---
title: "Port Combinations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Port-Combinations"
toc_id: NQZwd9GMLZQKI6PNaWbtXw
content_id: IuPEnPaF94J0eFfQbj4~iA
---

### Port Combinations

The following table shows the port combinations used in the constructor templates.

| PortA | PortB | Comment |
| --- | --- | --- |
| port <output> | port <output> | Connect a kernel output to a parent graph output |
| port <output> | port <input> | Connect a kernel output to a kernel input |
| port <output> | port <inout> | Connect an output of a kernel or a subgraph to an inout port of another kernel or a subgraph |
| port <input> | port <input> | Connect a graph input to a kernel input |
| port <input> | port <inout> | Connect an input of a parent graph to an inout port of a child subgraph or a kernel |
| port <inout> > | port <input> | Connect an inout port of a parent graph or a kernel to an input of another kernel or a subgraph |
| port <inout> | port <output> | Connect an inout port of a subgraph or a kernel to an output of a parent graph |
| parameter& | kernel& | Connect an initialized parameter variable to a kernel ensuring that the compiler allocates space for the variable in the memory around the kernel |

Related reference

Logical I/O Ports
