---
title: "Introduction to Graph Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Introduction-to-Graph-Programming"
toc_id: DihYUn_C8xoNNjTcIRhTng
content_id: P1_Qz308kcytDPETRRInQg
---

# Introduction to Graph Programming

An AI Engine program must include a Data Flow Graph Specification written in C++. An adaptive data flow (ADF) graph is a network with a single AI Engine kernel or multiple AI Engine kernels connected by data streams. The graph can interact with the programmable logic (PL), global memory, and/or the host processor using specific constructs. `input_plio` and `output_plio` port objects can be used to establish stream connections to or from the programmable logic, `input_gmio` and `output_gmio` port objects can be used to establish memory-mapped connections to or from the global memory, and RTP (Runtime Parameter) objects can be used to setup and control parameters required by the kernels during graph execution.

A graph can have multiple kernels, input and output ports. Graph connectivity is equivalent to the nets in a data flow graph. Connections can be between kernels, between kernel and input ports, or between kernel and output ports. You can configure each as a connection. A graph executes when all the data samples equaling the buffer or stream of data expected by the kernels in the graph become available. It produces data samples equaling the buffer or stream of data expected at the output of all the kernels in the graph.

As described in C++ Template Support you can use template classes or functions for writing the AI Engine graph or kernels. You can compile and execute the application using the AI Engine tool chain. This chapter provides an introduction to writing an AI Engine program.

The example that is used in this chapter can be found as a template example in the AMD Vitis™ environment when creating a new AI Engine project.

Related reference

Adaptive Data Flow Graph Specification Reference
