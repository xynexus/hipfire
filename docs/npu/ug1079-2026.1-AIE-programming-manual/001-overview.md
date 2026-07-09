---
title: "Overview"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Overview"
toc_id: Tetu2r4JIGY8FJGQ42AsGA
content_id: EdZvTOZcbpYUA2Jg6o4ieA
---

# Overview

AI Engines are an array of very-long instruction word (VLIW) processors with single instruction multiple data (SIMD) vector units that are highly optimized for compute-intensive applications, specifically digital signal processing (DSP), 5G wireless applications, and AI technology such as machine learning (ML).

The AI Engine array supports three levels of parallelism:

- **Instruction Level Parallelism (ILP):** Through the VLIW architecture allowing multiple operations to be executed in a single clock cycle.
- **SIMD:** Through vector registers allowing multiple elements (for example, eight) to be computed in parallel.
- **Multicore:** Through the AI Engine array, allowing up to 400 AI Engines to execute in parallel.

Instruction-level parallelism includes a scalar operation, up to two moves, two vector reads (loads), one vector write (store), and one vector instruction that can be executed—in total, a 7-way VLIW instruction per clock cycle. Data-level parallelism is achieved via vector-level operations where multiple sets of data can be operated on a per-clock-cycle basis.

Each AI Engine contains both a vector and scalar processor, dedicated program memory, local 32 KB data memory, access to local memory in itself and three neighboring AI Engines with the direction depending on the row it is in. It also has access to DMA engines and AXI4 interconnect switches to communicate via streams to other AI Engines or to the programmable logic (PL) or the DMA. Refer to the Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine)) for specific details on the AI Engine array and interfaces.

While most standard C code can be compiled for the AI Engine, the code might need restructuring to take full advantage of the parallelism provided by the hardware. The power of an AI Engine is in its ability to execute a multiply-accumulate (MAC) operation using two vectors, load two vectors for the next operation, store a vector from the previous operation, and increment a pointer or execute another scalar operation in each clock cycle. Specialized functions called intrinsics allow you to target the AI Engine vector and scalar processors and provide implementation of several common vector and scalar functions, so you can focus on the target algorithm. In addition to its vector unit, an AI Engine also includes a scalar unit which can be used for non-linear functions and data type conversions.

AI Engine programs consist of a data flow (ADF) graph specification written in C++. This specification can be compiled and executed using the AI Engine compiler. An adaptive data flow (ADF) graph application consists of nodes and edges where nodes represent compute kernel functions, and edges represent data connections. Kernels in the application can be compiled to run on the AI Engines, and are the fundamental building blocks of an ADF graph specification.

The ADF graph is a modified [Kahn process network](https://en.wikipedia.org/wiki/Kahn_process_networks) with the AI Engine kernels operating in parallel. AI Engine kernels operate on data streams. These kernels consume input blocks of data and produce output blocks of data. Kernel behavior can be modified using static data or runtime parameter (RTP) arguments that can be either asynchronous or synchronous.

The following figure shows the conceptual view of the ADF graph and its interfaces with the processing system (PS), programmable logic (PL), and DDR memory. The figure shows the following:

- **AI Engine:** Each AI Engine is a VLIW processor containing a scalar unit, a vector unit, two load units, and a single store unit.
- **AI Engine Kernel:** AI Enginekernels are written in C/C++.
- **ADF Graph:** The graph consists of a single AI Engine kernel or multiple AI Engine kernels connected by data streams and/or buffers. It interacts with the PL, global memory, and PS with specific constructs like PLIO (port attribute in the graph programming that is used to make stream connections to or from the programmable logic), GMIO (port attribute in the graph programming that is used to make external memory-mapped connections to or from the global memory), and RTP.

![wra1610154938733.png](../assets/001-01-wra1610154938733-png-723bbeca5555.png)

*Figure 1. Conceptual Overview of the ADF Graph*

**Recommended:** AI Engine

AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment))

### AI Engine Kernels

An AI Engine kernel is a C++ program written using specialized intrinsic functions. These intrinsic functions target the different functional units of the AI Engine processor, like the VLIW vector and scalar unit. The AI Engine kernel code is compiled using the AI Engine compiler that is included in the AMD Vitis™ core development kit. The AI Engine compiler compiles kernels into an ELF file, which runs on AI Engine processors. Chapters 2 through 6 provide a high-level overview of kernel programming, tools, and reference documents for AI Engine kernel programming. In addition, these chapters provide details on scalar/vector programming, kernel optimization, interface considerations, and some examples.

### AI Engine Graphs

Introduction to Graph Programming provides an overview of the AI Engine programming model, an introduction to ADF graphs, and information about compiling and simulating an AI Engine graph. Streaming Data API describes the APIs that are available for data communication between kernels.

### Controlling the AI Engine Graph

Runtime Graph Control API describes the various control APIs available to control and update the AI Engine graphs at runtime. You can use graph control APIs to initialize, run, update, and control graph execution from an external controller within the platform context. This platform can be simulation-only, an extensible target which can be connected to the PL kernels, or fixed for bare-metal applications.

Specialized Graph Constructs describes specialized graph constructs that are useful in modeling specific scenarios. This chapter includes constructs like FIFO specification, explicit packet switching, and so on.

AI Engine/Programmable Logic Integration enumerates important points to consider when communicating programmable logic and AI Engine considerations to interface with the programmable logic. It also discusses aspects of AI Engine programmable logic interface performance.

Graph Programming Model details more advanced topics in Adaptive Dataflow graph programming. This chapter includes details on topics such as:

- Graph topologies
- I/O specifications including the supported graph topologies
- I/O specifications like AI Engine to/from PL (PLIO)
- AI Engine to/from NoC/DDR (GMIO)
