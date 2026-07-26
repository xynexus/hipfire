---
title: "Revision History"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Revision-History"
toc_id: Ks1dASoalCnaCj4mRTPuRw
content_id: LHXqPzVSKscW6wFg4QbINQ
---

## Revision History

The following table shows the revision history for this document.

| Section | Revision Summary |
| --- | --- |
| 11/26/2025 Version 2025.2 |  |
| Kernel Location Constraints | Added physical port name information. |
| Shared Graph-Scoped Tables | Updated code block. |
| General updates | Deprecated writeincr_v (changed to writeincr). |
| 06/04/2025 Version 2025.1 |  |
| Memory and DMA Programming | Added chapter on local tile memory and DMA programming. |
| AI Engine Local Memory Access | Added section on local tile memory access. |
| Tiling Parameters and Buffer Descriptors | Added section on tiling parameters and buffer descriptors on an input or output port of an AIE memory DMA. |
| Tiling Parameters Specification | Added section to describe tiling parameter specification in the graph with regard to an input or output port of an AIE memory DMA. |
| Viewing Tiling Parameters in the Vitis IDE | Updated figure to show tile memory module tiling parameters table in Vitis IDE. |
| PLIO and GMIO Location Constraints | Added section on PLIO and GMIO location constraints. |
| 11/28/2024 Version 2024.2 |  |
| Synchronous Buffer Port Access | Clarified the behavior of locks on a synchronous output buffer, during multiple iterations of a kernel. |
| Circular Output Buffer | Clarified buffer addressing limitations when multiple kernels are mapped to the same tile. |
| Global Graph-Scoped Tables | Added this topic. |
| Explicit Packet Switching | Clarified that packet stream can be created from one or more AI Engine kernels to one or more multiple destination AI Engine kernels. |
| Packet Switching Graph Constructs | Clarified that data read data from a packet stream is read as an integer value by default, but that can then can be cast to a different data type if desired. |
| Example of Packet Switching between PL and AI Engine, Example of Packet Switching from an AI Engine to Multiple AI Engines, and Example on Packet Switching from Multiple AI Engines to an AI Engine | Added examples of how to set up packet switching connections between PL and AI Engine. |
| Logical I/O Ports | Added topic that explains the usage of the graph port objects. |
| 06/05/2024 Version 2024.1 |  |
| General | Removed --disable-multirate option on the AI Engine compiler. |
| Changed output_stream to output_cascade, and changed input_stream to input_cascade. |  |
| Changed writeincr_v<n> to writeincr. |  |
| Updated to reflect that bfloat16 is supported only on AI Engine ML devices. |  |
| Removed unsupported data types: cacc32, aac40, cacc40, acc56, cacc56, acc72, cacc72, accfloat, and caccfloat. |  |
| Scalar Programming | Added note that GNU defined types work for x86 simulation but not for AI Engine simulation. |
| Chess Directives and C++ Attributes | Provides Chess directives and the equivalent C++ attribute. |
| Parallel Streams Access | Added recommendation that when resource annotations are used on streams, to use the annotation consistently on the same stream. |
| Data Reshaping | Added that the AI Engine API provides functions to extract real or imaginary parts from complex scalar or vector data. |
| Casting and Datatype Conversion | Added details about the API supporting conversions between vector data types. |
| Packet Processing | Added getPacketid API example. |
| Packet Switching Graph Constructs | Added examples to illustrate packet merge, and packet split APIs. Also, added an example of using floating point data. |
| Examples | Added warning note that two kernels connected by buffers should have both linear or circular addressing if the two kernels are in the same tile. |
| Throughput Measurements Using Event APIs | Changed end_profiling to stop_profiling. |
| Vector Arithmetic Operations | Added aie::saturating_add, and aie::saturating_sub. Added example code to show the difference between aie::add and aie::saturating_add. |
| Design Considerations for Graphs Interacting with Programmable Logic | Added Conceptual Representation of AIE-PL Interface. |
| Event API | Moved Event API section from Appendix C in UG1079 to Appendix A in UG1076. |
| 12/04/2023 Version 2023.2 |  |
| General | Updated to reflect the usage of the new unified AMD Vitis™ IDE. |
| Casting and Datatype Conversion | Added more information about scalar floating-point operations options. |
| Inline Keywords | Added inline keyword explanation. |
| Input and Output Buffers | Specified maximum single buffer port size. |
| Asynchronous Buffer Port Access | Added important note that operations on asynchronous buffer should be done after the buffer is acquired. |
| Stream Data Types | Added supported cascade accumulator data types. |
| 10/18/2023 Version 2023.2 |  |
| Buffer Port Connection for Multirate Processing | Added code example to accompany the Up-Converter Followed by a Single Rate Kernel figure. |
| Viewing Loop II in the Vitis IDE | Added new section. |
| Using Vitis Unified IDE and Reports | Updated for the Vitis unified IDE. |
| Design Flow Using RTL Programmable Logic | Added the PL Kernels Inside AI Engine Graph section. |
| AI Engine to PL Interface Text Input Format | Updated section with new examples. |
| Examples | Added the Using Input and Output Buffer as Intermediate Storage section. |
| Creating a Data Flow Graph (Including Kernels) | Added important note that the main function must have a return statement. Otherwise, aiecompiler will error out. |
| Introduction to Scalar and Vector Programming, AI Engine Data Types, andVector Registers | Updated instances of windows to input and output buffers. |
| 06/23/2023 Version 2023.1 |  |
| General | Editorial update; no technical update. |
| 05/23/2023 Version 2023.1 |  |
| General | Editorial update; no technical update. |
| 05/16/2023 Version 2023.1 |  |
| General | Updated window to buffer, as in input_window to input_buffer and output_window to output_buffer. |
| Vector Data Types | Added uint16 and uint32 to Supported Vector Types and Sizes table. |
| Input and Output Buffers | Added chapter to introduce the input and output buffers. These buffers are a replacement for windows, which are deprecated in 2023.1. |
| Introduction to Scalar and Vector Programming | Updated code examples to use buffers throughout the chapter. |
| AI Engine API Overview | Added aie::print_matrix. |
| Iterators | Added aie::begin_restrict_vector and aie::cbegin_restrict_vector. |
| Example Designs Using the AI Engine API | Updated for the 2023.1 release and to use buffers. |
| Recommended Project Directory Structure | Added guidance that only one source file is allowed to be specified as kernel source via adf::source. |
| Streaming Data API | Updated for the 2023.1 release. |
| Reading and Advancing an Input Stream and Writing and Advancing an Output Stream | Added note about how to indicate the end of the stream, with command example. |
| Packet Split and Merge Connections | Added new section. |
| Multicast Support | Added note to indicate that the compiler supports multirate processing for multicast connections. |
| Conditional Ports | Added section to describe the compiler feature that allows for conditional ports on template functions. |
| Array of Graph Objects | Added section to describe how to conditionally instantiate an array of graph objects. |
| Graph Programming Model | Updated code snippets, and screenshots throughout chapter. |
| Data Throughput Estimate in Hardware | Added new section to describe ways to measure interface throughput of the AI Engine in hardware. |
| Throughput Measurement Using Timers | Added new section to describe the way to measure interface throughput using timers. |
| Throughput Measurements Using Event APIs | Added new section to describe the way to measure interface throughput using events. |
| Throughput Measurements using Event Trace | Added new section. |
| Connection Constructor Templates | Added new section that describes how to specify templated connect statements in a graph.. |
| Design Analysis and Programming using Intrinsics | Added note about input_window and output_window being deprecated, and when these are found in code examples in this appendix, to use input_buffer and output_buffer instead. |
| 10/19/2022 Version 2022.2 |  |
| Document title | Changed title to AI Engine Kernel and Graph Programming Guide. |
| Document-wide changes | Added graph programming content from AI Engine Tools and Flows User Guide (UG1076). |
| Vector Arithmetic Operations | Updated examples syntax. |
| Rounding and Saturation Modes | Changed truncate to saturate. |
| Casting and Datatype Conversion | Updated the conversion functions (to_float() and to_fixed()) example. |
| Operator Overloading | Updated operators example code. |
| Data Reshaping | Updated to include aie::vector<int32,8>. |
| Matrix Multiplication | Updated section for the release. |
| Window-Based Access | Made clarifying updates. |
| Stream-Based Access | Added Stream Connection for Multi-Rate Processing subsection. |
| Relative Constraints | New topic. |
| MAC on 16x16 bits | Made clarifying updates. |
| Relative Constraints | New topic. |
| Graph Topologies | New section. |
| Other Constraints | Added async_repetition constraint. |
| 05/25/2022 Version 2022.1 |  |
| Vector Data Types | Highlighted data type sizes supported natively by the AI Engine. |
| Vector Registers | Added the grow_replicate function on aie::vector. |
| Loops | Added Loop Flattening and Unrolling section. |
| Scheduling Separator | Added information about scheduling separator pragma. |
| Parallel Streams Access | Added information about accessing two input and/or output streams in parallel. |
| Profiling Kernel Code | Added information about profiling kernel code using cycles() API. |
| Runtime Parameter Specification | Removed section about AI Engine-to-AI Engine runtime parameter support. |
| Data Communication via AXI4-Stream Interconnect | Updated the section. |
| Buffer vs. Stream in Data Communication | Updated section to reflect wIndow multicast support. |
| Example Designs Using the AI Engine API | New chapter. Added new FIR Filter and Matrix Multiplication example designs. |
| Update, Extract, and Shift | Added information about using the update API on accumulators. |
| 11/10/2021 Version 2021.2 |  |
| Scalar Processing Unit | Updated for AI Engine API. |
| AI Engine Memory | Added information about heap and stack size. |
| AI Engine API | New section. |
| Introduction to Scalar and Vector Programming | Updated for AI Engine API. |
| AI Engine API Overview | New section. |
| Vector Arithmetic Operations |  |
| Vector Reduction |  |
| Bitwise Operations |  |
| Data Comparison |  |
| Data Reshaping |  |
| Iterators |  |
| Operator Overloading |  |
| Multiple Lanes Multiplication - sliding_mul |  |
| Matrix Multiplication - mmul |  |
| API Operation Examples |  |
| Loops | Updated information. |
| Floating-Point Operations | Updated for AI Engine API. |
| Single Kernel Programming using Intrinsics | Appendix describing programming using intrinsics. |
| Design Analysis and Programming using Intrinsics | Appendix describing design analysis and programming using intrinsics. |
| 07/19/2021 Version 2021.1 |  |
| Accumulator Registers | Added information about print acc value, as well as streaming data APIs. |
| Casting and Datatype Conversion | Added a note about the AI Engine floating-point. |
| Initialization | Added information about the static keyword. |
| Load and Store with Virtual Resource Annotations | Added new section. |
| Buffer vs. Stream in Data Communication | Added information. |
| DDR Memory Access through GMIO | Removed information related to PL GMIO. |
| Mapping Algorithm onto the AI Engine | Clarified description. |
| Coding with Intrinsics | Added information about the (always_inline) attribute. |
| 02/04/2021 Version 2020.2 |  |
| Initial release. | N/A |
