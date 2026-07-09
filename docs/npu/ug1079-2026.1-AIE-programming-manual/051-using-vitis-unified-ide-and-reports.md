---
title: "Using Vitis Unified IDE and Reports"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Using-Vitis-Unified-IDE-and-Reports"
toc_id: W3uKzM07eV7jqyzRjvMuZw
content_id: LFFYbM~fEgSB~3e96x6k2A
---

## Using Vitis Unified IDE and Reports

Vitis Unified IDE manages the AI Engine components and system project. The Vitis Unified IDE provides views for AI Engine kernel development, and it is essential in the kernel development, debugging, and analysis.

Vitis Unified IDE has a debug view which displays the following:

- Registers
- Variables
- Available breakpoints
- Variables to register/memory mapping
- Internal/external memory contents
- Disassembly view
- Pipeline view for instructions

When launching the simulation, if Enable Profile is selected in launch configurations, it shows the `printf` output in the console. The Enable Trace check box in launch configurations is for generating event trace data which helps better understand when and how events such as memory stall and stream stall occur. Event trace is helpful in performance tuning.

When launching the debug perspective, if Enable Pipeline View is selected in launch configurations, the AI ENGINE PIPELINE view can be shown.

![rje1698438565381.png](../assets/051-01-rje1698438565381-png-b58f51d07b70.png)

*Figure 1. Debug Configurations*

In the debug perspective, debug commands enable you to resume, step into, and step over. The AI Engine source code is shown, and it is possible to set breakpoints by double-clicking lines. The windows Variables, Breakpoints, and Register Inspector are available to look into data memory or register status. The Disassembly view is helpful in understanding how instructions are used, especially how they are scheduled in the pipeline. The Pipeline view allows you to correlate instructions executed in a specific clock cycle with the labels in the Disassembly view.

![kpj1698438617597.png](../assets/051-02-kpj1698438617597-png-76c830d9a16b.png)

*Figure 2. Debug Perspective*

The generated code for an AI Engine (`Col_Row.cc`) includes the AI Engine kernels in the core and wrapper code. From the AI Engine wrapper code, you can step into the AI Engine kernel code by clicking multiple step-in buttons. Alternatively, you can open the AI Engine kernel source file from the design perspective, and set breakpoints in the file. You can use multiple views, such as Disassembly, Pipeline, Memory Inspector, Register Inspector, and Variables for debug, and performance tuning.

**Note:** Each tile supports a maximum of four breakpoints. To set new breakpoints, beyond the number allowed, you must clear existing breakpoints. The tool issues error messages if you try to set breakpoints beyond the maximum allowed.

The Disassembly view displays the compiler generated instruction target to the hardware. C/C++ source code can also be embedded between the lines for source code referencing. The instruction helps understand the compiled result, especially the loop pipelining result.

You can locate the loop in the kernel by scrolling or stepping in the Disassembly view. The loop iterates from zero-overhead loop start (ZLS) to zero-overhead loop end (ZLE). You can see how load instructions and MAC instructions are placed to be pipelined. The preamble and postamble instructions are before and after the zero-overhead loop body. These instructions fill and flush the pipeline stages.

The Microcode view provides a static view of compiled kernel instructions. Open Microcode view via AI Engine components or links in the Tile view in the analysis view in the Vitis IDE. By right-clicking the Microcode view, set the option to Enable AIE Cross Probing. Cross probing occurs between the Microcode view and source code.

![hll1698438683317.png](../assets/051-03-hll1698438683317-png-068660bc4a61.png)

*Figure 3. Microcode View with Cross-Probing*

Linker memory map reports for AI Engines cores are located in `Work/aie/core_ID/Release/core_id.map`. This file lists the locations of the program and data memories by functions, static variables, and the software stack. From these reports, the stack size, program memory size, global buffers, and their sizes can be extracted. You can generate an XML version of the linker report (`core_id.map.xml`) by specifying the option `-Xchess=\"main:bridge.xargs=-fB\"` to the AI Engine compiler.

`Work/<name>.aiecompile_summary` is the compilation summary that can be opened in the Vitis IDE. `Work/reports` contains multiple reports for the graph compilation result such as, kernels and buffers mapping result. Refer to the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for additional information about AI Engine compiler outputs.
