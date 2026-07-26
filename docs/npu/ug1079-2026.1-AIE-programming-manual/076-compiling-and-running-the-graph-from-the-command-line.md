---
title: "Compiling and Running the Graph from the Command Line"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Compiling-and-Running-the-Graph-from-the-Command-Line"
toc_id: MEKeSit3UDdcGyen~ZCktg
content_id: ze4iZxOybpvxv5N92~zVug
---

## Compiling and Running the Graph from the Command Line

1. To compile your graph, execute the following command. (See [Compiling an AI Engine Graph Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=rsb1512607764188.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more details.)`v++ -c --mode aie --target hw --config ./config.cfg project.cpp` The program is named project.cpp. The AI Engine compiler reads the input graph, compiles it to the AI Engine array, produces reports, and generates output files in the Work directory.

  ```
  v++ -c --mode aie --target hw --config ./config.cfg project.cpp
  ```

  The program is named project.cpp. The AI Engine compiler reads the input graph, compiles it to the AI Engine array, produces reports, and generates output files in the Work directory.
2. The AI Engine compiler parses the C++ input into a graphical intermediate form expressed in JavaScript object notation (JSON). The compiler then performs resource mapping, scheduling analysis. It maps kernel nodes in the graph to the processing cores in the AI Engine array and data buffers to memory banks. The JSON representation is displays mapping information. Each AI Engine also requires a schedule of all the kernels mapped into it.The input graph is first partitioned into groups of kernels to be mapped to the same core. You can view the output of the mapper as a tabular report in the file project_mapping_analysis_report.txt. This file reports the mapping of nodes to processing cores and data buffers to memory banks. Inter-processor communication is appropriately double-banked as ping-pong buffers.

  The input graph is first partitioned into groups of kernels to be mapped to the same core.

  You can view the output of the mapper as a tabular report in the file project_mapping_analysis_report.txt. This file reports the mapping of nodes to processing cores and data buffers to memory banks. Inter-processor communication is appropriately double-banked as ping-pong buffers.
3. The AI Engine compiler allocates the necessary locks, memory buffers, DMA channels, descriptors. The compiler also generates routing information to map the graph onto the AI Engine array. The compiler synthesizes a main program for each core. The program schedules all the kernels on the cores, and implements the necessary locking mechanism and data copy among buffers. The C/C++ program for each core is compiled to produce loadable ELF files. The AI Engine compiler also generates control APIs to control the graph initialization, execution, and termination from the `main` application and a simulator configuration script scsim_config.json. These are all stored within the Work directory under various sub-folders. See [Compiling an AI Engine Graph Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=rsb1512607764188.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more details.

  The AI Engine compiler allocates the necessary locks, memory buffers, DMA channels, descriptors. The compiler also generates routing information to map the graph onto the AI Engine array.

  The compiler synthesizes a main program for each core. The program schedules all the kernels on the cores, and implements the necessary locking mechanism and data copy among buffers. The C/C++ program for each core is compiled to produce loadable ELF files. The AI Engine compiler also generates control APIs to control the graph initialization, execution, and termination from the `main` application and a simulator configuration script scsim_config.json. These are all stored within the Work directory under various sub-folders. See [Compiling an AI Engine Graph Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=rsb1512607764188.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more details.
4. After compiling the AI Engine graph, the AI Engine compiler writes a summary of compilation results called `<graph-file-name>.aiecompile_summary`. You can view this in the Vitis IDE. The summary contains a collection of reports, and diagrams reflecting the state of the AI Engine application implemented in the compiled build. The summary is written to the working directory of the AI Engine compiler as specified by the `--work_dir` option, which defaults to ./Work.To open the AI Engine compiler summary, use the following command: `vitis -a ./Work/graph.aiecompile_summary`

  To open the AI Engine compiler summary, use the following command:

  ```
  vitis -a ./Work/graph.aiecompile_summary
  ```
5. To run the graph, execute the following command. See [Simulating an AI Engine Graph Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=yql1512608436352.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more details. `aiesimulator –-pkg-dir=./Work` This command starts the SystemC-based simulator with the control program being the `main` application. The graph APIs which are used in the control program configure the AI Engine array. Configuration includes setting up static routing, programming the DMAs, loading the ELF files onto the individual cores, and initiating AI Engine array execution. At the end of the simulation, the output data is produced in the directory aiesimulator_output and it should match the reference data. You can load the graph at device boot time in hardware, or through the host application. [Building and Running the System](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1700-vitis-accelerated-data-center&resourceid=xia1553473418160.html) in the Data Center Acceleration using Vitis ([UG1700](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1700-vitis-accelerated-data-center)) describes deploying the graph in hardware and the flow associated with it.

  ```
  aiesimulator –-pkg-dir=./Work
  ```

  This command starts the SystemC-based simulator with the control program being the `main` application. The graph APIs which are used in the control program configure the AI Engine array. Configuration includes setting up static routing, programming the DMAs, loading the ELF files onto the individual cores, and initiating AI Engine array execution.

  At the end of the simulation, the output data is produced in the directory aiesimulator_output and it should match the reference data.

  You can load the graph at device boot time in hardware, or through the host application. [Building and Running the System](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1700-vitis-accelerated-data-center&resourceid=xia1553473418160.html) in the Data Center Acceleration using Vitis ([UG1700](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1700-vitis-accelerated-data-center)) describes deploying the graph in hardware and the flow associated with it.

**Note:** AI Engine
