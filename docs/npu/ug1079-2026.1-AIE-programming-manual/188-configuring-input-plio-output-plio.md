---
title: "Configuring input_plio/output_plio"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Configuring-input_plio/output_plio"
toc_id: 4yxYTPZn5e5T5oPqyCvtew
content_id: IQ4xM5isRIgPcWPZsA0LVA
---

## Configuring input_plio/output_plio

You can configure an `input_plio`/`output_plio` object to make external stream connections that cross the AI Engine to the PL boundary. This occurs when a hardware platform is designed separately and the PL blocks are already instantiated inside the platform. This hardware design is exported from the Vivado tools as a package XSA and you must specify it when creating a new project in the AMD Vitis™ tools using that platform. The XSA contains a logical architecture interface specification that identifies which AI Engine I/O ports the platform supports. The following is an example interface specification containing stream ports (looking from the AI Engine perspective).

| AI Engine Port | Annotation | Type | Direction | Data Width | Clock Frequency (MHz) |
| --- | --- | --- | --- | --- | --- |
| S00_AXIS | Weight0 | stream | slave | 32 | 300 |
| S01_AXIS | Datain0 | stream | slave | 32 | 300 |
| M00_AXIS | Dataout0 | stream | master | 32 | 300 |

This interface specification describes how the platform exports two stream input ports (slave port on the AI Engine array interface) and one stream output port (master port on the AI Engine array interface). An `input_plio`/`output_plio` attribute specification to represents and connects these interface ports to their respective destination or source kernel ports in data flow graph.

The following example shows how the `input_plio`/`output_plio` attributes shown in the previous table can be used in a program to read input data from a file or write output data to a file. The width and frequency of the `input_plio`/`output_plio` port are also provided in the PLIO constructor.

```
adf::input_plio wts  = adf::input_plio::create("Weight0", adf::plio_32_bits, "inputwts.txt", 300);
adf::input_plio din  = adf::input_plio::create("Datain0", adf::plio_32_bits, "din.txt", 300);
adf::output_plio out = adf::output_plio::create("Dataout0", adf::plio_32_bits, "dout.txt", 300);
```

When simulated, the input weights and data are read from the two supplied files and the output data is produced in the designated output file in a streaming manner.

When a hardware platform is exported, all the AI Engine to PL stream connections are already routed to specific physical channels from the PL side.

#### Wide Stream Data Path PLIO

Typically, the AI Engine array runs at a higher clock frequency than the internal programmable logic. The `--pl-freq`option can be set to specify the frequency at which the PL blocks are expected to run. To balance the throughput between AI Engine and internal programmable logic, it is possible to design the PL blocks for a wider stream data path (64-bit, 128-bit), which is then sequentialized automatically into a 32-bit stream on the AI Engine stream network at the AI Engine to PL interface crossing.

The following example shows how wide stream `input_plio`/`output_plio` attributes can be used in a program to read input data from a file or write output data to a file.

```
adf::output_plio pl_out = adf::output_plio::create("TestLogicalNameOut", adf::plio_128_bits, "data/output.txt");
adf::input_plio  pl_in  = adf::input_plio::create("TestLogicalNameIn", adf::plio_128_bits, "data/input.txt");
...
adf::connect(pl_in.out[0], kernel_first.in[0]);
adf::connect(kernel_last.out[0], pl_out.in[0]);
```

In the previous example, two 128-bit PLIO attributes is declared: one for input and one for output. The `input_plio` and `output_plio` are then hooked up to the graph in the usual way. Data files specified in the `input_plio`/`output_plio` attributes are then automatically opened for reading the input or writing the output respectively.

When simulating `input_plio`/`output_plio` with data files, organize the data to accommodate both the width of the PL block and the data type of the connecting port on the AI Engine block. For example, organize a data file representing a 32-bit PL interface to an AI Engine kernel expecting `int16` as two columns per row, where each column represents a 16-bit value. As another example, organize a data file representing 64-bit PL interface to an AI Engine kernel expecting `cint16` as four columns per row, where each column represents a 16-bit real or imaginary value. The same 64-bit PL interface feeding an AI Engine kernel with `int32` port needs to organize the data as two columns per row of 32-bit real values. The following examples show the format of the input file for the previously mentioned scenarios.

```
64-bit PL interface feeding AI Engine kernel expecting cint16
input file:
0 0 0 0
1 1 1 1
2 2 2 2

64-bit PL interface feeding AI Engine kernel expecting int32
input file:
0 0
1 1
2 2
```

With these wide PLIO attribute specifications, the AI Engine compiler generates the AI Engine array interface configuration. It converts a 64-bit or 128-bit data into a sequence of 32-bit words. The AXI4-Stream protocol used by all PL IP blocks ensures that partial data can be sent on a wider data path. This includes the appropriate strobe signals describing which words are valid.

Related reference

Adaptive Data Flow Graph Specification Reference
