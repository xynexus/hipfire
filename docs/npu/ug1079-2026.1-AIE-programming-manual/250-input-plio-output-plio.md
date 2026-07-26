---
title: "input_plio/output_plio"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/input_plio/output_plio"
toc_id: kkcKcAZGeJ02sZEtH63d4g
content_id: r~r8OHSbF_JOqDHMFC6zyA
---

### input_plio/output_plio

These classes represent the I/O port specification used to connect AI Engine kernels to the external platform ports representing programmable logic.

##### Member Functions

```
create(plio_type plio_width, std::string datafile);
```

The `input_plio` or `output_plio` port specification represents a single 32-bit, 64-bit, or 128-bit AXI4-Stream input or output at the AI Engine array interface. The `datafile` is an input or output file path that sources input data or receives output data for simulation purposes. You can capture this data separately during a simulation run.

```
create(std::string logical_name, plio_type plio_width, std::string datafile);
```

The `input_plio` or `output_plio` port specification represents a single 32-bit, 64-bit, or 128-bit AXI4-Stream input or output at the AI Engine array interface. Here the `plio_width` can be one of `plio_32_bits` (default), `plio_64_bits`, or `plio_128_bits`. The `logical_name` must be the same as the annotation field of the corresponding port as presented in the logical architecture interface specification.

```
create(std::string logical_name, plio_type plio_width, std::string datafile, double frequency);
```

The `input_plio` or `output_plio` port specification represents a single 32-bit, 64-bit, or 128-bit AXI4-Stream input or output at the AI Engine array interface. Here `plio_width` can be one of `plio_32_bits` (default), `plio_64_bits`, or `plio_128_bits`. The frequency of the `input_plio`/`output_plio` port can also be specified as part of the constructor.

```
create(std::string logical_name, plio_type plio_width, std::string datafile, double frequency, bool hex);
```

The above `input_plio`/`output_plio` boolean port specification indicates if the contents in the input data file are in hex or binary formats.

The data in the data files must be organized according to the bus width of the `input_plio`/`output_plio` attribute (32, 64, or 128) per line as well as the data type of the graph port it is connected to. For example, a 64-bit `input_plio` feeding a kernel port with data type `int32` requires file data organized as two columns. However, the same 64-bit `input_plio` feeding to a kernel port with data type `cint16` requires the data to be organized into four columns, each representing a 16-bit real or imaginary part of the complex data type.
