---
title: "Buffer Port Attributes and API"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Port-Attributes-and-API"
toc_id: 8Snt6SFvTyLPbZqk64wP8Q
content_id: 2u5UNuJFz8QBWHmS0x2nSg
---

## Buffer Port Attributes and API

You can configure the following attributes on buffer ports:

- Size
- Margin
- Locking protocol (synchronous or asynchronous)
- Addressing mode (linear or circular)
- Single buffer versus ping-pong-buffer

The kernel function signature specifies margin, locking protocol and addressing mode. The graph specifies single buffer versus ping-pong buffer.

You can specify the size of a buffer port in either the kernel signature or in the graph constructor. If the kernel function signature does not specify a size or uses `adf::inherited_extent`, you must define the size in the graph using the `dimensions()` API. The `dimensions()` API is used in the constructor of the graph and configures buffer port size in terms of the number of samples.

`adf::inherited_extent`, represents the size of the buffer which is inferred from the context, for example, it is inferred from the specification in the graph.

The buffer port margin represents the number of samples copied from the end of one block to the beginning of the next one. If a margin parameter is specified, the total memory allocated is:

`sizeof`

In the case of a circular output buffer, you can specify the margin can be specified as `adf::inherited_margin`. In this case, the sink connected to this output port specifies the margin size.

Following is an example function prototype specifying margin size:

```
simple(input_buffer<int32, adf::extents<adf::inherited_extent>,
        adf::margin<MARGIN_SIZE>> & in0,
        output_buffer<int32> & out0);
```

**Note:** - Buffer port types define their dimensions and margin in number of samples of data. To determine the actual data size stored in physical memory, you need to multiply the total number of samples with supported data type in bytes.
- Buffer size and margin size must be a multiple of 16 bytes.
- Constructs related to I/O buffers are defined in the `adf` namespace. Therefore, these constructs need to be qualified with `adf::`, unless you include `using namespace adf;`.
