---
title: "Specifying Runtime Data Parameters"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Specifying-Runtime-Data-Parameters"
toc_id: c52ENXyfItcnV7zdVto69w
content_id: 3YXbPOjSsQquGOQtqMssLw
---

### Specifying Runtime Data Parameters

##### Parameter Inference

If an integer scalar value appears in the formal arguments of a kernel function, then that parameter becomes a runtime parameter. In the following example, the arguments `select` and `result_out` are runtime parameters.

```
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
  void simple_param(input_buffer<int32> &in, output_buffer<int32> &out, int32 select, int32 &result_out);
#endif
```

`int8`

`int16`

`int32`

`int64`

`uint8`

`uint16`

`uint32`

`uint64`

`cint16`

`cint32`

`float`

**Note:** AI Engine

AI Engines

AI Engines

`filter_with_array_param`

```
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

  void filter_with_array_param(input_buffer<cint16> & in, output_buffer<cint16> * out, const int32 (&coefficients)[32]);

#endif
```

Implicit ports are inferred for each parameter in the function argument, including the array parameters. The following table describes the type of port inferred for each function argument.

| Formal Parameter | Port Class |
| --- | --- |
| T | Input |
| const T | Input |
| T & | Inout |
| const T & | Input |
| const T (&)[ …] | Input |
| T(&)[…] | Inout |

From the table, you can see that when the AI Engine cannot make externally visible changes to the function parameter, an input port is inferred. When the formal parameter is passed by value, a copy is made, so changes to that copy are not externally visible. When a parameter is passed with a `const` qualifier, the parameter cannot be written, so these are also treated as input ports.

When the AI Engine kernel is passed a parameter reference and can modify it, an inout port is inferred and can be used to allow reading back of results from the control processor.

**Note:** `graph::read()`

cannot

`graph::update()`

**Note:** `arg`

`kernel_function(int32 foo_in, int32 &foo_out)`

##### Parameter Hookup

You can connect both input and inout runtime parameter ports to corresponding hierarchical ports in their enclosing graph. This mechanism exposes parameters for runtime modification. In the following graph, an instance is created of the previously defined `simple_param` kernel. This kernel has two input ports, one `output` port and one `inout` port.

- The first argument to appear in the argument list, `in[0]`, is an input buffer.
- The second argument is an output buffer.
- The third argument is a runtime parameter (it is not a buffer or stream type). This is inferred as an input parameter, `in[1]`, because it is passed by value.
- The fourth argument is a runtime parameter. This is inferred as an `inout` parameter, `inout[0]`, because it is passed by reference.

**Note:** `in`

`out`

`inout`

In the following graph definition, a `simple_param` kernel is instantiated and buffers are connected to `in[0]` and `out[0]` (the input and output buffers of the kernel). The input runtime parameter is connected to the graph `input` port, `select_value`, and the `inout` runtime parameter is connected to the graph `inout` port, `result_out`.

```
class parameterGraph : public graph {
private:
  kernel first;

public:
  input_port select_value;
  input_plio in;
  output_plio out;
  inout_port result_out;
  parameterGraph() {
    first = kernel::create(simple_param);
    ......
    connect(in.out[0], first.in[0]);
    connect(first.out[0], out.in[0]);
    connect<parameter>(select_value, first.in[1]);//default sync rtp input
    connect<parameter>(first.inout[0], result_out);//default async rtp output
  }
};
```

An array parameter can be hooked up in the same way. The compiler automatically allocates space for the array data so that it is accessible from the processor where the kernel is mapped.

```
class arrayParameterGraph : public graph {
private:
  kernel first;

public:
  input_port coeffs;
  input_plio in;
  output_plio out;
  arrayParameterGraph() {
    first = kernel::create(filter_with_array_param);
    ......
    connect(in.out[0], first.in[0]);
    connect(first.out[0], out.in[0]);
    connect<parameter>(coeffs, first.in[1]);
  }
};
```

##### Input Parameter Synchronization

The default behavior for input runtime parameters ports is triggering behavior. This means that the parameter plays a part in the rules that determine when a kernel can execute. In this graph example, the kernel executes when the following three conditions are met:

- A valid buffer of 32 bytes of input data is available
- An empty buffer of 32 bytes is available for the output data
- A write to the input parameter takes place

In triggering mode, a single write to the input parameter allows the kernel to execute once, setting the input parameter value on individual kernel calls.

You can use an alternative mode to allow input kernels parameters to be set asynchronously. To specify that parameters update asynchronously, use the async modifier when connecting a port.

```
connect<parameter>(param_port, async(first.in[1]));
```

When you designate a kernel port as asynchronous, it no longer plays a role in the firing rules for the kernel. When the parameter is written once, the value is retained in subsequent firings. At any time, the PS can write a new value for the runtime parameter. That value is observed on the next and any subsequent kernel firing.

##### Inout Parameter Synchronization

The default behavior for inout runtime parameters ports is asynchronous behavior. This means the controlling processor or another kernel can read back the parameter, but this does not affect the producer kernel’s execution. For synchronous behavior from the `inout` parameter where the kernel blocks until the parameter value is read out on each invocation of the kernel, you can use a sync modifier when connecting the `inout` port to the enclosing graph as follows.

```
connect<parameter>(sync(first.inout[1]), param_port);
```
