---
title: "constraint<T>"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/constraint-T"
toc_id: a9Xy0~64PR6_i0uO3OerFw
content_id: aWrAk~4PPXFnuT49mZCJUg
---

### constraint<T>

This template class builds scalar data constraints on kernels, connections, and ports.

##### Scope

A constraint must appear inside a user graph constructor.

##### Member Functions

```
constraint<T> operator=(T)
```

This overloaded equality operator allows you to assign a value to a scalar constraint.

##### Constructors

The default constructor is not used. Instead the following special constructors are used with specific meaning.

```
constraint<std::string>& initialization_function(kernel&)
```

This constraint allows you to set a specific initialization function for each kernel. The constraint expects a string denoting the name of the initialization function. Where multiple kernels are packed on a core, each initialization function packed on the core is called exactly once. No kernel functions schedule until all the initialization functions packed on a core complete.

An initialization function cannot return a value and cannot have input/output arguments, that is, the function prototype must be as follows.

```
void init_function_name(void)
```

**Note:** `graph::run`

You can us this function can to initialize global variables and set or clear rounding and saturation modes. It cannot use buffer or stream APIs to access memory or stream interfaces, but stream intrinsics (for example, `get_ss()`) can be used.

```
constraint<float>& runtime<ratio>(kernel&)
```

This constraint allows you to set a specific core usage fraction for a kernel. This is computed as a ratio of the number of cycles taken by one invocation of a kernel (processing one block of data) to the cycle budget. An application’s cycle budget is usually fixed based on expected data throughput and the size of the block being processed.

```
constraint<std::string>& source(kernel&)
```

This constraint allows you to specify the source file containing the definition of each kernel function. You must specify a source constraint for each kernel.

```
constraint<int>& fifo_depth(connect&)=[<depth> | (depth)]
```

This constraint allows you to specify the amount of slack to be inserted on a streaming connection to allow deadlock free execution.

```
void single_buffer(port<T>&)
```

This constraint allows you to specify single buffer constraint on a buffer port. By default, a buffer port uses double buffering.

```
template <typename _XILINX_ADF_T=api_impl::variant>
constraint<_XILINX_ADF_T> initial_value(adf::port<adf::input>& p);
```

This constraint allows you to set the initial value for an asynchronous AI Engine input runtime parameter port. It allows the destination kernel to start asynchronously with the specified initial value. You can set both scalar and array runtime parameters using this constraint.

Example scalar: `initial_value(k.in[1])=6;`

Example array: `initial_value(k.in[2])={1,2,3};`

```
constraint<int> stack_size(adf::kernel& k);
```

This constraint allows you to set stack size for individual kernel.

```
constraint<int> heap_size(adf::kernel& k);
```

This constraint allows you to set heap size for individual kernel.
