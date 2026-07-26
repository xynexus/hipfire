---
title: "kernel"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/kernel"
toc_id: MalS2HcTVL3IsF3B48raOw
content_id: Zd9H6pG7oR5BDtbpLvaYKw
---

### kernel

This class represents a single node of the graph. User-defined graph types contain kernel objects as member variables that wrap over some C function computation mapped to the AI Engine array.

##### Scope

You can declare `kernel` objects in class scope as member variables in a user-defined graph type (i.e., inside a class that inherits from `graph`).

You must initialize `kernel` objects by assignment in the graph constructor.

##### Member Functions

```
static kernel & create( function );
```

The static create method creates a kernel object from a C kernel function. It automatically determines how many input ports and output ports each kernel has and their appropriate element type. Any other arguments in a kernel function are treated as runtime parameters or lookup tables, passed to the kernel on each invocation. Runtime parameters are passed by value, while lookup tables are passed by reference each time the kernel is invoked by the compiler generated static-schedule.

```
kernel & operator()(…)
```

Takes one or more `parameter` objects as arguments. The number of `parameter` arguments must match the number of non-buffer formal arguments in the kernel function used to construct the `kernel`. When used in the body of a graph constructor to assign to a `kernel` member variable, the operator ensures that updated parameter arguments are passed to the kernel function on every invocation.

##### Member Variables

```
std::vector<port<input>> in;
```

This variable provides access to the logical inputs of a kernel, allowing user graphs to specify connections between kernels in a graph. The i'th index selects the i’th input port (buffer, stream, or rtp) declared in the kernel function arguments.

```
std::vector<port<output>> out;
```

This variable provides access to the logical outputs, allowing user graphs to specify connections between kernels in a graph. The i'th index selects the i’th output port (buffer or stream) declared in the kernel function arguments.

```
std::vector<port<inout>> inout;
```

This variable provides access to the logical inout ports, allowing user graphs to specify connections between kernels in a graph. The i'th index selects the i’th inout port (rtp) declared in the kernel function arguments.
