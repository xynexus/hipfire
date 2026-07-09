---
title: "port<T>"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/port-T"
toc_id: 3MC0hxSFqcKYE0puBGGymw
content_id: zBv3OWiAxFR7ystljvUiZA
---

### port<T>

##### Scope

Objects of type `port<T>` are port objects. You can declare these in class scope as member variables of a user-defined graph type (i.e., member variables of a class that inherits from `graph`). Alternatively define them implicitly for a kernel according to its function signature. The template parameter `T` can be one of `input`, `output`, or `inout`.

##### Aliases

`input_port` is an alias for the type `port<input>`.

`output_port` is an alias for the type `port<output>`.

`inout_port` is an alias for the type `port<inout>`.

##### Purpose

Used to connect between kernels within a graph and across levels of hierarchy in customer specification containing platform, graphs, and subgraphs.

##### Operators

```
port<T>& negate(port<T>&)
```

When applied to a destination port within a connection, this operator inverts the Boolean semantics of the connected source port. Therefore, it has the effect of converting a 0 to 1 and 1 to 0.

```
port<T>& async(port<T>&)
```

When applied to a destination RTP port within a connection, this operator specifies an asynchronous update of the destination port's RTP buffer from the source port that it is connected to or from the external control application if the source is a graph port left unconnected. Therefore, the receiving kernel does not wait for the value for each invocation, rather it uses previous value stored in the corresponding buffer.

```
port<T>& sync(port<T>&)
```

When applied to a source RTP port within a connection, this operator specifies a synchronous read of the source port's RTP buffer from the destination port that it is connected to or from the external control application if the destination is a graph port left unconnected. Therefore, the receiving kernel waits for a new value to be produced for each invocation of the producing kernel.

Related reference

Logical I/O Ports
