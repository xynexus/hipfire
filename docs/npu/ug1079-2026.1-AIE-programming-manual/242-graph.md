---
title: "graph"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/graph"
toc_id: utNZA24QPQ2lA41kMRQjzA
content_id: XicDd7x0aCSyRlV7TkXvSg
---

### graph

This is the main graph abstraction exported by the ADF tools. Ensure user-defined graphs are inherited from `class graph`.

##### Scope

You mus declare all instances of those user-defined graph types that form part of a user design in global scope. Declare these under any name space.

##### Member Functions

```
virtual return_code init() ;
```

This method loads and initializes a precompiled graph object onto the AI Engine array using a predetermined set of processor tiles. Currently, no relocation is supported. All existing information in the program memory, data memory, and stream switches belonging to the tiles being loaded is replaced. The loaded processors are left in a disabled state.

```
virtual return_code run();
virtual return_code run(unsigned int num_iterations);
```

This method enables the processors associated with a graph to start execution from the beginning of their respective `main` programs. Without any arguments, the graph runs forever. The API with arguments can set the number of iterations for each run differently. This is a non-blocking operation on the PS application.

```
virtual return_code end();
virtual return_code end(unsigned int cycle_timeout);
```

The `end` method waits for the termination of the graph. A graph is considered to be terminated when all its active processors exit their `main` thread and disable themselves. This is a blocking operation for the PS application. This method also cleans up the state of the graph. It forces the release of all locks and cleans up the stream switch configurations used in the graph.

The `end` method with cycle timeout terminates and cleans up the graph when the timeout expires rather than waiting for any graph related event. Attempting to `run` the graph after `end` without re-initializing it can give unpredictable results.

```
virtual return_code wait();
virtual return_code wait(unsigned int cycle_timeout);

virtual return_code resume();
```

The `wait` method pauses graph execution temporarily without cleaning up its state so that it can be restarted with a `run` or `resume` method. The `wait` method without arguments is useful when waiting for a previous `run` with a fixed number of iterations to finish. This can be followed by another `run` with a new set of iterations.

The `wait` method with cycle timeout pauses the graph execution when the timeout expires counted from a previous `run` or `resume` call. Follow this with a `resume` to let the graph to continue to execute. Attempting to `run` after a `wait` with cycle timeout call can lead to unpredictable results because the graph can be paused in an unpredictable state and the `run` can restart the processors from the beginning of their `main` programs.

```
virtual return_code update(input_port& pName, <type> value);

virtual return_code update(input_port& pName, const <type>* value, size_t size);
```

These methods are various forms of runtime parameter update APIs that can be used to update scalar or array runtime parameter ports. The port name is a fully qualified path name such as `graph1.graph2.port` or `graph1.graph2.kernel.port`. The `<type>` can be one of `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `cint16`, `cint32`, `float`, `cfloat`. For array runtime parameter updates, a `size` argument specifies the number of elements in the array to be updated. `size` must match the RTP array size defined in the graph, so the entire RTP array updates at the same time.

```
virtual return_code read(input_port& pName, <type>& value);

virtual return_code read(input_port& pName, <type>* value, size_t size);
```

These methods are various forms of runtime parameter read APIs that can be used to read scalar or array runtime parameter ports. The port name is a fully qualified path name such as `graph1.graph2.port`or `graph1.graph2.kernel.port`. The `<type>` can be one of `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `cint16`, `cint32`, `float`, `cfloat`. For array runtime parameter reads, a `size` argument specifies the number of elements in the array to be read.
