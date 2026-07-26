---
title: "Basic Iterative Graph Execution"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Basic-Iterative-Graph-Execution"
toc_id: CwcTA_oz6AmEREG9xgeNcA
content_id: 8476~1rkLciNUC3AOsUzBA
---

### Basic Iterative Graph Execution

The following graph control API shows how to use graph APIs to initialize, run, wait, and terminate graphs for a specific number of iterations. A graph object `mygraph` is declared using a pre-defined graph class called `simpleGraph`. Then, in the `main` application, this graph object is initialized and run.

The `init()` method loads the graph to the AI Engine array at prespecified AI Engine tiles. This includes loading the ELF binaries for each AI Engine, configuring the stream switches for routing, and configuring the DMAs for I/O. It leaves the processors in a disabled state.

The `run()` method starts the graph execution by enabling the processors. Use the `run` API to execute a specific number of iterations of the graph by supplying a positive integer argument at runtime. This form is useful for debugging your graph execution.

```
#include "project.h"
simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(3); // run 3 iterations
  mygraph.wait(); // wait for 3 iterations to finish
  mygraph.run(10); // run 10 iterations
  mygraph.end(); // wait for 10 iterations to finish
  return 0;
}
```

You can use the `wait()` API to wait for the first run to finish before starting the second run. `wait` has the same blocking effect as `end` except that it allows re-running the graph again without having to re-initialize it. Calling `run` back-to-back without an intervening `wait` to finish that run can have an unpredictable effect because the `run` API modifies the loop bounds of the active processors of the graph.

##### Graph Iteration

A graph can have multiple kernels, input and output ports. The graph connectivity, which is equivalent to the nets in a data flow graph is either between the kernels, between kernel and input ports, or between kernel and output ports, and can be configured as a connection. A graph runs for an iteration when it consumes data samples equal to the buffer or stream of data expected by the kernels in the graph, and produces data samples equal to the buffer or stream of data expected at the output of all the kernels in the graph.
