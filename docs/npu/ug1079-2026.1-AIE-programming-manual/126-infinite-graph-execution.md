---
title: "Infinite Graph Execution"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Infinite-Graph-Execution"
toc_id: wwEIOgRoXs7zxiGtQwwGcg
content_id: vWMNP0FMdwahDIVPOGu7bg
---

### Infinite Graph Execution

The following graph control API shows how to run the graph infinitely.

```
#include "project.h"
simpleGraph mygraph;

int main(void) {
  mygraph.init();  // load the graph
  mygraph.run();   // start the graph and return to main
  ......
  return 0;
}
```

The API declares a graph object `mygraph` using a pre-defined graph class called `simpleGraph`. Then, in the `main` application, this graph object initializes and runs.

The `init()` method loads the graph to the AI Engine array at prespecified AI Engine tiles. This includes loading the ELF binaries for each AI Engine, configuring the stream switches for routing, and configuring the DMAs for I/O. It leaves the processors in a disabled state.

The `run()` method starts the graph execution by enabling the processors. This graph runs forever because the number of iterations to be run is not provided to the `run()` method.

`graph::run()` without an argument runs the AI Engine kernels for a previously specified number of iterations (which is infinity by default if the graph is run without any arguments). If the graph is run with a finite number of iterations, for example, `mygraph.run(3);mygraph.run()` the second run call also runs for three iterations. Use `graph::run(-1)` to run the graph infinitely regardless of the previous run iteration setting.
