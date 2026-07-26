---
title: "Parameter Update/Read Using Graph APIs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Parameter-Update/Read-Using-Graph-APIs"
toc_id: XVSF43XQ_6GQ85kkdJ35CA
content_id: EV6lDuEMpioqqvJvuG3zgw
---

### Parameter Update/Read Using Graph APIs

In default compilation mode, the `main` application compiles as a separate control thread. The control thread must execute on the PS in parallel with the graph executing on the AI Engine array. The `main` application can use update and read APIs to access runtime parameters declared within the graphs at any level. This section describes these APIs using examples.

##### Synchronous Update/Read

The following code shows the `main` application of the `simple_param` graph described in Specifying Runtime Data Parameters.

```
#include "param.h"
parameterGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(2);

  mygraph.update(mygraph.select_value, 23);
  mygraph.update(mygraph.select_value, 45);

  mygraph.end();
  return 0;
}
```

In this example, the graph `mygraph` initializes first and then runs for two iterations. It has a triggered input parameter port `select_value` that must be updated with a new value for each invocation of the receiving kernel. The first argument of the `update` API identifies the port to update and the second argument provides the value. Other update APIs are supported based on the direction of the port, its data type, and whether it is a scalar or array parameter.

If the program is compiled with a fixed number of test iterations, then for triggered parameters the number of update API calls in the `main` program must match the number of test iterations, otherwise the simulation could be waiting for additional updates. For asynchronous parameters, updates occur asynchronously with the graph execution. If no update is made, the kernel uses the previous value.

If the previous graph was compiled with a synchronous inout parameter, you must interleave the update and read calls. The following example shows this.

```
#include "param.h"
parameterGraph mygraph;

int main(void) {
  int result0, result1;
  mygraph.init();
  mygraph.run(2);

  mygraph.update(mygraph.select_value, 23);
  mygraph.read(mygraph.result_out, result0);
  mygraph.update(mygraph.select_value, 45);
  mygraph.read(mygraph.result_out, result1);

  mygraph.end();
  return 0;
}
```

In this example, it is assumed that the graph produces a scalar result every iteration through the inout port `result_out`. The `read` API reads out the value of the port synchronously after each iteration. The first argument of the `read` API is the graph inout port to be read back. The second argument is the location where the value is stored (passed by reference).

The synchronous protocol ensures that the read operation will wait for the value to be produced by the graph before sampling it and the graph will wait for the value to be read before proceeding to the next iteration. This is why it is important to interleave the `update` and `read` operations.

##### Asynchronous Update/Read

When an input parameter is specified with asynchronous protocol, the kernel execution waits for the first update to happen for parameter initialization. However, an arbitrary number of kernel invocations can take place before the next update. This is usually the intent of the asynchronous update during application deployment. However, for debugging, `wait` API can be used to finish a predetermined set of iterations before the next update as shown in the following example.

```
#include "param.h"
asyncGraph mygraph;

int main(void) {
  int result0, result1;
  mygraph.init();

  mygraph.update(mygraph.select_value, 23);
  mygraph.run(5);
  mygraph.wait();
  mygraph.update(mygraph.select_value, 45);
  mygraph.run(15);
  mygraph.end();
  return 0;
}
```

In the previous example, after the initial update, the system runs five iterations to completion. Then it performs another update followed by another set of 15 iterations. If the graph uses asynchronous inout ports, you can read back data immediately after the wait (or end).

Another template for asynchronous updates is to use timeouts in `wait` API as shown in the following example.

```
#include "param.h"
asyncGraph mygraph;

int main(void) {
  int result0, result1;
  mygraph.init();
  mygraph.run();
  mygraph.update(mygraph.select_value, 23);
  mygraph.wait(10000);
  mygraph.update(mygraph.select_value, 45);
  mygraph.resume();
  mygraph.end(15000);
  return 0;
}
```

This example sets the graph up to run forever. However, after the `run` API is called, it still blocks for the first update to happen for parameter initialization. Then, it runs for 10,000 cycles (approximately) before allowing the control thread to make another update. The new update takes effect at the next kernel invocation boundary. Then the graph is allowed to run for another 15,000 cycles before terminating.

Related reference

Adaptive Data Flow Graph Specification Reference
