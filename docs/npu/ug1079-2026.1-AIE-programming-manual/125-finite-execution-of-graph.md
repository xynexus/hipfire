---
title: "Finite Execution of Graph"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Finite-Execution-of-Graph"
toc_id: qKOMIKvpEVMCVQAzgpocFQ
content_id: X96oK7Hc1V_krJAz_An6NQ
---

### Finite Execution of Graph

`graph.run(n)`

AI Engine

`graph.run(n)`

`main()`

`graph.end()`

**Note:** Important:

`graph.wait()`

`graph.run()`

`graph.resume()`

`graph.end()`

`graph.end()`

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

Related information

Graph Objects
