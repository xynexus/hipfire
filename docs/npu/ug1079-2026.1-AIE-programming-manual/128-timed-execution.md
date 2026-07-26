---
title: "Timed Execution"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Timed-Execution"
toc_id: t8~wBs5xp9_g1ie4dCC7Cg
content_id: O41vzywbd41~OLkOjocy2g
---

### Timed Execution

In multi-rate graphs, all kernels need not execute for the same number of iterations. In such situations, a timed execution model is more suitable for testing.

There are variants of the `wait` and `end` APIs with a positive integer that specifies a cycle timeout. This is the number of AI Engine cycles that the API call blocks before disabling the processors and returning. The blocking condition does not depend on any graph termination event. The graph can be in an arbitrary state at the expiration of the timeout.

```
#include "project.h"
simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run();
  mygraph.wait(10000);  // wait for 10000 AI Engine cycles
  mygraph.resume();     // continue executing
  mygraph.end(15000);   // wait for another 15000 cycles and terminate
}
```

**Note:** `resume()`

`resume`

AI Engines

`resume`

AI Engine
