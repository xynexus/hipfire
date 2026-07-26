---
title: "Parallel Graph Execution"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Parallel-Graph-Execution"
toc_id: Kb9yD_~Ma62howkWlPMQXg
content_id: Ezr8Y~YLcxwRvNhh0eNlUg
---

### Parallel Graph Execution

In the previous API methods, only the `wait()` and `end()` methods are blocking operations that can block the `main` application indefinitely. Therefore, if you declare multiple graphs at the top level, you need to interleave the APIs suitably to execute the graphs in parallel, as shown.

```
#include "project.h"
simpleGraph g1, g2, g3;

int main(void) {
  g1.init(); g2.init(); g3.init();
  g1.run(<num-iter>); g2.run(<num-iter>); g3.run(<num-iter>);
  g1.end(); g2.end(); g3.end();
  return 0;
}
```

**Note:** `run`

`init`

`run`

`end`
