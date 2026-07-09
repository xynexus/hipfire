---
title: "Shared Graph-Scoped Tables"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Shared-Graph-Scoped-Tables"
toc_id: fJEaMmj1v~~078Jp6Swi5g
content_id: We9EQ~qURxW0SOyiY25fYQ
---

### Shared Graph-Scoped Tables

Sometimes, multiple kernels use the same table definition. The AI Engine architecture is a distributed address-space architecture. This means each processor binary image that executes such a kernel needs to have that table defined in its own local memory. To ensure correct graph linkage spread across multiple processors, declare the tables as `extern` in both the kernel source file and the graph class definition file. Then, specify the table definition in a separate header file that is attached as a property to the kernel as shown below.

```
#include <adf.h>

extern int16 lutarray[8]; // extern in graph context for parameter::array binding

class shared_table_graph : public adf::graph {
public:
  adf::kernel k1;
  adf::kernel k2;

  // One parameter per kernel consumer so we can place each in the correct tile
  adf::parameter p1;
  adf::parameter p2;

  adf::input_plio in1, in2;
  adf::output_plio out1, out2;

  shared_table_graph() {
    // Create kernels
    k1 = adf::kernel::create(simple1);
    k2 = adf::kernel::create(simple2);

    // Create parameters bound to the same symbol
    p1 = adf::parameter::array(lutarray);
    p2 = adf::parameter::array(lutarray);

    // Connect parameters to kernels (graph-level linkage)
    adf::connect(p1, k1);
    adf::connect(p2, k2);
```

This ensures the header file that defines the table is included in the final binary link wherever this kernel is used, without causing re-definition errors.

**Note:** Large lookup tables (>32 KB) are not supported.

**Note:** - Managed explicitly as runtime parameters, or
- Declared at the file scope, which is shared across all kernels mapped to the same AI Engine.
