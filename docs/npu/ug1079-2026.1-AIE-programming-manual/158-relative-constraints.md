---
title: "Relative Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Relative-Constraints"
toc_id: Ys5BuotDnfWnc~HXTMOLqA
content_id: 8OHxoc1sXN9bpvOzfE1DCw
---

### Relative Constraints

Kernel, PLIO, GMIO and shared buffer objects can be placed relative to each other. These types of constraints are called relative constraints. The constraints allow for placement of a destination node relative to the source node, and is specified as a row or column offset in 2-D grid. Specify either the row offset or column offset. If you don not specify the row or column offset, it is "don't care."

`adf::relative_offset`

```
k = adf::kernel::create(matrix_mul);
k1 = adf::kernel::create(shuffle_4x16);
in0=adf::input_plio::create("Datain0", adf::plio_64_bits, "data/matA.txt");

adf::location<adf::kernel>(k1) = adf::location<adf::kernel>(k) + adf::relative_offset({.col_offset = 1, .row_offset= 1});
adf::location<adf::PLIO>(in0) = adf::location<adf::kernel>(k) + adf::relative_offset({.col_offset = 0});

adf::shared_buffer<int32> mtxA,mtxB;
adf::location<adf::buffer>(mtxA) = adf::location<adf::buffer>(mtxB) + adf::relative_offset({.col_offset = 1, .row_offset=1});
adf::location<adf::kernel>(adf::first) = adf::location<adf::buffer>(mtxA) + adf::relative_offset({.col_offset = 1});
```

**Note:** Shared buffers only support relative column offsets with other types.
