---
title: "Buffer Location Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Location-Constraints"
toc_id: tLmAWK3kS52UWtgS5qko6A
content_id: YpHis_je7n9seMDbtBN7zw
---

### Buffer Location Constraints

The AI Engine compiler tries to automatically allocate buffers for buffers, lookup tables, and runtime parameters in the most efficient manner possible. However, you might want to explicitly control their placement in memory. Similar to the kernels shown previously in this section, buffers inferred on a kernel port can also use location constraints. You can map them to specific tiles, banks, or address offsets. The following example illustrates this.

```
#include <adf.h>
#include "kernels.h"
#define NUMCORES (COLS*ROWS)

template <int COLS, int ROWS, int STARTCOL, int STARTROW>
class indep_nodes_graph2 : public adf::graph {
 public:
   adf::kernel kr[NUMCORES];
   adf::port<input> datain[NUMCORES] ;
   adf::port<output> dataout[NUMCORES] ;

 indep_nodes_graph() {
  for (int i = 0; i < COLS; i++) {
    for (int j = 0; j < ROWS; j++) {
      int k = i*ROWS + j;
      kr[k] = adf::kernel::create(mykernel);
      adf::source(kr[k])  = "kernels/kernel.cc";
      adf::runtime<ratio>(kr[k]) = 0.9;
      adf::location<adf::kernel>(kr[k]) = adf::tile(STARTCOL+i, STARTROW+j); // kernel location
      adf::location<adf::buffer>(kr[k].in[0]) =
        { adf::address(STARTCOL+i, STARTROW+j, 0x0),
          adf::address(STARTCOL+i, STARTROW+j, 0x2000) };          // double buffer location
      adf::location<adf::stack>(kr[k]) = adf::bank(STARTCOL+i, STARTROW+j, 2); // stack location
      adf::location<adf::buffer>(kr[k].out[0]) = adf::location<adf::kernel>(kr[k]); // relative buffer location
    }
  }

  for (int i = 0; i < NUMCORES; i++) {
    adf::connect(datain[i], kr[i].in[0]);
    adf::connect(kr[i].out[0], dataout[i]);
  }
 };
};
```

In the previous code, the location of double buffers at port `kr[k].in[0]` is constrained to the specific memory tile address offsets that are created using the `address(col,row,offset)` constructor. Furthermore, the location of the system memory (including the sync buffer, stack and static heap) for the processor that executes kernel instance `kr[k]` is constrained to a particular bank using the `bank(col,row,bankid)` constructor. Finally, the tile location of the buffers connected to the port `kr[k].out[0]` is constrained to be the same tile as that of the kernel instance `kr[k]`. Buffer location constraints are applied on kernel buffer ports.

`--constraints`

AI Engine

```
{
  "PortConstraints": {
    "gr.k[0].in[0]": {
      "buffers": [{
          "column": 16,
          "row": 1,
          "bankId": 0,
          "offset": 16320
        }, {
          "column": 16,
          "row": 1,
          "bankId": 3,
          "offset": 0
        }]
    }
  }
}
```

```
{
  "PortConstraints": {
    "gr.fir24.in[1]": {
      "buffers": [{
          "column": 17,
          "row": 1,
          "bankId": 0,
          "offset": 16320
        }, {
          "column": 17,
          "row": 1,
          "bankId": 3,
          "offset": 0
        }, {
          "column": 18,
          "row": 1,
          "bankId": 0,
          "offset": 16224
        }]
    }
  }
}
```

**Note:** Important:

Related reference

Adaptive Data Flow Graph Specification Reference
