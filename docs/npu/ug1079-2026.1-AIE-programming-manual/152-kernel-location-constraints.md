---
title: "Kernel Location Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Location-Constraints"
toc_id: sSDoILNdBwgiO7MuynqSWw
content_id: IqnGed_ZZEMdzlLy2HEqmQ
---

### Kernel Location Constraints

When building large graphs with multiple subgraphs, it can be useful to control the exact mapping of kernels to AI Engines. You can do this either relative to other kernels or in an absolute sense. The AI Engine compiler lets you specify location constraints for kernels. Combined with the C++ template class specification, this creates a robust, scalable, and predictable mapping of your graph to the AI Engine array. It also reduces the choices for the mapper to try, which can considerably speed up the mapper. Consider the following graph specification:

```
#include <adf.h>
#include "kernels.h
#define NUMCORES (COLS*ROWS)

template <int COLS, int ROWS, int STARTCOL, int STARTROW>
class indep_nodes_graph1 : public adf::graph {
 public:
   adf::kernel kr[NUMCORES];
   adf::port<input> datain[NUMCORES] ;
   adf::port<output> dataout[NUMCORES] ;

 indep_nodes_graph1() {
  for (int i = 0; i < COLS; i++) {
    for (int j = 0; j < ROWS; j++) {
      int k = i*ROWS + j;
      kr[k] = adf::kernel::create(mykernel);
      adf::source(kr[k])  = "kernels/kernel.cc";
      adf::runtime<ratio>(kr[k]) = 0.9;
      adf::location<adf::kernel>(kr[k]) = adf::tile(STARTCOL+i, STARTROW+j);
    }
  }
  for (int i = 0; i < NUMCORES; i++) {
    adf::connect(datain[i], kr[i].in[0]);
    adf::connect(kr[i].out[0], dataout[i]);
  }
 };
};
```

The template parameters identify a COLS x ROWS logical array of kernels (COLS x ROWS = NUMCORES) that are placed within a larger logical device of some dimensionality starting at (STARTCOL, STARTROW) as the origin. Each kernel in that graph is constrained to be placed on a specific AI Engine. This is accomplished using an absolute location constraint for each kernel placing it on a specific processor tile. For example, the following declaration creates a 1 x 2 kernel array starting at offset (3,2). When embedded within a 4 x 4 logical device topology, the kernel array is constrained to the top right corner.

```
indep_nodes_graph1<1,2,3,2> mygraph;
```

The physical port names can differ from the code names. To add a location constraints check the Variable column in reports/*mapping_analysis_report.txt.

**Note:** Important:

`location<absolute>(k)`

`proc(x,y)`

`adf::location<adf::kernel>(k)`

`adf::tile(x,y)`

Related reference

Adaptive Data Flow Graph Specification Reference
