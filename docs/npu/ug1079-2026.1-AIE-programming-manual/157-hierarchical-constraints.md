---
title: "Hierarchical Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Hierarchical-Constraints"
toc_id: TB6jRPDh8IYnSnos5ke3jw
content_id: rGEYE5YRE2LRnwPGiGPbMg
---

### Hierarchical Constraints

When creating complex graphs with multiple subgraph classes or instances, you can apply location constraints to each kernel or port instance individually. Apply these constraints during subgraph instantiation instead of during the definition. In this case, you need to specify the graph qualified name of that kernel instance or kernel port instance in the constraint as shown in the following example. Ensure kernels or their ports being constrained as described previously are defined as public members of the subgraph.

```
class ToplevelGraph : public adf::graph {
 public:
  indep_nodes_graph1<1,2,3,2> mygraph;
  adf::port<adf::input> datain[2] ;
  adf::port<adf::output> dataout[2] ;

  ToplevelGraph() {
    for (int i = 0; i < 2; i++) {
      adf::connect(datain[i], mygraph.datain[i]);
      adf::connect(mygraph.dataout[i], dataout[i]);

      // hierarchical constraints
      adf::location<adf::stack>(mygraph.kr[i]) = adf::bank(3, 2+i, 2);
      adf::location<adf::buffer>(mygraph.kr[i].out[0]) = adf::location<adf::kernel>(mygraph.kr[i]);
    }
  };
};
```

**Note:** Work

```
v++ -c --mode aie --constraints Work/temp/graph_aie_mapped.aiecst src/graph.cpp
```

To reuse FIFO location constraints, use `aie_routed.aiecst` file.

```
v++ -c --mode aie --constraints Work/temp/graph_aie_routed.aiecst src/graph.cpp
```
