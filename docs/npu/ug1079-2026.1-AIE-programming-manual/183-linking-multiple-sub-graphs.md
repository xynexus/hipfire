---
title: "Linking Multiple Sub-Graphs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Linking-Multiple-Sub-Graphs"
toc_id: EzyGOPcxKI0GhLMmPkmyrQ
content_id: jen4vugRxO7tb~MGsvIjOw
---

### Linking Multiple Sub-Graphs

You can link several sub-graphs to create a larger test or application graph. You can develop graphs independently from I/Os, then embed them in other graphs that define PLIO and GMIO to connect to external systems. This allows you to independently test your graph and connect it to other graphs to build larger applications.

This method mans you do not need to know the size of the buffers or the type of connection within the sub-graph. This allows you to independently develop the graphs and the kernels within the graph. When compiling the larger test or application graph the compiler automatically adapts various I/O port types with the necessary hardware interposers. For example, if the test graph includes a connect statement that links a stream PLIO input to a buffer input, the compiler handles the connection automatically. It connects the stream PLIO output port of one graph to the buffer input port of another.

```
class TestMoreComplexGraph: public adf::graph {
public:
  adf::input_plio plin;
  adf::output_plio plout;

  SplitGraph AddedGraph;
  MergeGraph AnotherGraph;
  SimplestGraph Simple;

  TestMoreComplexGraph()
  {
    plin = adf::input_plio::create("input2",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin.out[0],AddedGraph.din);

    adf::connect(AddedGraph.dout[0],Simple.din);
    adf::connect(Simple.dout,AnotherGraph.din[0]);
    adf::connect(AddedGraph.dout[1],AnotherGraph.din[1]);

    plout = adf::output_plio::create("output2",adf::plio_64_bits,"data/Output2.txt",500);
    adf::connect(AnotherGraph.dout,plout.in[0]);
  };
};
```

```
TestMoreComplexGraph MoreComplexUnitTest;

int main(int argc, char ** argv) {

  MoreComplexUnitTest.init();
  MoreComplexUnitTest.run(NFRAMES*NITERATIONS);
  MoreComplexUnitTest.end();
  return 0;
}
```

![ucp1676907453942.png](../assets/183-01-ucp1676907453942-png-c51039ebc989.png)

*Figure 1. Graph View*

You do not need to change anything in graph `Simple` when it is connected to `AddedGraph` and `AnotherGraph`. It can be error prone as soon as you start to have many sub-graphs. To avoid errors, it is recommended that `SimplestGraph` graph definition can be in its own file. Any modification to the graph file changes the behavior of `UnitTest` and `MoreComplexUnitTest`.
