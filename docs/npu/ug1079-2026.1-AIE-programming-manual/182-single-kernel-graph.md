---
title: "Single Kernel Graph"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Single-Kernel-Graph"
toc_id: 0EcHMlzwlnbIg3MHfQ1jtQ
content_id: lEwPDVSqSEyIFKMVNnoHGA
---

### Single Kernel Graph

The simplest graph is a single kernel instantiated in a class that inherits from the `adf::graph` class with specified inputs and outputs.

```
class SimplestGraph: public adf::graph {
private:
  adf::kernel k;

public:
  adf::port<input> din;
  adf::port<output> dout;

SimplestGraph() {
  k = adf::kernel::create(passthrough);
  adf::source(k) = "passthrough.cpp";
  adf::runtime<ratio>(k) = 0.9;
  adf::connect(din, k.in[0]);
  adf::connect(k.out[0], dout);
  dimensions(k.in[0]) = {FRAME_LENGTH};
  dimensions(k.out[0]) = {FRAME_LENGTH};
  };
};
```

`Input_64.txt`

`Output1.txt`

`Simple`

`TestGraph`

```
class TestGraph: public adf::graph {
public:
  adf::input_plio plin1;
  adf::output_plio plout1;

  SimplestGraph Simple;

  TestGraph()
  {
    plin1 = adf::input_plio::create("input1",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin1.out[0],Simple.din);

    plout1 = adf::output_plio::create("output1",adf::plio_64_bits,"data/Output1.txt",500);
    adf::connect(Simple.dout,plout1.in[0]);
  };
};
```

The test bench program instantiates the test graph and calls the control commands to initialize run and end the graph.

```
#include "graph.h"

TestGraph UnitTest;

int main(int argc, char ** argv) {
  UnitTest.init();
  UnitTest.run(NFRAMES*NITERATIONS);
  UnitTest.end();
  return 0;
}
```

The resulting graph display in the Vitis IDE:

![fvo1676906659992.png](../assets/182-01-fvo1676906659992-png-fb4a4d990bc4.png)

*Figure 1. Graph View*
