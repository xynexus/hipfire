---
title: "Hierarchical Graphs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Hierarchical-Graphs"
toc_id: 17aar2PIE7zIjEmrLySf9Q
content_id: ~aQzer3TWWouyFE0gFNujw
---

### Hierarchical Graphs

```
template <int NK>
class MultiKernelGraph: public adf::graph {
private:
  adf::kernel k[NK];

public:
  adf::port<input> din;
  adf::port<output> dout;

  MultiKernelGraph() {
    for(int i=0;i<NK;i++)
    {
      k[i] = adf::kernel::create(passthrough);
      adf::source(k[i]) = "kernels.cpp";

      adf::runtime<ratio>(k[i]) = 0.9;
    }
    adf::connect(din, k[0].in[0]);
    for(int i=0;i<NK-1;i++)
      adf::connect(k[i].out[0], k[i+1].in[0]);
    adf::connect(k[NK-1].out[0], dout);
  };
};
```

```
class TestGraphMulti: public adf::graph {
public:
  adf::input_plio plin;
  adf::output_plio plout;

  MultiKernelGraph<3> Multi;

  TestGraphMulti()
  {
    plin = adf::input_plio::create("input2",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin.out[0],Multi.din);

    plout = adf::output_plio::create("output2",adf::plio_64_bits,"data/Output2.txt",500);
    adf::connect(Multi.dout,plout.in[0]);
  };
};
```

![noc1664183008580.png](../assets/184-01-noc1664183008580-png-67e8aeb8eb3d.png)

*Figure 1. Graph View*

As seen earlier, a graph can contain kernels and sub-graphs. The test bench .cpp code can instantiate multiple graphs and run them independently.

```
TestGraphSimple Simple_UnitTest;
TestGraphMulti Multi_UnitTest;

int main(int argc, char ** argv) {

  Simple_UnitTest.init();
  Multi_UnitTest.init();

  Simple_UnitTest.run(NFRAMES*NITERATIONS);
  Multi_UnitTest.run(NFRAMES*NITERATIONS);

  Simple_UnitTest.end();
  Multi_UnitTest.end();
  return 0;
}
```

The Vitis IDE displays two independent graphs on the graph view.

![aci1676908385299.png](../assets/184-02-aci1676908385299-png-a7ea4652980c.png)

*Figure 2. Multiple Graphs*

You can also instantiate these two graphs in a third graph which is instantiated in the test bench:

```
class TestGraph: public adf::graph {
public:
  adf::input_plio plin1,plin2;
  adf::output_plio plout1,plout2;

  SimplestGraph Simple;
  MultiKernelGraph Multi;

  TestGraph()
  {
    plin1 = adf::input_plio::create("input1",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin1.out[0],Simple.din);

    plout1 = adf::output_plio::create("output1",adf::plio_64_bits,"data/Output1.txt",500);
    adf::connect(Simple.dout,plout1.in[0]);

    plin2 = adf::input_plio::create("input2",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin2.out[0],Multi.din);

    plout2 = adf::output_plio::create("output2",adf::plio_64_bits,"data/Output2.txt",500);
    adf::connect(Multi.dout,plout2.in[0]);

  };
};
```

```
TestGraph UnitTest;

int main(int argc, char ** argv) {

  UnitTest.init();
  UnitTest.run(NFRAMES*NITERATIONS);
  UnitTest.end();
  return 0;
}
```

![ivd1676908448639.png](../assets/184-03-ivd1676908448639-png-025ab1ea712b.png)

*Figure 3. Composite Graph View*

You can build complex graphs with sequential and parallel kernels. These graphs can split and merge data multiple times using buffers and streams across the AI Engine array.

```
VeryComplexGraph() {
  // Declare Kernels
  for(int i=0;i<2;i++)
  {
    simple[i] = adf::kernel::create(passthrough);
    adf::source(simple[i]) = "kernels.cpp";
    adf::runtime<ratio>(simple[i]) = 0.9;
  }

  // Connections
  // Inputs
  connect(din[0],split[0].din);
  connect(din[1],split[1].din);

  // Outputs
  connect(merge[2].dout,dout[0]);
  connect(split[3].dout[0],dout[1]);
  connect(split[3].dout[1],dout[2]);

  // Internal connections
  connect(split[0].dout[0],multi3.din);
  connect(multi3.dout,merge[2].din[0]);
  connect(split[0].dout[1],merge[0].din[0]);
  connect(split[1].dout[0],merge[0].din[1]);
  connect(merge[0].dout,simple[0].in[0]);
  connect(simple[0].out[0],split[2].din);
  connect(split[2].dout[0],multi2[1].din);
  connect(multi2[1].dout,merge[2].din[1]);
  connect(split[2].dout[1],simple[1].in[0]);
  connect(simple[1].out[0],merge[1].din[0]);
  connect(split[1].dout[1],multi2[0].din);
  connect(multi2[0].dout,merge[1].din[1]);
  connect(merge[1].dout,split[3].din);
};
```

![uvn1677060517683.png](../assets/184-04-uvn1677060517683-png-f417f4282cc3.png)

*Figure 4. Complex Graph View*

In this graph view, a complex dataflow diverges and converges in multiple points. If part of the dataflow needs to pass through the PL, you see matching input and output ports. Connect these ports to your PL kernels during the link phase.
