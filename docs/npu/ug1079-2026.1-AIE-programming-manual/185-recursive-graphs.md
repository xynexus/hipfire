---
title: "Recursive Graphs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Recursive-Graphs"
toc_id: kUMzpOj63qg4~vA7i29eJw
content_id: MALaIXhK3uL2gdVSPh3uBg
---

### Recursive Graphs

```
template<int Level>
class RecursiveGraph: public adf::graph {
private:
  adf::kernel k;
  RecursiveGraph<Level-1> RG;

public:
  adf::port<input> din;
  adf::port<output> dout;

  RecursiveGraph() {
    k = adf::kernel::create(passthrough);
    adf::source(k) = "kernels.cpp";
    adf::runtime<ratio>(k) = 0.9;

    adf::connect(din, k.in[0]);
    adf::connect(k.out[0], RG.din);
    adf::connect(RG.dout, dout);
  };
};


template<>
class RecursiveGraph<1>: public adf::graph {
private:
  adf::kernel k;

public:
  adf::port<input> din;
  adf::port<output> dout;

  RecursiveGraph() {
    k = adf::kernel::create(passthrough);
    adf::source(k) = "kernels.cpp";
    adf::runtime<ratio>(k) = 0.9;

    adf::connect(din, k.in[0]);
    adf::connect(k.out[0], dout);
  };
};
```

The class `RecursiveGraph` is parametrized with the template parameter `Level` which represents the depth of the recursive function. Within the `RecursiveGraph` class, an instantiati on of the same class is performed as `RecursiveGraph<Level-1> RG;`with the template parameter `Level-1`. This will recursively instantiate `RecursiveGraph` `Level-1` times. The `RecursiveGraph<1>` class specialization behaves as the stop condition of the recursion.

```
class TestRecursiveGraph: public adf::graph {
public:
  adf::input_plio plin;
  adf::output_plio plout;

  RecursiveGraph<5> Recursive;

  TestRecursiveGraph()
  {
    plin = adf::input_plio::create("input",adf::plio_64_bits,"data/Input_64.txt",500);
    adf::connect(plin.out[0],Recursive.din);

    plout = adf::output_plio::create("output",adf::plio_64_bits,"data/Output.txt",500);
    adf::connect(Recursive.dout,plout.in[0]);
  };
};
```

In the following example, five instances of `RecursiveGraph` are instantiated in the `TestRecursiveGraph` graph.

```
TestRecursiveGraph RecursiveUnitTest;

int main(int argc, char ** argv) {

  RecursiveUnitTest.init();
  RecursiveUnitTest.run(NFRAMES*NITERATIONS);
  RecursiveUnitTest.end();
  return 0;
}
```

The recursive graph as opposed to a sequential graph view obtained in the Vitis IDE is the following:

![las1676909368478.png](../assets/185-01-las1676909368478-png-4ace4447d396.png)

*Figure 1. Recursive Graph and Sequential Multi-kernel Graph View*

The hierarchy obtained using recursivity is clearly visible in the graph view.
