---
title: "Creating a Data Flow Graph (Including Kernels)"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Creating-a-Data-Flow-Graph-Including-Kernels"
toc_id: x9m0A8vXBccSMW5IiJx6EA
content_id: xAhQ~atExK0NVEBClZZBEw
---

## Creating a Data Flow Graph (Including Kernels)

1. Define your application graph class in a separate header file (for example `project.h`). First, add the Adaptive Data Flow (ADF) header (adf.h) and include the kernel function prototypes. The ADF library includes all the required constructs for defining and executing the graphs on AI Engines.`#include <adf.h> #include "kernels.h"`

  ```
  #include <adf.h>
  #include "kernels.h"
  ```
2. Define your graph class using objects from the `adf` namespace. All user graphs derive from the class `graph`.`include <adf.h> #include "kernels.h" class simpleGraph : public adf::graph { private: adf::kernel first; adf::kernel second; };` This is the beginning of a graph class definition that declares two kernels (`first` and `second`).

  ```
  include <adf.h>
  #include "kernels.h"

  class simpleGraph : public adf::graph {
  private:
    adf::kernel first;
    adf::kernel second;
  };
  ```

  This is the beginning of a graph class definition that declares two kernels (`first` and `second`).
3. Add some top-level input/output objects, `input_plio` and `output_plio`, to the graph.`#include <adf.h> #include "kernels.h" class simpleGraph : public adf::graph { private: adf::kernel first; adf::kernel second; public: adf::input_plio in; adf::output_plio out; };`

  ```
  #include <adf.h>
  #include "kernels.h"

  class simpleGraph : public adf::graph {
  private:
    adf::kernel first;
    adf::kernel second;
  public:
    adf::input_plio in;
    adf::output_plio out;

  };
  ```
4. Use the `kernel::create` function to instantiate the `first` and `second` C++ kernel objects using the functionality of the C function `simple`.`#include <adf.h> #include "kernels.h" class simpleGraph : public adf::graph { private: adf::kernel first; adf::kernel second; public: adf::input_plio in; adf::output_plio out; simpleGraph() { first = adf::kernel::create(simple); second = adf::kernel::create(simple); } };`

  ```
  #include <adf.h>
  #include "kernels.h"

  class simpleGraph : public adf::graph {
  private:
    adf::kernel first;
    adf::kernel second;
  public:
    adf::input_plio in;
    adf::output_plio out;
    simpleGraph() {
        first = adf::kernel::create(simple);
        second = adf::kernel::create(simple);
    }
  };
  ```
5. Configure input and output PLIO objects with the specified width and input/output files. Add connectivity information, which is equivalent to nets in a data flow graph. In this description, input/output objects are referenced by indices.The first input buffer or stream argument in the `simple` function has index 0 in an array of input ports (`in`). Subsequent input arguments take ascending consecutive indices. The first output buffer or stream argument in the `simple` function is assigned index 0 in an array of output ports (`out`). Subsequent output arguments take ascending consecutive indices. `#include <adf.h> #include "kernels.h" class simpleGraph : public adf::graph { private: adf::kernel first; adf::kernel second; public: adf::input_plio in; adf::output_plio out; simpleGraph() { first = adf::kernel::create(simple); second = adf::kernel::create(simple); in = adf::input_plio::create(plio_32_bits, "data/input.txt"); out = adf::output_plio::create(plio_32_bits, "data/output.txt"); adf::connect(in.out[0], first.in[0]); adf::connect(first.out[0], second.in[0]); adf::connect(second.out[0], out.in[0]); adf::dimensions(first.in[0]) = {128}; adf::dimensions(first.out[0]) = {128}; adf::dimensions(second.in[0]) = {128}; adf::dimensions(second.out[0]) = {128}; } };` ![dnz1645125744064.png](../assets/073-01-dnz1645125744064-png-1a6e52dc5221.png) This figure represents the graph connectivity specified in the previous graph code. Graph connectivity can be viewed when you open the compilation results in the Vitis IDE. For more information, see [Viewing AI Engine Compilation Summary Results](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1702-vitis-accelerated-reference&resourceid=yds1589213243641.html) in the Vitis Reference Guide ([UG1702](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1702-vitis-accelerated-reference)). As shown in the previous figure, the input port from the top level is connected into the input port of the first kernel, the output port of the first kernel is connected to the input port of the second kernel, and the output port of the second kernel is connected to the output exposed to the top level. The first kernel executes when 128 bytes of data (32 complex samples) are collected in a buffer from an external source. This is specified using a `dimensions(first.in[0])={128}` construct. Likewise, the second kernel executes when its input buffer has valid data being produced as the output of the first kernel. Finally, the output of the second kernel is connected to the top-level output port and the `dimensions(second.out[0])={128}` specifies the number of samples of data kernels produced upon termination. `buf0` and `buf0d` are ping pong buffers allocated for the `first` kernel input buffer. Similarly, `buf2` and `buf2d` are ping pong buffers allocated for the `second` kernel output buffer. Notice that `buf1` which is the buffer output from the `first` kernel to the `second` kernel is not a ping pong buffer. This is because both the `first` and `second` kernel buffers are in a single AI Engine tile where they execute sequentially. See Runtime Ratio for more information.

  The first input buffer or stream argument in the `simple` function has index 0 in an array of input ports (`in`). Subsequent input arguments take ascending consecutive indices. The first output buffer or stream argument in the `simple` function is assigned index 0 in an array of output ports (`out`). Subsequent output arguments take ascending consecutive indices.

  ```
  #include <adf.h>
  #include "kernels.h"

  class simpleGraph : public adf::graph {
  private:
    adf::kernel first;
    adf::kernel second;
  public:
    adf::input_plio in;
    adf::output_plio out;

    simpleGraph() {
      first = adf::kernel::create(simple);
      second = adf::kernel::create(simple);

      in = adf::input_plio::create(plio_32_bits, "data/input.txt");
      out = adf::output_plio::create(plio_32_bits, "data/output.txt");
      adf::connect(in.out[0], first.in[0]);
      adf::connect(first.out[0], second.in[0]);
      adf::connect(second.out[0], out.in[0]);
      adf::dimensions(first.in[0]) = {128};
      adf::dimensions(first.out[0]) = {128};
      adf::dimensions(second.in[0]) = {128};
      adf::dimensions(second.out[0]) = {128};
    }
  };
  ```

  This figure represents the graph connectivity specified in the previous graph code. Graph connectivity can be viewed when you open the compilation results in the Vitis IDE. For more information, see [Viewing AI Engine Compilation Summary Results](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1702-vitis-accelerated-reference&resourceid=yds1589213243641.html) in the Vitis Reference Guide ([UG1702](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1702-vitis-accelerated-reference)). As shown in the previous figure, the input port from the top level is connected into the input port of the first kernel, the output port of the first kernel is connected to the input port of the second kernel, and the output port of the second kernel is connected to the output exposed to the top level. The first kernel executes when 128 bytes of data (32 complex samples) are collected in a buffer from an external source. This is specified using a `dimensions(first.in[0])={128}` construct. Likewise, the second kernel executes when its input buffer has valid data being produced as the output of the first kernel. Finally, the output of the second kernel is connected to the top-level output port and the `dimensions(second.out[0])={128}` specifies the number of samples of data kernels produced upon termination.

  `buf0` and `buf0d` are ping pong buffers allocated for the `first` kernel input buffer. Similarly, `buf2` and `buf2d` are ping pong buffers allocated for the `second` kernel output buffer. Notice that `buf1` which is the buffer output from the `first` kernel to the `second` kernel is not a ping pong buffer. This is because both the `first` and `second` kernel buffers are in a single AI Engine tile where they execute sequentially. See Runtime Ratio for more information.
6. Set the source file and tile usage for each of the kernels. The source file kernel.cc contains kernel first and kernel second source code. Then the ratio of the function runtime compared to the cycle budget, known as the runtime ratio, and must be between `0` and `1`. The cycle budget is the number of instruction cycles a function can take to do either of the following:Consume data from its input (when dealing with a rate limited input data stream), or Produce a block of data on its output (when dealing with a rate limited output data stream). Changing block sizes can affect the cycle budget. `#include <adf.h> #include "kernels.h" class simpleGraph : public adf::graph { private: adf::kernel first; adf::kernel second; public: adf::input_plio in; adf::output_plio out; simpleGraph(){ first = adf::kernel::create(simple); second = adf::kernel::create(simple); in = adf::input_plio::create(plio_32_bits, "data/input.txt"); out = adf::output_plio::create(plio_32_bits, "data/output.txt"); adf::connect(in.out[0], first.in[0]); adf::connect(first.out[0], second.in[0]); adf::connect(second.out[0], out.in[0]); adf::dimensions(first.in[0]) = {128}; adf::dimensions(first.out[0]) = {128}; adf::dimensions(second.in[0]) = {128}; adf::dimensions(second.out[0]) = {128}; adf::source(first) = "kernels.cc"; adf::source(second) = "kernels.cc"; adf::runtime<ratio>(first) = 0.1; adf::runtime<ratio>(second) = 0.1; } };` Note: See Runtime Ratio for more information.

  - Consume data from its input (when dealing with a rate limited input data stream), or
  - Produce a block of data on its output (when dealing with a rate limited output data stream).

  Changing block sizes can affect the cycle budget.

  ```
  #include <adf.h>
  #include "kernels.h"

  class simpleGraph : public adf::graph {
  private:
    adf::kernel first;
    adf::kernel second;
  public:
    adf::input_plio in;
    adf::output_plio out;

    simpleGraph(){

      first = adf::kernel::create(simple);
      second = adf::kernel::create(simple);

      in = adf::input_plio::create(plio_32_bits, "data/input.txt");
      out = adf::output_plio::create(plio_32_bits, "data/output.txt");

      adf::connect(in.out[0], first.in[0]);
      adf::connect(first.out[0], second.in[0]);
      adf::connect(second.out[0], out.in[0]);
      adf::dimensions(first.in[0]) = {128};
      adf::dimensions(first.out[0]) = {128};
      adf::dimensions(second.in[0]) = {128};
      adf::dimensions(second.out[0]) = {128};

      adf::source(first) = "kernels.cc";
      adf::source(second) = "kernels.cc";

      adf::runtime<ratio>(first) = 0.1;
      adf::runtime<ratio>(second) = 0.1;

    }
  };
  ```
7. Define a top-level application file (for example project.cpp) that contains an instance of your graph class.`#include "project.h" simpleGraph mygraph; int main(void) { adf::return_code ret; mygraph.init(); ret=mygraph.run(<number_of_iterations>); if(ret!=adf::ok){ printf("Run failed\n"); return ret; } ret=mygraph.end(); if(ret!=adf::ok){ printf("End failed\n"); return ret; } return 0; //Must have return statement }`

  ```
  #include "project.h"

  simpleGraph mygraph;

  int main(void) {
    adf::return_code ret;
    mygraph.init();
    ret=mygraph.run(<number_of_iterations>);
    if(ret!=adf::ok){
      printf("Run failed\n");
      return ret;
    }
    ret=mygraph.end();
    if(ret!=adf::ok){
      printf("End failed\n");
      return ret;
    }
    return 0; //Must have return statement
  }
  ```

**Note:** Important:

`mygraph.run()`

AI Engine

`while`

`mygraph.run(<number_of_iterations>)`

**Note:** Important:

AI Engine

ADF APIs have return enumerate type `return_code` to show the API running status.

The `main` program is the driver for the graph. The `main` program loads, executes, and terminates the graph.

**Note:** Write kernel code such that no name clashes occur when two kernels are assigned to the same core.

Related concepts

Runtime Graph Control API
