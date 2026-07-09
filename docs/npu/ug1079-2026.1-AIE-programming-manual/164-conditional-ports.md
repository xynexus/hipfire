---
title: "Conditional Ports"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Conditional-Ports"
toc_id: mOJNT27n~QsncDuVscBrQQ
content_id: q09FoMYEGroWKNoTS8xBpA
---

## Conditional Ports

Certain graphs design can require the instantiation of a port based on a template parameter. For example if you want to use the same templated graph with or without a port that connects to a cascade in or out kernel port. In this case, use the conditional port feature.

```
typename std::conditional<SEL,T1,T2>::type Object
```

Where,

- `T1` is an `input_port`, `output_port`, `kernel`, or user-defined graph.
- `T2` is a dummy type as `std::tuple<>` or `int`.
- `Object` is the name of the object to instantiate.

The syntax to conditionally instantiate an array of ports or sub-graphs is as follows:

```
typename std::array<T3,N>::type Object
```

Where,

- `T3` is an `input_port`, `output_port`, `kernel`, or user-defined graph.
- `N` is the number of items in the array.
- `Object` is the name of the object to instantiate.

Here is an example of conditional port instantiation:

```
#include "adf.h"

using namespace adf;

void k0(input_stream<int32> *, output_stream<int32> *);
void k0_cascin(input_stream<int32> *, output_stream<int32> *, input_cascade<acc48>*);
void k0_cascout(input_stream<int32> *, output_stream<int32> *, output_cascade<acc48>*);
void k0_cascin_cascout(input_stream<int32> *, output_stream<int32> *, input_cascade<acc48>*, output_cascade<acc48>*);

template<bool HAS_CASCADE_IN, bool HAS_CASCADE_OUT>
struct SubGraph: public graph
{
  input_port strmIn;
  input_port strmOut;
  kernel k1;
  typename std::conditional<HAS_CASCADE_IN, input_port, int>::type cascIn;
  typename std::conditional<HAS_CASCADE_OUT, output_port, int>::type cascOut;

  SubGraph()
  {
    if constexpr (HAS_CASCADE_IN && HAS_CASCADE_OUT)
    {
      k1 = adf::kernel::create(k0_cascin_cascout);
    }
    else if constexpr (HAS_CASCADE_IN)
    {
      k1 = adf::kernel::create(k0_cascin);
    }
    else if constexpr (HAS_CASCADE_OUT)
    {
      k1 = adf::kernel::create(k0_cascout);
    }
    else
    {
      k1 = adf::kernel::create(k0);
    }
    adf::connect(strmIn, k1.in[0]);
    adf::connect(k1.out[0], strmOut);
    if constexpr (HAS_CASCADE_IN)
    {
      adf::connect(cascIn, k1.in[1]);
    }
    if constexpr (HAS_CASCADE_OUT)
    {
      adf::connect(k1.out[1], cascOut);
    }
    adf::source(k1) = "kernels.cc";
    adf::runtime<ratio>(k1) = 0.6;
  }
};
```

In this example, the four functions have different ports. One of these functions is instantiated in the graph depending on the template parameters. All functions have one input stream and one output stream, however, the cascade input and output ports are optional. The two template parameters in the `std::conditional` control whether the cascade ports are created at the sub-graph level. They also specify which function to instantiate in the sub-graph, and connect kernel ports to the correct sub-graph port.

The following example shows instantiation of a complete sub-graph which is not dependent on a template parameter:

```
#include "adf.h"


template<int ID>
void f0(input_stream<int32> *, output_stream<int32> *);


struct Sub0: public graph
{
  input_port _in0;
  input_port _out0;
  adf::kernel _k0;
  Sub0()
  {
    _k0 = adf::kernel::create(f0<0>);
    adf::connect(_in0, _k0.in[0]);
    adf::connect(_k0.out[0], _out0);
    adf::runtime<adf::ratio>(_k0) = 0.9;
    adf::source(_k0) = "k0.cpp";
  }
};


struct Sub1: public graph
{
  adf::input_port _in0;
  adf::input_port _out0;
  adf::kernel _k0;
  Sub1()
  {
    _k0 = adf::kernel::create(f0<1>);
    adf::connect(_in0, _k0.in[0]);
    adf::connect(_k0.out[0], _out0);
    adf::runtime<adf::ratio>(_k0) = 0.8;
    adf::source(_k0) = "k0.cpp";
  }
};


template<int ID>
struct MyGraph: public graph
{
  adf::input_plio _plioI;
  adf::output_plio _plioO;

  constexpr static bool hasSub0() {return ID & 0x1;}
  constexpr static bool hasSub1() {return ID & 0x2;}

  typename std::conditional<hasSub0(), Sub0, int >::type _sub0;
  typename std::conditional<hasSub1(), Sub1, int >::type _sub1;

  MyGraph()
  {
    _plioI = adf::input_plio::create("plio_I"+std::to_string(ID), adf::plio_32_bits,   "input"+std::to_string(ID)+".txt");
    _plioO = adf::output_plio::create("plio_O"+std::to_string(ID), adf::plio_64_bits, "output"+std::to_string(ID)+".txt");
    if constexpr (hasSub0() && hasSub1())
    {
      adf::connect(_plioI.out[0], _sub0._in0);
      adf::connect(_sub0._out0, _sub1._in0);
      adf::connect(_sub1._out0, _plioO.in[0]);
    }
    else if constexpr (hasSub0())
    {
      adf::connect(_plioI.out[0], _sub0._in0);
      adf::connect(_sub0._out0, _plioO.in[0]);
    }
    else if constexpr (hasSub1())
    {
      adf::connect(_plioI.out[0], _sub1._in0);
      adf::connect(_sub1._out0, _plioO.in[0]);
    }
  }
};
```

Depending on the template parameter `ID`, the graph `MyGraph` contains either one or both of the sub-graphs `Sub0` and `Sub1`. If `ID=0`, no sub-graph is instantiated, and the compiler errors out. The two lines containing the `std::conditional` conditionally instantiate the two sub-graphs. The connection of the kernels to the I/Os is also driven by the template parameter `ID`.
