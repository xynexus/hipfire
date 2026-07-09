---
title: "Array of Graph Objects"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Array-of-Graph-Objects"
toc_id: ljVN7yyle6ic~D5ms0ofGg
content_id: 5e15rH9OQ47W_R~wvD7x2A
---

## Array of Graph Objects

A graph can contain multiple ports of the same type depending on some template parameter. This case is handled efficiently by the standard C vector notation, for example, `input_port in[5];`.

If you want this set of ports to be conditionally instantiated, the compiler will reject this notation. In that case, you must use C++ `std::array` notation:

```
std::array<T,N> Object;
```

Where,

- `T` is the type of the object `input_port/output_port`, `input_plio/output_plio`, `input_gmio/output_gmio`, sub-graph, kernel, and so on.
- `N` is the number of elements of the array.
- `Object` is the name of the array to instantiate.

The following example shows how to declare various objects in arrays:

```
#include "adf.h"
#include <stdlib.h>


void f0(input_stream<int32> *, output_stream<int32> *);

constexpr int getNum(int id) {return id == 1? 2: 3;}

template<int ID>
struct Sub0: public graph
{
  std::array<adf::input_port,getNum(ID)> _ins;
  std::array<adf::output_port,getNum(ID)> _outs;
  std::array<adf::kernel,getNum(ID)> _ks;
  Sub0()
  {
    for (auto &k: _ks)
    {
      k = adf::kernel::create(f0);
      adf::runtime<adf::ratio>(k) = 0.9;
      adf::source(k) = "k0.cpp";
    }
    for (int ind = 0; ind < _ins.size(); ++ind)
    {
      adf::connect(_ins[ind], _ks[ind].in[0]);
      adf::connect(_ks[ind].out[0], _outs[ind]);
    }
  }
};


template<int ID>
struct MyGraph: public graph
{
  std::array<input_plio , getNum(ID)> _plioIs;
  std::array<output_plio, getNum(ID)> _plioOs;
  Sub0<ID> _sub;

  MyGraph()
  {
    for (int ind = 0; ind < getNum(ID); ++ind)
    {
      char iName[40];
      char oName[40];
      sprintf(iName, "data/i_%d_%d.txt", ID, ind);
      sprintf(oName, "data/o_%d_%d.txt", ID, ind);

      _plioIs[ind] = adf::input_plio::create(adf::plio_32_bits, iName);
      _plioOs[ind] = adf::output_plio::create(adf::plio_32_bits, oName);

      adf::connect(_plioIs[ind].out[0], _sub._ins[ind]);
      adf::connect(_sub._outs[ind], _plioOs[ind].in[0]);
    }
  }
};
```
