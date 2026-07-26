---
title: "Global Graph-Scoped Tables"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Global-Graph-Scoped-Tables"
toc_id: JV1YUivNDyuYRlZC03kulA
content_id: B7x4vElhG20pUbyqXIKcuQ
---

### Global Graph-Scoped Tables

The previous example uses an eight entry look-up table accessed as a global variable. However, many other algorithms require much larger look-up tables. Because AI Engine local memory is at a premium, it is more efficient for the AI Engine compiler to manage the look-up table explicitly for specific kernels rather than leave a large amount of stack or heap space on every processor. Do not declare these tables as static in the kernel header file.

```
#ifndef USER_PARAMETER_H
#define USER_PARAMETER_H

#include <adf.h>

int16 lutarray[8] = {1,2,3,4,5,6,0,0} ;

#endif
```

The kernel source continues to include the header file and use the table as before. But, now you must declare this table as `extern` in the graph class header and use the `parameter::array(…)` function to create a parameter object explicitly in the graph. You also need to attach this parameter object to the kernel as shown in the following code:

```
#include <adf.h>
extern int16 lutarray[8];
class simple_lut_graph : public adf::graph {
public:
  adf::kernel k;
  adf::parameter p;

  simple_lut_graph() {
    k = adf::kernel::create(simple);
    p = adf::parameter::array(lutarray);
    adf::connect(p,k);
    ...
  }
}
```

Including this explicit specification of the look-up table in the graph description ensures that the compiler is aware of the requirement to reserve a suitably sized piece of memory for the look-up table when it allocates memory for kernel input and output buffers.

```
const int size=1024;
extern int16 lnr_lutab[size*2*2];
extern int16 lnr_lutcd[size*2*2];

class adaptive_graph : public adf::graph
{
  public:
    adf::input_plio in;
    adf::output_plio out;
    adf::kernel k;

  adaptive_graph(){
    k = adf::kernel::create(linear_approx);
    adf::source(k) = "linear_approx.cc";
    in=adf::input_plio::create("Datain10", adf::plio_64_bits, "data/input1.txt");
    out=adf::output_plio::create("Dataout1", adf::plio_64_bits, "data/output1.txt");
    adf::runtime<ratio>(k) = 0.8;

    auto buf_lnr_ab = adf::parameter::array(lnr_lutab);
    auto buf_lnr_cd = adf::parameter::array(lnr_lutcd);
    adf::connect(buf_lnr_ab,k);
    adf::connect(buf_lnr_cd,k);
    adf::location<adf::parameter>(buf_lnr_ab)={ adf::address(8,1,0x8000) };//optional
    adf::location<adf::parameter>(buf_lnr_cd)={ adf::address(8,1,0xC000) };//optional
    adf::location<adf::buffer>(k.out[0])={ adf::address(8,0,0x8000),adf::address(8,0,0xC000) };//optional
    adf::location<adf::stack>(k)={ adf::address(8,1,0x4000) };//optional
    adf::location<adf::buffer>(k.in[0])={ adf::address(8,0,0x0000), adf::address(8,0,0x4000) };//optional

    adf::connect(in.out[0], k.in[0]);
    adf::connect(k.out[0], out.in[0]);
    adf::dimensions(k.in[0])={1024};//elements
    adf::dimensions(k.out[0])={1024};//elements
  }
};
```
