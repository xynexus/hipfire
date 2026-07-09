---
title: "Graphs with Libraries Components"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Graphs-with-Libraries-Components"
toc_id: 6PqGCM4Q05Yd1reDp3~_3A
content_id: Hx4ymbFPYMLbD63Gq7zs4g
---

### Graphs with Libraries Components

The Vitis libraries targeting AI Engine are currently split into two levels:

- **Level 1:** Basic kernels
- **Level 2:** Graphs that use multiple Level 1 kernels with parameters specified through templates. The following examples contain PL data movers and host code that uses Level 2 functions in hardware.

This example uses the DSP Library (`DSPLib`) and includes two filters. The first filter has five sections to meet the system’s overall throughput requirements. The system settings are as follows:

```
#include <adf.h>
#include "fir_sr_sym_graph.hpp"
#include "fir_interpolate_hb_graph.hpp"

// Filter parameters
#define DATA_TYPE cint16
#define COEFF_TYPE int16

#define FIR_LEN_CHAN 151
#define SHIFT_CHAN 15
#define ROUND_MODE_CHAN 0
#define AIES_CHAN 5

#define FIR_LEN_HB 43
#define SHIFT_HB 15
#define ROUND_MODE_HB 0

#define WINDOW_SIZE 1024

// Simulation parameters
#define NUM_ITER 8
```

The graph that instantiates the filter sub-graph is shown below:

```
namespace dsplib = xf::dsp::aie;

class FirGraph: public adf::graph
{
private:
  // Channel Filter coefficients
  std::vector<int16> chan_taps = std::vector<int16>{
  -17, -65, -35, 34, -13, -6, 18, -22,
  18, -8, -5, 18, -26, 26, -16, -1,
  21, -36, 40, -31, 8, 21, -46, 59,
  -53, 26, 13, -54, 81, -83, 56, -6,
  -54, 102, -122, 101, -43, -38, 116, -164,
  161, -102, 1, 114, -204, 235, -190, 74,
  83, -231, 319, -310, 193, 5, -229, 406,
  -468, 380, -147, -174, 487, -684, 680, -437,
  -10, 553, -1030, 1262, -1103, 474, 596, -1977,
  3451, -4759, 5660, 26983};

  // HalfBand Filter coefficients
  std::vector<int16> hb_taps = std::vector<int16>{
  23, -63, 143, -281, 503, -845, 1364, -2173,
  3557, -6568, 20729, 32767};

using channel = xf::dsp::aie::fir::sr_sym::
fir_sr_sym_graph<DATA_TYPE, COEFF_TYPE, FIR_LEN_CHAN, SHIFT_CHAN, ROUND_MODE_CHAN, WINDOW_SIZE, AIES_CHAN>;

using halfband = xf::dsp::aie::fir::interpolate_hb::
fir_interpolate_hb_graph<DATA_TYPE, COEFF_TYPE, FIR_LEN_HB, SHIFT_HB, ROUND_MODE_HB, WINDOW_SIZE>;

public:
  adf::port<input> in;
  adf::port<output> out;

  // Constructor - with FIR graph classes initialization
  FirGraph(){

    channel chan_FIR(chan_taps);
    halfband hb_FIR(hb_taps);

    // Margin gets automatically added within the FIR graph class.
    // Margin equals to FIR length rounded up to nearest multiple of 32 Bytes.
    adf::connect(in, chan_FIR.in[0]);
    adf::connect(chan_FIR.out[0], hb_FIR.in[0]);
    adf::connect(hb_FIR.out[0], out);
  };
};
```

As usual, another graph defines the connections to the external world. This setup gives you more flexibility when reusing or modifying this graph:

```
class TopGraph : public adf::graph
{
public:
  adf::input_plio in;
  adf::output_plio out;

  FirGraph F;

  TopGraph()
  {
    in = adf::input_plio::create("128 bits read in",
                      adf::plio_128_bits,"data/input_128b.txt", 250);
    out = adf::output_plio::create("128 bits read out",
                      adf::plio_128_bits,"data/output_128b.txt", 250);

    adf::connect (in.out[0],F.in);
    adf::connect (F.out, out.in[0]);
  };
};
```

And, finally the test bench used to launch the simulation is shown below:

```
TopGraph LibBasedGraph;

int main(void) {
  LibBasedGraph.init() ;
  LibBasedGraph.run(NUM_ITER) ;
  LibBasedGraph.end() ;
  return 0 ;
}
```

The graph view in the Vitis IDE is shown below:

![rzj1677063770211.png](../assets/186-01-rzj1677063770211-png-4f36fc0dd5ae.png)

*Figure 1. DPSLib Graph View*

As shown, the `chan_FIR` is built on five kernels (`AIES_CHAN = 5`). These are grouped in a single graph. The `hb_FIR` filter is based on a single kernel, defined within its own sub-graph.

**Note:** Important:
