---
title: "Design Flow Using RTL Programmable Logic"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Design-Flow-Using-RTL-Programmable-Logic"
toc_id: oArHb1HMEhXek4wb5U9LxA
content_id: rHbuZ0BWil_MiT~DPKsFPg
---

## Design Flow Using RTL Programmable Logic

The ADF graph does not support RTL blocks. Communication between the RTL blocks and the ADF graph requires PLIO interfaces. In the following example, `interpolator` and `classify` are AI Engine kernels. The `interpolator` AI Engine kernel streams data to a PL RTL block, which, in turn, streams data back to the AI Engine `classify` kernel.

```
class clipped : public graph {
  private:
    kernel interpolator;
    kernel classify;

  public:
    input_plio in;
    output_plio out;
    output_plio clip_in;
    input_plio clip_out;

    clipped() {
      in  = input_plio::create("DataIn1", plio_32_bits,"data/input.txt");
      out = output_plio::create("DataOut1", plio_32_bits,"data/output.txt");
      clip_out = input_plio::create("clip_out",plio_32_bits,"data/input1.txt");
      clip_in  = output_plio::create("clip_in", plio_32_bits,"data/output1.txt");

      interpolator = kernel::create(fir_27t_sym_hb_2i);
      classify     = kernel::create(classifier);

      connect(in.out[0], interpolator.in[0]);
      connect(interpolator.out[0], clip_in.in[0]);
      connect(clip_out.out[0], classify.in[0]);
      connect(classify.out[0], out.in[0]);

      std::vector<std::string> myheaders;
      myheaders.push_back("include.h");

      adf::headers(interpolator) = myheaders;
      adf::headers(classify) = myheaders;

      source(interpolator) = "kernels/interpolators/hb27_2i.cc";
      source(classify)    = "kernels/classifiers/classify.cc";

      runtime<ratio>(interpolator) = 0.8;
      runtime<ratio>(classify) = 0.8;
    };
};
```

`clip_in` and `clip_out` are ports to and from the `polar_clip` PL RTL kernel which is connected to the AI Engine kernels in the graph. For example, the `clip_in` port is the output of the `interpolator` AI Engine kernel that is connected to the input of the `polar_clip` RTL kernel. The `clip_out` port is the input of the `classify` AI Engine kernel and the output of the `polar_clip` RTL kernel.

#### RTL Blocks and AI Engine

The following example shows application code.

```
#include "graph.h"

clipped clipgraph;

#if defined(__AIESIM__ ) || defined(__X86SIM__)
int main(int argc, char ** argv) {
    clipgraph.init();
    clipgraph.run();
    clipgraph.end();
    return 0;
}
#endif
```

To make the `aiesimulator` work, you must create input test bench files related to the RTL kernel. data/output_interp.txt is the test bench input to the RTL kernel. The `aiesimulator` generates the output file from the `interpolator` AI Engine kernel. The data/input_classify.txt file contains data from the `polar_clip` kernel which is input to the AI Engine `classify` kernel.

**Note:** `polar_clip`

#### RTL Blocks in Hardware Emulation and Hardware Flows

Hardware emulation and hardware flows fully support RTL kernels. You need to add the RTL kernel as an `nk` option and link the interfaces with the `sc` option, as shown in the following code. If necessary, adjust any clock frequency using `freqHz`. The following is an example of a Vitis configuration file.

```
[connectivity]
nk=mm2s:1:mm2s
nk=s2mm:1:s2mm
nk=polar_clip:1:polar_clip
sc=mm2s.s:ai_engine_0.DataIn1
sc=ai_engine_0.clip_in:polar_clip.in_sample
sc=polar_clip.out_sample:ai_engine_0.clip_out
sc=ai_engine_0.DataOut1:s2mm.s
[clock]
freqHz=100000000:polar_clip.ap_clk
```

For more information on RTL kernels and the Vitis flow, see [RTL Kernel Development](https://docs.amd.com/access/sources/dita/topic?isLatest=true&url=ug1701-vitis-accelerated-embedded&resourceid=ppy1726709921392.html&ft:locale=en-US) in the Embedded Design Development Using Vitis ([UG1701](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1701-vitis-accelerated-embedded)).

#### PL Kernels Inside AI Engine Graph

After system linking with AI Engine graph and PL Kernels, you can view their connections in the system diagram in the AI Engine graph view. Open the linker summary in the Vitis IDE with the following command:

```
vitis -a <USER_NAME>.xsa.link_summary
```

![gfa1695064501018.png](../assets/172-01-gfa1695064501018-png-3afce6c2ac1e.png)

*Figure 1. PL Kernels Inside AI Engine Graph*
