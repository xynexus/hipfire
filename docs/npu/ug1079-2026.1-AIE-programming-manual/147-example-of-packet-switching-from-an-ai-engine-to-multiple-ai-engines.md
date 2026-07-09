---
title: "Example of Packet Switching from an AI Engine to Multiple AI Engines"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Example-of-Packet-Switching-from-an-AI-Engine-to-Multiple-AI-Engines"
toc_id: gtyFk12nYSGnmf1DR7Viuw
content_id: 6cyXAB0_CPFV0EdHjFbGZQ
---

### Example of Packet Switching from an AI Engine to Multiple AI Engines

If the performance allows, AI Engine kernels can be used to dispatch packets to multiple destinations. This section specifically addresses the AI Engine kernel code necessary to manage the packets, particularly the packet header. The following graph illustrates this process:

```
#include <adf.h>
#include "kernels.h"

class mygraph: public adf::graph {
private:
  adf:: kernel core[4],core_m;

  adf:: pktsplit<4> sp;
public:
  adf::input_plio  in;
  adf::output_plio  out[4];
  mygraph() {
    core[0] = adf::kernel::create(aie_core1);
    core[1] = adf::kernel::create(aie_core2);
    core[2] = adf::kernel::create(aie_core3);
    core[3] = adf::kernel::create(aie_core4);
    core_m = adf::kernel::create(aie_master_core);
    adf::source(core[0]) = "aie_core1.cpp";
    adf::source(core[1]) = "aie_core2.cpp";
    adf::source(core[2]) = "aie_core3.cpp";
    adf::source(core[3]) = "aie_core4.cpp";
    adf::source(core_m) = "aie_master_core.cpp";
    adf::runtime<ratio>(core_m) = 0.9;
    adf::repetition_count(core_m)=4;
    adf::location<adf::kernel>(core_m)=adf::tile(0,0);

    in=adf::input_plio::create("Datain0", adf::plio_32_bits,  "data/input.txt");
    sp = adf::pktsplit<4>::create();
    for(int i=0;i<4;i++){
      out[i]=adf::output_plio::create("Dataout"+std::to_string(i), plio_32_bits,  "data/output"+std::to_string(i)+".txt");
      adf::runtime<ratio>(core[i]) = 0.9;
      adf::repetition_count(core[i])=1;
      adf::connect<adf::pktstream > (sp.out[i], core[i].in[0]);
      adf::connect<> (core[i].out[0], out[i].in[0]);
    }

    adf::connect<> (in.out[0], core_m.in[0]);
    adf::connect<adf::pktstream> (core_m.out[0], sp.in[0]);
  }
};
```

The graph view of the above graph is as follows:

![nvw1726782038467.png](../assets/147-01-nvw1726782038467-png-892efdb0f934.png)

*Figure 1. Graph View of AI Engine to Multiple AI Engines*

Below is the code for the AI Engine kernel responsible for sending packets:

```
#include <aie_api/aie.hpp>
const uint32 pktType=0;

void aie_master_core(input_stream<int32> *in,output_pktstream *out){
	int32 user_id=readincr(in); //this is somehow specified by user
	uint32 ID=getPacketid(out,user_id);//get packet ID from user speficified destination
	writeHeader(out,pktType,ID); //Generate header for output
	for(int i=0;i<8;i++){
		int32 tmp=readincr(in);
		writeincr(out,tmp,i==7);//TLAST=1 for last word
	}
}
```

The `pktsplit` connection is established at compile time. You cannot modify the connection. You can use the `getPacketid()` API to extract the packet ID from a specific index of the `pktsplit` output. This ID is generates automatically during compilation and can be used to transfer the packet from a source to a target connection in the graph.

**Note:** Only packets matching the extracted packet ID reach the intended receiver kernel. When received, the kernel can access or discard the packet's header as necessary.

Following is an example code of the receiver kernel:

```
#include <aie_api/aie.hpp>

void aie_core1(input_pktstream *in,output_stream<int32> *out){
	readincr(in);//read header and discard

	bool tlast;
	for(int i=0;i<8;i++){
		int32 tmp=readincr(in,tlast);
		tmp+=1;
		writeincr(out,tmp,i==7);//TLAST=1 for last word
	}
}
```
