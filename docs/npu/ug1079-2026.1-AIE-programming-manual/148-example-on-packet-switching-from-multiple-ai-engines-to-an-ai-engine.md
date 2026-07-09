---
title: "Example on Packet Switching from Multiple AI Engines to an AI Engine"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Example-on-Packet-Switching-from-Multiple-AI-Engines-to-an-AI-Engine"
toc_id: RkEq9D0dxiU_CkK6dCHVeg
content_id: M3MK1otLiHIsuqXF6GlFqQ
---

### Example on Packet Switching from Multiple AI Engines to an AI Engine

AI Engine kernels can merge packets from different sources before sending the data to destinations such as PL kernels. This section covers the AI Engine kernel code that manages packets and their headers. The following graph is an example:

```
#include <adf.h>
#include "kernels.h"

class mergegraph: public adf::graph {
private:
  adf:: kernel core[4],core_m;

  adf:: pktmerge<4> mg;
public:
  adf::input_plio  in[4];
  adf::output_plio  out;
  mergegraph() {
    core[0] = adf::kernel::create(aie_core1);
    core[1] = adf::kernel::create(aie_core2);
    core[2] = adf::kernel::create(aie_core3);
    core[3] = adf::kernel::create(aie_core4);
    core_m = adf::kernel::create(aie_combine_core);
    adf::source(core[0]) = "aie_core1.cpp";
    adf::source(core[1]) = "aie_core2.cpp";
    adf::source(core[2]) = "aie_core3.cpp";
    adf::source(core[3]) = "aie_core4.cpp";
    adf::source(core_m) = "aie_combine_core.cpp";
    adf::runtime<ratio>(core_m) = 0.9;
    adf::repetition_count(core_m)=4;

    out=output_plio::create("Dataout0", plio_32_bits,  "data/output.txt");

    mg = adf::pktmerge<4>::create();
    for(int i=0;i<4;i++){
      in[i]=adf::input_plio::create("Datain"+std::to_string(i), plio_32_bits,  "data/input"+std::to_string(i)+".txt");
      adf::runtime<ratio>(core[i]) = 0.9;
      adf::repetition_count(core[i])=1;
      adf::connect (in[i].out[0], core[i].in[0]);
    adf::connect<adf::pktstream > (core[i].out[0], mg.in[i]);
    }

    adf::connect<adf::pktstream> (mg.out[0], core_m.in[0]);
    adf::connect<> (core_m.out[0], out.in[0]);
  }
};
```

The graph view of the above graph is as follows:

![iet1726781250569.png](../assets/148-01-iet1726781250569-png-1df43249a3ae.png)

*Figure 1. Graph View of Multiple AI Engines to AI Engine*

AI Engine

```
#include <aie_api/aie.hpp>

const uint32 pktType=0;
const int PACKET_NUM=4;
static uint32 ID_TABLE[PACKET_NUM];
static int iteration=0;
void aie_combine_core(input_pktstream *in,output_stream<int32> *out){
    if(iteration==0){
		for(int i=0;i<PACKET_NUM;i++){
			uint32 packet_id=getPacketid(in,i);
			ID_TABLE[packet_id%PACKET_NUM]=i;
			printf("merge input index=%d, compiler assigned packet ID=%d\n",i,packet_id);
		}
	}
	iteration++;

    int32 header=readincr(in);
	uint32 ID=header & 0x1f;//packet id from header
    uint32 index=ID_TABLE[ID];//get merge input index, which is fixed by graph code.

	for(int i=0;i<8;i++){
		int32 tmp=readincr(in);
        if(index==0){//operate based on the index of the merge input
            ...
        }else if(index==1){
            ...
        }else if(index==2){
            ...
        }else if(index==3){
            ...
        }
		writeincr(out,tmp,i==7);//TLAST=1 for last word
	}
}
```

The connection of the `pktmerge` is predetermined by the graph code. However, the packet ID for each `pktmerge` input can change with subsequent compilations. Therefore, the kernel code constructs a table called `ID_TABLE` that maps the packet ID to an index. This index can then be used to perform operations and is fixed, provided the graph connection remains unchanged.

Following is an example code of the `producer` kernel:

```
#include <aie_api/aie.hpp>

const uint32 pktType=0;

void aie_core2(input_stream<int32> *in,output_pktstream *out){
	uint32 ID=getPacketid(out,0); //index always =0 in the kernel code
	writeHeader(out,pktType,ID); //Generate header for output

	for(int i=0;i<8;i++){
		int32 tmp=readincr(in);
		tmp+=2;
		writeincr(out,tmp,i==7);//TLAST=1 for last word
	}
}
```

For all `producer` kernels, the index parameter in the `getPacketid` API for `output_pktstream`is always set to 0, as shown in the above code snippet
