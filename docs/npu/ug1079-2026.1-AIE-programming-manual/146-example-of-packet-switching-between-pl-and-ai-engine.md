---
title: "Example of Packet Switching between PL and AI Engine"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Example-of-Packet-Switching-between-PL-and-AI-Engine"
toc_id: Pfa9okbWhZH9SKxGh~b5pQ
content_id: kQgBHTQeULKrTSh9kpfSvw
---

### Example of Packet Switching between PL and AI Engine

The PL to AI Engine interface allows multiple low bandwidth PL sources to use packet switching to distribute data to different destinations in the AI Engine. The same interface can also merge multiple AI Engine packet streams and transmit them to the PL.

To view the packet header format, see [Packet Processing](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1079-ai-engine-kernel-coding&resourceid=met1620730486038.html) in the AI Engine Kernel and Graph Programming Guide ([UG1079](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1079-ai-engine-kernel-coding)). If a packet originates from the PL, it is the PL's responsibility to generate the correct packet header and data with the TLAST field set to true for the last sample of the packet. Conventionally, packets created in the programmable logic can have a row and column initialization of -1,-1, which indicates that the packet is not possible from inside AI Engine.

If the PL receives packets from the AI Engine, it needs to decode the header to determine the packet's origin and treat it accordingly. For example, based on the decoded packet ID, the PL dispatches packet data to the correct destinations.

AI Engine

```
class PLPacketGraph: public adf::graph {
 private:
    adf:: kernel core[4];
    adf:: pktsplit<4> sp;
    adf:: pktmerge<4> mg;
 public:
    adf::input_plio in;
    adf::output_plio out;
    mygraph() {
      core[0] = adf::kernel::create(aie_pktstream_core1);
      core[1] = adf::kernel::create(aie_pktstream_core2);
      core[2] = adf::kernel::create(aie_pktstream_core3);
      core[3] = adf::kernel::create(aie_pktstream_core4);
      adf::source(core[0]) = "aie_pktstream_core1.cpp";
      adf::source(core[1]) = "aie_pktstream_core2.cpp";
      adf::source(core[2]) = "aie_pktstream_core3.cpp";
      adf::source(core[3]) = "aie_pktstream_core4.cpp";

      in=input_plio::create("Datain0", plio_32_bits, "data/input.txt");
      out=output_plio::create("Dataout0", plio_32_bits, "data/output.txt");

      sp = adf::pktsplit<4>::create();
      mg = adf::pktmerge<4>::create();
      for(int i=0;i<4;i++){
        adf::runtime<ratio>(core[i]) = 0.9;
        adf::connect(sp.out[i], core[i].in[0]);
        adf::connect(core[i].out[0], mg.in[i]);
      }
      adf::connect(in.out[0], sp.in[0]);
      adf::connect(mg.out[0], out.in[0]);
    }
};
```

An AI Engine kernel code example is as follows.

```
const uint32 pktType=0;
void aie_pktstream_core1(input_pktstream *in,output_pktstream *out){
  readincr(in);//read header and discard because only the correct packet arrives
  uint32 ID=getPacketid(out,0);//for output pktstream, index always =0
  writeHeader(out,pktType,ID); //Generate header for output

  bool tlast;
  for(int i=0;i<8;i++){
    int32 tmp=readincr(in,tlast);
    tmp+=1;
    writeincr(out,tmp,i==7);//TLAST=1 for last word
  }
}
```

The PL kernel doesn't have a helper function, such as `getPacketid()`, to extract a specific destination's packet ID. To obtain the correct packet ID, the compiled report files, `Work/temp/packet_ids_c.h` and `Work/temp/packet_ids_v.h`, can be used in C/C++ or Verilog source files.

Work/temp/packet_ids_c.h

```
#define Datain0_0 0
#define Datain0_1 1
#define Datain0_2 2
#define Datain0_3 3
#define Dataout0_0 0
#define Dataout0_1 1
#define Dataout0_2 2
#define Dataout0_3 3
```

The macro Datain0_0 connects the PL to the 0th index of the `pktsplit` output, while the macro Datain0_1 connects it to the first index. It's worth noting that the macro name remains the same across different compilations unless there is a change in the graph structure. However, the macro value (such as the `packet ID`) can differ among compilations.

Based on these macro names, you can write the following example HLS helper function for the PL kernel as shown:

```
#include "packet_ids_c.h"

static const unsigned int pktType=0;
static const int PACKET_NUM=4; //How many kernels do packet switching
static const int PACKET_LEN=8; //Length for a packet

static const unsigned int packet_ids[PACKET_NUM]={Datain0_0, Datain0_1, Datain0_2, Datain0_3}; //macro values are generated in packet_ids_c.h

ap_uint<32> generateHeader(unsigned int pktType, unsigned int ID){
#pragma HLS inline
	ap_uint<32> header=0;
	header(4,0)=ID;
	header(11,5)=0;
	header(14,12)=pktType;
	header[15]=0;
	header(20,16)=-1;//source row
	header(27,21)=-1;//source column
	header(30,28)=0;
	header[31]=header(30,0).xor_reduce()?(ap_uint<1>)0:(ap_uint<1>)1;
	return header;
}
void hls_packet_sender(......){
	for(unsigned int iter=0;iter<num;iter++){
		for(int i=0;i<PACKET_NUM;i++){//Iterate on PL kernels that do packet switching
			unsigned int ID=packet_ids[i]; //get packet ID from AIE compilation
			ap_uint<32> header=generateHeader(pktType,ID); //packet header
			ap_axiu<32,0,0,0> tmp;
			tmp.data=header;
			tmp.keep=-1;
			tmp.last=0;
			out.write(tmp);//write packet header
			for(int j=0;j<PACKET_LEN;j++){ //generate packet data
......
```

AI Engine

[Vitis Tutorials: AI Engine Development: Packet Switching](https://docs.amd.com/r/en-US/Vitis-Tutorials-AI-Engine-Development/Packet-Switching)
