---
title: "Packet Switching Graph Constructs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Switching-Graph-Constructs"
toc_id: n9ZNVPzBaWvho4r_jqt9zg
content_id: Cno5V4U~ylXdBa6RdOU2cg
---

### Packet Switching Graph Constructs

Packet-switched streams are essentially multiplexed data streams that carry different types of data at different times. Packet-switched streams do not provide deterministic latency due to the potential for resource contention with other packet-switched streams. The multiplexed data flows in units of packets with a 32-bit packet header and a variable number of payload words. A header word must be sent before the payload data. The last word of the packet requires the TLAST signal. Two new data types called `input_pktstream` and `output_pktstream` are introduced to represent the multiplexed data streams as input to or output from a kernel, respectively.

See Packet Stream Operations for more details on the packet headers and data types.

To explicitly control the multiplexing and de-multiplexing of packets, two new templated node classes: `pktsplit<n>` and `pktmerge<n>`, are added to the ADF graph library. A node instance of class `pktmerge<n>` is a n:1 multiplexer of n packet streams producing a single packet stream. A node instance of class `pktsplit<n>` is a 1:n de-multiplexer of a packet stream producing n different packet streams. The maximum number of allowable packet streams is 32 on a single physical channel (n ≤ 32).

A kernel can receive packets of data either as buffers of data or as `input_pktstream`. And a kernel can send packets of data either as buffers of data or as `output_pktstream`.

```
connect (<SOURCE>.out[0], <DEST>.in[0]);
```

When a kernel receives packets of data as a buffer of data, the header and TLAST drop prior to the kernel receiving the buffer. If the kernel writes an output buffer of data, packet header and TLAST are automatically inserted, when the buffer is transferred by DMA to a packet stream.

However, if the kernel receives `input_pktstream` of data, the kernel needs to process the packet header and TLAST, in addition to the packet data. Similarly, if the kernel sends an `output_pktstream`, it needs to insert the packet header and TLAST. It also needs to add the packet data into the output stream.

These concepts are illustrated in the following example.

```
class ExplicitPacketSwitching: public adf::graph {
 private:
    adf:: kernel core[4];
    adf:: pktsplit<4> sp;
    adf:: pktmerge<4> mg;
 public:
    adf::input_plio in;
    adf::output_plio out;
    mygraph() {
      core[0] = adf::kernel::create(aie_core1);
      adf::source(core[0]) = "aie_core1.cpp";

      ......
      sp = adf::pktsplit<4>::create();
      mg = adf::pktmerge<4>::create();
      for(int i=0;i<4;i++){
        adf::connect(sp.out[i], core[i].in[0]);
        adf::connect(core[i].out[0], mg.in[i]);
      }
      adf::connect(in.out[0], sp.in[0]);
      adf::connect(mg.out[0], out.in[0]);
    }
};
```

The graph has one input PLIO port and one output PLIO port. The input packet stream from the PL is split four ways and input to four different AI Engine kernels. The output streams from the four AI Engine kernels are merged into one packet stream which is output to the PL. The following figure shows the Vitis IDE Graph view of the code.

![vct1678739144375.png](../assets/145-01-vct1678739144375-png-22bfee42fa51.png)

*Figure 1. Graph View*

When the AI Engine kernel uses buffer port input, the packet stream to the buffer decodes automatically and packet data transfer to the buffer via DMA. When the AI Engine kernel uses buffer port output and the buffer is connected to packet stream, the packet header is inserted automatically. The last sample of the buffer data is enabled with TLAST to denote the completion of the packet.

To read data from a packet stream in the AI Engine, the `input_pktstream` interface is used. However, this data is read as an integer value by default. If other data types are to be read, the data can first be read as an integer value and then cast to the desired data type.

Following is an example kernel code that accepts and transfers floating-point data type:

```
const uint32 pktType=0;
void aie_core1_float(input_pktstream *in,output_pktstream *out){
  readincr(in);//read header and discard
  uint32 ID=getPacketid(out,0);//for output pktstream
  writeHeader(out,pktType,ID); //Generate header for output

  bool tlast;
  for(int i=0;i<4;i++){
    int32 tmp=readincr(in,tlast);//read data as integer type
    float tmp_f=reinterpret_cast<float&>(tmp);//Reinterpret memory as float
    tmp_f+=1.0f;
    writeincr(out,tmp_f,i==3);//TLAST=1 for last word
  }
}
```

The input data must also be in integer format for simulation. For example, if float values `0.0f`, `1.0f`, `2.0f` and `3.0f` are sent to the kernel, they are converted into integer values in the input data file:

```
0
1065353216
1073741824
1077936128
```

After the kernel execution, the float output values are `1.0f`, `2.0f`, `3.0f` and `4.0f`. In a simulation output data file, the output values are in integer format, as follows:

```
1065353216
1073741824
1077936128
1082130432
```

Related reference

Adaptive Data Flow Graph Specification Reference
