---
title: "Buffer Port Multicasting"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Port-Multicasting"
toc_id: z5ze7vho_dKcdGpsjQbyiQ
content_id: MARvZYhO_wZazFYBAD1ISQ
---

### Buffer Port Multicasting

The AI Engine compiler does not limit the buffer ports to a one-to-one connection. Under certain conditions, multiple kernels can share the same output buffer to perform various tasks. You can connect a producer to as many consumers as needed. The AI Engine compiler automatically infers a MM2S DMA to read the output buffer. The compiler also infers as many S2MM DMAs as there are consumers to write to their respective input buffers.

```
private:
adf::kernel mk;
adf::kernel tk0,tk1,tk2,tk3;
...
connect net0 ( mk.out[0] , tk0.in[0] );
connect net1 ( mk.out[0] , tk1.in[0] );
connect net2 ( mk.out[0] , tk2.in[0] );
connect net3 ( mk.out[0] , tk3.in[0] );
...
dimensions(tk0.in[0]) = {128};
dimensions(tk1.in[0]) = {128};
dimensions(tk2.in[0]) = {128};
dimensions(tk3.in[0]) = {128};
```

Kernel function prototypes:

```
tk0(input_buffer<int32, adf::extents<adf::inherited_extent>> & in0,
      output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out0);
tk1(input_buffer<int32, adf::extents<adf::inherited_extent>,
                adf::margin<32>> & in0,
        output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out0);
tk2(input_buffer<int32, adf::extents<adf::inherited_extent>,
                adf::margin<64>> & in0,
        output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out0);
tk3(input_buffer<int32, adf::extents<adf::inherited_extent>> & in0,
        output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out0);
```

`tk0`

`tk1`

`tk2`

`tk3`

`mk`

AXI4-Stream

AI Engine

![qaq1691072135869.png](../assets/090-01-qaq1691072135869-png-ce853c5f3208.png)

*Figure 1. One Kernel Serving Four Kernels*

The code connects the same kernel output to four different kernel inputs. The AI Engine compiler adds DMAs between kernels to copy the `buf5(d)` buffer content to other buffers using the AXI4-Stream interconnect network.
