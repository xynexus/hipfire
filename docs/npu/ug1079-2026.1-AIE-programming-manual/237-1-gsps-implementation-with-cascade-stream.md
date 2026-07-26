---
title: "1 Gsps Implementation with Cascade Stream"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/1-Gsps-Implementation-with-Cascade-Stream"
toc_id: txL730VD12p3xzpB3jsqRg
content_id: rcSIzPNSObUtMlyJHOZuEA
---

### 1 Gsps Implementation with Cascade Stream

The AI Engine vector unit supports 8 MACs per cycle for cint16 multiply-accumulate cint16 types. If a four lane implementation of mul4/mac4 intrinsics is adopted, then there are two complex operations on each lane.

![ljj1606874814465.png](../assets/237-01-ljj1606874814465-png-b057ea5a7ee0.png)

Computing four outputs requires 16 mac4() because each output requires 32 complex MACs. This means, computing four outputs requires 16 cycles using an AI Engine. So the sample rate of an AI Engine (assuming it runs at 1 GHz) is as follows.

This calculates the compute bound of an AI Engine. However, you still need to consider the memory bound to see if that sample rate can be met. Assume that one stream input and one stream output are used for data transfer and coefficients are stored in the AI Engine internal memory. The stream interface of an AI Engine supports 32 bits per cycle. It is capable of transferring one sample of data every cycle.

Thus, the sample rate from the data transferring view is as follows.

This is larger than the compute bound, which is 250 Msps. Therefore the AI Engine implementation operates at 250 Msps.

![abo1606888463225.png](../assets/237-02-abo1606888463225-png-aa1b4d6fea21.png)

*Figure 1. One AI Engine FIR Filter Realization*

Based on the calculations, it is possible to achieve 1 Gsps via a stream input and output stream interface. If you split the MAC operations of a single kernel implementation into four kernels, 4*250Msps = 1 Gsps, compute throughput can be achieved. Those four kernels are connected through cascade streaming. Therefore, the AI Engine compute bound matches AI Engine interface throughput.

![enx1606888501912.png](../assets/237-03-enx1606888501912-png-be726a5a38b6.png)

*Figure 2. 1 Gsps Implementation with Four Cascaded Kernels*
