---
title: "Packet Switching and the aiesimulator"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Switching-and-the-aiesimulator"
toc_id: k_LdKk7DjLtb1hXZIktvbw
content_id: S7LBILobBAgH4oVCyvCO0w
---

### Packet Switching and the aiesimulator

The `aiesimulator` supports explicit packet switching. Consider the example of the previous graph that expects packet switched data from the PL. The data is split inside the AI Engine and sent to four AI Engine kernels. On the output side the four kernel outputs merge into one output stream to the PL.

The input data file from the PL contains the packet switched data from the PL for the four AI Engine kernels in the previous example. It contains the data for different kernels, packet by packet. Each packet of data is for one iteration of an AI Engine kernel. The data format is as follows.

```
2415853568
0
1
2
3
4
5
6
TLAST
7
```

`2415853568`

`0x8fff0000`

**Note:** `aiesimulator`

You can construct the header for each packet manually, or write helper functions to generate the header. The AI Engine compiler generates a packet switching report file Work/reports/packet_switching_report.json that lists the packet IDs used in the graph. Additionally, it also generates Work/temp/packet_ids_c.h and Work/temp/packet_ids_v.h header files. You can include these in your C or Verilog kernel code.
