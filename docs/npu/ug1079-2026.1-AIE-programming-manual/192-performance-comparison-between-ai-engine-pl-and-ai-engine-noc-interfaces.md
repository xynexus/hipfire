---
title: "Performance Comparison Between AI Engine/PL and AI Engine/NoC Interfaces"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Performance-Comparison-Between-AI-Engine/PL-and-AI-Engine/NoC-Interfaces"
toc_id: oNbFE2DB6B8hfGmaDxJ81g
content_id: lLOCtGTWyDjVRLPoj3bVLA
---

## Performance Comparison Between AI Engine/PL and AI Engine/NoC Interfaces

The AI Engine array interface consists of the PL and NoC interface tiles. The AI Engine array interface tiles manage the two following high performance interfaces:

- AI Engine to PL
- AI Engine to NoC

The following image shows the AI Engine array interface structure.

![cob1762427533316.png](../assets/192-01-cob1762427533316-png-6d9213822713.png)

*Figure 1. AI Engine Array Interface Topology*

One AI Engine to PL interface tile contains:

- Eight streams from the PL to the AI Engine
- Six streams from the AI Engine to the PL

The following table shows one AI Engine to PL interface tile capacity.

| Connection Type | Number of Connections | Data Width (bits) | Clock Domain | Bandwidth per Connection (GB/s) | Aggregate Bandwidth (GB/s) |
| --- | --- | --- | --- | --- | --- |
| PL to AI Engine array interface | 8 | 64 | PL(500 MHz) | 4 | 32 |
| AI Engine array interface to PL | 6 | 64 | PL(500 MHz) | 4 | 24 |

**Note:** - A nominal 1 GHz AI Engine clock for a -1L speed grade device at VCCINT = 0.70V
- The PL interface running at half the frequency of the AI Engine

The exact number of PL and NoC interface tiles is device-specific. For example, in the VC1902 device, there are 50 columns of AI Engine array interface tiles. However, only 39 array interface tiles are available to the PL interface. Therefore, the aggregate bandwidth for the PL interface is approximately:

- 24 GB/s * 39 = 0.936 TB/s from AI Engine to PL
- 32 GB/s * 39 =1.248 TB/s from PL to AI Engine

The number of array interface tiles available to the PL interface and the total bandwidth of the AI Engine to PL interface for other devices varies across speed grades. See Versal AI Core Series Data Sheet: DC and AC Switching Characteristics ([DS957](https://docs.amd.com/go/en-US/ds957-versal-ai-core)) for more information.

The `input_gmio`/`output_gmio` attribute uses DMA in the AI Engine to NoC interface tile. The DMA has two 32-bit incoming streams from the AI Engine and two 32-bit streams to the AI Engine. In addition, it has one 128-bit memory mapped AXI master interface to the NoC NMU. The following table shows the performance of one AI Engine to NoC interface tile.

| Connection Type | Number of connections | Bandwidth per connection (GB/s) | Aggregate Bandwidth (GB/s) |
| --- | --- | --- | --- |
| AI Engine to DMA | 2 | 4 | 8 |
| DMA to NoC | 1 | 16 | 16 |
| DMA to AI Engine | 2 | 4 | 8 |
| NoC to DMA | 1 | 16 | 16 |

The exact number of AI Engine to NoC interface tiles is device-specific. For example, in the VC1902 device, there are 16 AI Engine to NoC interface tiles. So, the aggregate bandwidth for the NoC interface is approximately:

- 8 GB/s * 16 = 128 GB/s from AI Engine to PL
- 8 GB/s * 16 = 128 GB/s from PL to AI Engine

When accessing DDR memory, the integrated DDR memory controller (DDRMC) number in the platform limits the performance of DDR memory read and write. For example, if all four DDRMCs in a VC1902 device are fully used, the hard limit to access DDR memory is as follows.

- 3200 Mb/s * 64 bit * 4 DDRMCs / 8 = 102.4 GB/s

The performance of `input_gmio`/`output_gmio` accessing DDR memory through the NoC is further restricted by the NoC lane number in the horizontal and vertical NoC, inter-NoC configurations, and QoS. Note that DDR memory read and write efficiency is largely affected by the access pattern and other overheads. For more information about the NoC, memory controller use, and performance numbers, see the Versal Adaptive SoC Programmable Network on Chip and Integrated Memory Controller LogiCORE IP Product Guide ([PG313](https://docs.amd.com/access/sources/dita/map?Doc_Version=1.1%20English&url=pg313-network-on-chip)).

For a single connection from or to the AI Engine, both `input_plio`/`output_plio` and `input_gmio`/`output_gmio` have a hard bandwidth limit of 4 GB/s. Some advantages and disadvantages for choosing `input_plio`/`output_plio` or `input_gmio`/`output_gmio` are shown in the following table.

|  | input_plio/output_plio | input_gmio/output_gmio |
| --- | --- | --- |
| Advantages | Number of AI Engine to PL interface streams are larger, hence larger aggregate bandwidth No interference between different stream connections Supports packet switching | No PL resource required No timing closure requirement |
| Disadvantages | Congestion risk if there are too many stream connections in a region of the device Timing closure required for achieving best performance | Less input_gmio/output_gmio ports available Aggregate bandwidth is lower Multiple input_gmio/output_gmio ports competing for bandwidth |
