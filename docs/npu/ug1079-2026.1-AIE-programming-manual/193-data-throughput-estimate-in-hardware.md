---
title: "Data Throughput Estimate in Hardware"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Throughput-Estimate-in-Hardware"
toc_id: 78EcZLwYKAVOZ9mXd6gZfw
content_id: zKgGtTHZaG4L0Vwa~9KzjA
---

### Data Throughput Estimate in Hardware

When a system does not meet performance requirements, the root cause can be one of the following:

- Some kernels of the graph cannot keep pace with the input/output sample rate, leading to a global performance degradation.
- The throughput at the AI Engine array interface does not meet the performance requirements for the following reasons:Maximum theoretical throughput is not sufficient. Real throughput obtained is drastically lower than expected.

  - Maximum theoretical throughput is not sufficient.
  - Real throughput obtained is drastically lower than expected.

The Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine)) lists the maximum achievable bandwidth of the various interfaces.

For the PLIOs, it depends on the frequency, bitwidth, and efficiency of the PL kernels. For the GMIOs, it depends on the processor, the DDR memory access pattern, and also on the traffic on the various NoC lanes.

A lower interface throughput does not always mean that the PL or DDR throughput is too low. It can be the graph itself that cannot keep pace of the sample rate, transmitting backpressure to the data source.

The number of DDR interfaces of the device can be: two in the VC1352, and four in the VC1802 and VC1902. The number of DDR chips on the board (DDR4, LPDDR4, ...) can be less than the maximum supported by the device. Furthermore, the platform that is built upon the device mounted on a board might support fewer memories than the number on the board. All these parameters have a direct impact on the maximum throughput achievable by the system you want to implement.

![lyz1678999164292.png](../assets/193-01-lyz1678999164292-png-1bbfdb9c821e.png)

*Figure 1. NoC Block Diagram*

Each DDR interface has four ports. These are connected through switches to the four independent lanes of the Horizontal NoC and also to the Vertical NoC just above. The Processor System has also two (APU + RPU) connections on each HNoC lane. The VNoC has two lanes going up the device, so the traffic coming from the four incoming lanes is combined over the two lanes. Each lane has two streams, one in each direction, and each stream has a maximum throughput of 14 GB/s.

The VCK 190 has four DDR ports which are connected to the HNoC and to the four VNocs above. Among these four ports, only three are declared in the base platform provided by AMD, thus limiting the available bandwidth to the AI Engine. There are 16 NMUs/NSUs on the VC1902, each one is capable of 16 GB/s of throughput in each direction. The top HNoC is fed by the four VNoCs, leading to a maximum of 8x14GB/s. This is much less than the theoretical 16x16 GB/s potential throughput using the 16 NMUs/NSUs.

You may be interested in the input or output throughput, or the overall throughput. The interface may be the PLIO or the GMIO to access the programmable logic or the DDR or the processing system. The preceding figure shows multiple DDR interfaces connected to an HNoC, which is connected to the PS and multiple VNoCs. On the top an HNoC connects to the different VNoCs on one side and to the AI Engine array on the other side.

Estimating the throughput (in Hardware) of the design at the AI Engine interfaces are possible using different methods, including:

- Timers
- Events
- Profiling
- Event trace
- NoC profiling (GMIO only)
