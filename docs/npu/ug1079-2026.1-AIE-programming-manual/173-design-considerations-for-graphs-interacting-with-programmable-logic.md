---
title: "Design Considerations for Graphs Interacting with Programmable Logic"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Design-Considerations-for-Graphs-Interacting-with-Programmable-Logic"
toc_id: HDdDRKHeiQYieZ~uVfNbcg
content_id: bLRVwaABLUB49DKgEmXTNg
---

## Design Considerations for Graphs Interacting with Programmable Logic

The AI Engine array includes AI Engine tiles and AI Engine array interface tiles located on the last row of the array. The types of interface tiles include AI Engine-PL and AI Engine-NoC.

The PL interface tile interfaces and adapts the signals between the AI Engines and the PL region. Knowledge of this is essential to take full advantage of the bandwidth between AI Engines and the PL. The following figure shows an expanded view of a single PL interface tile.

Following is a conceptual representation of the AI Engine-PL interface interacting with PL and AI Engine tiles:

![ndx1714153292522.png](../assets/173-01-ndx1714153292522-png-359cbe44c440.png)

*Figure 1. Conceptual Representation of AIE-PL Interface*

The CDC path between PL and AI Engine. The latency of the path can vary when PL frequency or phase changes.

Generally, the higher the frequency of the PL, the lower the latency in absolute time. And the higher the frequency of the PL, the higher the throughput or sample rate of the PL kernels.

**Note:** AI Engine

When using event APIs to do profiling, the probing points are inside the AXI4-Stream switch box of the AI Engine-PL interface. However, if you use the `--debug.aie.chipscope` option of v++, the ILA probing points are on the PL wrapper logic. Thus, there are multiple cycle differences between the two methods when measuring the latency of the AI Engine graph.

Also, the path inside the AI Engine-PL interface including the AXI4-Stream switch box has the capability of buffering. The `tready` signal from AI Engine is asserted after the device is booted, that is, even before the AI Engine graph is run by host code. So, if PL kernel starts transferring data to AI Engine, it fills all the buffers inside the AI Engine-PL interface, until back pressure occurs.

![lyq1586537410333.png](../assets/173-02-lyq1586537410333-png-42f1b4eab6d2.png)

*Figure 2. AI Engine-PL Interface Tile*

**Note:** AI Engine

AI Engine

**Note:**
