---
title: "Multiple Kernels Coding Example: FIR Filter"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Multiple-Kernels-Coding-Example-FIR-Filter"
toc_id: RaoQoNy~12r5oj3_PkhtGQ
content_id: ggfN6eqCYguavlEhZvBArQ
---

## Multiple Kernels Coding Example: FIR Filter

This section uses the filter design to demonstrate how to split the application into multiple AI Engines when one engine cannot meet the computational demand. A finite impulse response (FIR) filter is a filter whose impulse response (or response to any finite length input) is of finite duration.

![eul1606806208916.png](../assets/236-01-eul1606806208916-png-8ed34e2e4b8c.png)

**Note:** AI Engine

K

In the previous equation, N denotes the taps to be used to calculate each output. The calculation process when a 32 taps filter is used as an example is shown in the following figure. int16 complex types for data and coefficient are also used as an example.

![igx1606887920100.png](../assets/236-02-igx1606887920100-png-c7875cf4dea6.png)

*Figure 1. FIR Filter*
