---
title: "Graph Execution Control"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Graph-Execution-Control"
toc_id: HwxCOF3vZbwjLJKAxi50CQ
content_id: _w655tFl3NIBxLy_CmEMvg
---

## Graph Execution Control

In AMD Versal™ Adaptive SoCs with AI Engines, the processing system (PS) can be used to dynamically load, monitor, and control the graphs that are executing on the AI Engine array. Even if the AI Engine graph is loaded once as a single bitstream image, you can use the PS program to monitor execution state and modify runtime parameters of the graph.

The `graph` base class provides APIs to control the initialization and execution of the graph that can be used in the `main` program. The user `main` application used in simulating the graph does not support `argc`, `argv` parameters.

Related reference

Adaptive Data Flow Graph Specification Reference
