---
title: "AI Engine Architecture Overview"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-Architecture-Overview"
toc_id: E1ZAhTRZJ6fnbORWH7~z0w
content_id: 1AVfJe8dYvoDOx0UA5JfXg
---

## AI Engine Architecture Overview

The AI Engine array consists of a 2D array of AI Engine tiles. Each AI Engine tile contains an AI Engine, memory module, and tile interconnect module. The AI Engine is a highly-optimized processor featuring a single-instruction multiple-data (SIMD) and very long instruction word (VLIW) instruction set architecture containing six functional units: scalar, vector, two load, one store, and one instruction fetch and decode.

One VLIW instruction supports a maximum of the following elements:

- Two loads
- One store
- One scalar operation
- One fixed-point or floating-point vector operation
- Two move instructions

There is also a memory module available. The memory module is shared between its north, south, east, or west AI Engine neighbors, depending on the location of the tile within the array. An AI Engine can access its north, south, east, or west, and its own memory module.

![rvo1606893424499.png](../assets/003-01-rvo1606893424499-png-ba4640288be0.png)

*Figure 1. AI Engine Tile Details*

Each AI Engine tile has an AXI4-Stream switch that is a fully programmable 32-bit AXI4-Stream crossbar. The switch supports both circuit-switched and packet-switched streams with back-pressure. Through MM2S DMA and S2MM DMA, the AXI4-Stream switch provides stream access to and from AI Engine data memory. The switch also contains two 16-deep 33-bit (32-bit data + 1-bit TLAST) wide FIFOs, which can be chained to form a 32-deep FIFO by circuit-switching the output of one of the FIFOs to the other FIFO’s input.

The Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine)) contains more details on the AI Engine architecture.
