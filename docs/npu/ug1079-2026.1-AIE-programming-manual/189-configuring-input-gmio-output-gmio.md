---
title: "Configuring input_gmio/output_gmio"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Configuring-input_gmio/output_gmio"
toc_id: 42xYQQ~83gSSDXSTuSYhSA
content_id: m19lU4XWnGC29rKgnjhWKg
---

## Configuring input_gmio/output_gmio

You can use an `input_gmio` or `output_gmio` object to make external memory-mapped connections to or from the global memory. These connections are made between an AI Engine graph and the logical global memory ports of a hardware platform design. The platform can be a base platform from AMD or a custom one. You can export a custom platform from Vivado as an AMD device support archive (XSA) package.

AI Engine tools support mapping the `input_gmio` or `output_gmio` ports to the tile DMA, one to one. It does not support mapping multiple `input_gmio`/`output_gmio` ports to one tile DMA channel. There is a limit on the number of `input_gmio`/`output_gmio` ports supported for a given device.

For example, the XCVC1902 device on the VCK190 board has 16 AI Engine to NoC master units (NMU) in total. For each AI Engine to NMU, it supports two MM2S and two S2MM channels. So, there can be at most 32 AI Engine GMIO inputs, and 32 AI Engine GMIO outputs. Note that it can be further limited by the existing hardware platform.

**Note:** AI Engine

When developing dataflow graph applications on an existing hardware platform, you need to know which global memory ports the underlying XSA exports. You also need to understand each port’s functionality to connect your application correctly. Any input or output ports exposed on the platform are recorded within the XSA. You can view them as a logical architecture interface.
