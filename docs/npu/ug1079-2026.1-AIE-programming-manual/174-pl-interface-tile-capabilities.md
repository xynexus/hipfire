---
title: "PL Interface Tile Capabilities"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/PL-Interface-Tile-Capabilities"
toc_id: StyySNYqwcY~PTA3x~zJxw
content_id: zDQ07sdexyaWlaeFBF6vKg
---

### PL Interface Tile Capabilities

The AI Engine clock can run at up to 1 GHz for -1L speed grade devices, or higher, for -2 and -3 speed grade devices. The default width of a stream channel is 32 bits. Because this frequency exceeds the PL clock frequency, you must perform a CDC to the PL region, for example. For example, to either one-half or a quarter of the AI Engine clock frequency.

**Recommended:** AMD

AI Engine

For C++ HLS PL kernels, choose an appropriate target frequency depending on the complexity of the algorithm implemented. The `--hls.clock` option can be used in the Vitis compiler when compiling HLS C/C++ into Xilinx object (XO) files.
