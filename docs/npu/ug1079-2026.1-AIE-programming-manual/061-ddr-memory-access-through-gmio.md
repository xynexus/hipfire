---
title: "DDR Memory Access through GMIO"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/DDR-Memory-Access-through-GMIO"
toc_id: 9s2kJPYdCK~WM3tnDMYRLw
content_id: GAlnB5Aiam4nCn9_QGKZ7g
---

## DDR Memory Access through GMIO

The main data streams to and from the AI Engine are as follows:

- The AI Engine to PL streaming interface
- GMIO, which is used to make external memory-mapped connections to or from the global memory.

The interface between the PS and AI Engine has a low throughput and is ideal for configuration. The AI Engine-GMIO directly connects to the DDR memory through the AI Engine-NoC master unit (NMU).

The bandwidth of AI Engine GMIO depends on the number of NMUs and DDR memory controllers used in the platform.

Related reference

Configuring input_gmio/output_gmio
