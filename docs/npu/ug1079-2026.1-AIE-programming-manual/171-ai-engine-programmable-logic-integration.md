---
title: "AI Engine/Programmable Logic Integration"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine/Programmable-Logic-Integration"
toc_id: Y4Hv3eZiDRLExaQK9OjFiA
content_id: BN3QTvLedJOiSZJE8Hy9nw
---

# AI Engine/Programmable Logic Integration

When you are ready to consider interfacing to the programmable logic (PL), choose the platform to interface with. A platform is a fully contained image that defines both the hardware (XSA) as well as the software (bare metal, Linux, or both). The XSA contains the hardware description of the platform, which is defined in the AMD Vivado™ Design Suite, and the software is defined with the use of a bare-metal setup, or a Linux image defined through PetaLinux. Depending on your application requirements, can use an example reference platform provided by AMD, or a custom platform.

AMD recommends interfacing to the PLIO port attributes which represent external stream connections that cross the AI Engine-PL boundary. PLIO represents an ADF graph interface to the PL. This PL could be, for example:

- a PL kernel,
- a platform IP representing a signal source or sink, or
- a data mover to interface the ADF graph to memory

Alternatively interface connections can also be GMIO port attributes which represent external memory-mapped connections to or from the global memory. See Graph Programming Model for further details on attributes.

**Note:** AI Engine

Vitis
