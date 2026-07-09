---
title: "Kernel Inputs and Outputs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Inputs-and-Outputs"
toc_id: P7dutpoE2QfxAgaLNiCnyg
content_id: 0uue0e7SRQlLyF~hNK466w
---

## Kernel Inputs and Outputs

AI Engine kernels operate on either streams or blocks of data. AI Engine kernels operate on data of specific types, for example, int32 and cint32. A buffer stores blocks of data used by a kernel. Kernels consume input streams and/or buffers, and produce output streams and/or buffers. Kernels access data streams in a sample-by-sample fashion. For additional details on stream APIs, see Streaming Data API. For additional details on buffers, see Input and Output Buffers.

AI Engine kernels can also have RTP ports which are accessible by the PS. For more information about RTP, see Runtime Graph Control API.
