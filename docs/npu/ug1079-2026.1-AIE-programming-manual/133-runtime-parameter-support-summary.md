---
title: "Runtime Parameter Support Summary"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Runtime-Parameter-Support-Summary"
toc_id: EE3Q1mAkYwmrSPfKN4OsQA
content_id: GpWJfS9fEdBcosPd_F~5UQ
---

## Runtime Parameter Support Summary

This section summarizes the AI Engine runtime parameter (RTP) support status.

| AI Engine RTP(from/to PS) | Input | Output |  |  |
| --- | --- | --- | --- | --- |
| Synchronous | Asynchronous | Synchronous | Asynchronous |  |
| Scalar | Default | Supported | Supported | Default |
| Array | Default | Supported | Supported | Default |

Code snippets for RTP connections from or to the PS:

```
//Synchronous RTP, default for input
connect<parameter>(fromPS, first.in[0]);
//Synchronous RTP
connect<parameter>(fromPS, sync(first.in[0]));
//Asynchronous RTP
connect<parameter>(fromPS, async(first.in[0]));
//Asynchronous RTP, default for output
connect<parameter>(second.inout[0], toPS);
//Asynchronous RTP
connect<parameter>(async(second.inout[0]), toPS);
//Synchronous RTP
connect<parameter>(sync(second.inout[0]), toPS);
```
