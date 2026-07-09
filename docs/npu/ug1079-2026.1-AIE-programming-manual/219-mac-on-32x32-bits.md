---
title: "MAC on 32x32 bits"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-32x32-bits"
toc_id: Ka4Muaf0B1xnGB~aiOIxXA
content_id: V6aRyIti5OM2ZSHzROpJMg
---

#### MAC on 32x32 bits

The following figure shows how `start`, `offsets`, and `step` work on the cint16 data type.

![bwq1606892453741.png](../assets/219-01-bwq1606892453741-png-68b0a5573a51.png)

*Figure 1. MAC4 on cint16 x cint16 Type*

`mac4` has four output lanes. The first column of data is selected by adding `xstart` to every 4 bits of `xoffsets`. The subsequent column of data is selected by adding `xstep` to its previous column.

The coefficients of `mac4` are chosen similarly by `zstart`, `zoffset`, and `zstep`.
