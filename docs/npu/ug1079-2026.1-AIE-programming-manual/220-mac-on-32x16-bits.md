---
title: "MAC on 32x16 bits"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-32x16-bits"
toc_id: JwLzNANoKxcVsb6q8YOhbQ
content_id: yEvkoyLQamFbrt8CTbgXJg
---

#### MAC on 32x16 bits

An example of MAC with pre-adding is as follows. With pre-adding, you can add data from the X buffer alone or combine data from both the X and Y buffers. The `start`, `offsets`, and `step` parameters work similar as previous example. There is a `ystart` parameter for `Y` buffer or another data from `X` buffer. The `step` parameter works reversely for `Y` or another data from `X` buffer.

![yhh1606892515564.png](../assets/220-01-yhh1606892515564-png-132f3483f88a.png)

*Figure 1. LMAC8_SYM on int32 x int16 Type*
