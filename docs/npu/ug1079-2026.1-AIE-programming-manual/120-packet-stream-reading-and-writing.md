---
title: "Packet Stream Reading and Writing"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Stream-Reading-and-Writing"
toc_id: wm6QL2qXwbUbrcCO_FOpLQ
content_id: L012owgjk7aTQi7uN3yEJw
---

#### Packet Stream Reading and Writing

A data packet includes a 32-bit packet header word, followed by data words. The last data word uses the TLAST field to signal the end-of-packet. Use the following operations to read and advance input packet streams and write and advance output packet streams.

```
int32 readincr(input_pktstream *w);
int32 readincr(input_pktstream *w, bool &tlast);

void writeincr(output_pktstream *w, int32 value);
void writeincr(output_pktstream *w, int32 value, bool tlast);
```

The API with the TLAST argument helps to read or write the end-of-packet condition if the packet size is not fixed.

If the packet header word has sideband TLAST set to true, it signals that the packet is complete contains no additional data. That is, the packet contains only the packet header.
