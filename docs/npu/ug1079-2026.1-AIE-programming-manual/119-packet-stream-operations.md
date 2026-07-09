---
title: "Packet Stream Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Stream-Operations"
toc_id: 4N3Ni8IQnko6l1e_5Q52gQ
content_id: msmf7ges76Zu2kG2eaH4HA
---

### Packet Stream Operations

| Input Stream Types | Output Stream Types |
| --- | --- |
| input_pktstream | output_pktstream |

Two additional stream data types characterize streaming data that consists of packetized interleaving of several different streams. These data types are useful when the number of independent data streams in your program exceeds the number of hardware stream channels or ports available. Explicit Packet Switching describes this mechanism in more detail.
