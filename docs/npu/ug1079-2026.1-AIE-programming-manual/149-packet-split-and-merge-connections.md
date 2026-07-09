---
title: "Packet Split and Merge Connections"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Split-and-Merge-Connections"
toc_id: wC8~wmqEHl3E6Slyiv25QQ
content_id: vhrzeQntaCvE7HjZCnDnrw
---

### Packet Split and Merge Connections

Multiple streams can share routing by connecting `pktmerge` to `pktsplit`. When buffers are transferred via `pktmerge` connected to `pktsplit`, each `pktmerge.in[i]` is routed to the corresponding `pktsplit.out[i]`. The in-degree of `pktmerge` must be equal to the out-degree of `pktsplit`. An example graph code is as follows:

```
for (int i=0; i<WAYS; i++) {
  connect<>(plioIn[i].out[0], kOut[i].in[0]);
  connect<>(kOut[i].out[0], merge.in[i]);
  connect<>(split.out[i], kIn[i].in[0]);
  connect<>(kIn[i].out[0], plioOut[i].in[0]);
}
connect<> (merge.out[0], split.in[0]);
```

The following figure shows the graph view.

![pnd1678899403657.png](../assets/149-01-pnd1678899403657-png-8920e3a21bc5.png)

*Figure 1. Pktmerge to Pktsplit Graph View*

##### Packet Split and Merge Sizes

Currently, packet switching up to 32 streams is supported. A maximum 32 to 1 `pktmerge`, and 1 to 32 `pktsplit` are supported. Using packet switching with large `fanout`/`fanin` (16/32 streams) can be resource expensive. Take care when using these in designs.

**Recommended:** AMD

##### Packet Split and Merge Broadcast

`pktmerge`

`pktsplit`

`pktsplit`

`pktmerge.in[i]`

`pktsplit.out[i]`

`merge.in[WAYS-1]`

`split.out[WAYS-1]`

`split.out[WAYS]`

```
for (int i=0; i<WAYS; i++) {
  connect<>(plioIn[i].out[0], kOut[i].in[0]);
  connect<>(kOut[i].out[0], merge.in[i]);
  connect<>(split.out[i], kIn[i].in[0]);
  connect<>(kIn[i].out[0], plioOut[i].in[0]);
}
connect<>(split.out[WAYS-1], kIn[WAYS].in[0]);
connect<>(kIn[WAYS].out[0],plioOut[WAYS].in[0]);
connect<> (merge.out[0], split.in[0]);
```

The graph view is shown in the following figure.

![met1678900235590.png](../assets/149-02-met1678900235590-png-3c728518ab28.png)

*Figure 2. Pktmerge to Pktsplit Graph View with Broadcast*
