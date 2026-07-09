---
title: "Packet Processing"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Packet-Processing"
toc_id: TLoMj_PMIUp3LprgfSygVQ
content_id: VtdWWpil7UsP4DVNeCllcw
---

#### Packet Processing

The first 32-bit word of a packet must always be a packet header, which encodes several bit fields as shown in the following table.

| Bits | Field |
| --- | --- |
| 4-0 | Packet ID |
| 11-5 | 7'b0000000 |
| 14-12 | Packet Type |
| 15 | 1'b0 |
| 20-16 | Source Row |
| 27-21 | Source Column |
| 30-28 | 3'b000 |
| 31 | Odd parity of bits[30:0] |

The compiler assigns the packet ID based on routing requirements. In the graph, the `pktsplit` output index and `pktmerge` input index can be different with the assigned packet ID. For example, the following figure shows a compilation result of a `pktsplit`:

![fgi1726780588674.png](../assets/121-01-fgi1726780588674-png-4faf4bde4226.png)

*Figure 1. Graph View of pktsplit with Packet ID*

From the above graph view, the index of the `pktsplit` output is shown first, followed by the corresponding packet ID. So, inside the AI Engine kernel, use the `getPacketid` API to query the packet ID. This ensures that the code is valid across different compilations.

The packet type can be any 3-bit pattern that you want to insert to identify the type of packet. The source row and column denote the AI Engine tile coordinates from where the packet originated. By convention, source row and column for packets originating in the programmable logic (PL) is -1,-1.

It is your responsibility to construct and send an appropriate packet header at the beginning of every packet. On the receive side, the packet header needs to be received and decoded before reading the data.

The following operations help to assemble or disassemble the packet header in the AI Engine kernel.

```
void writeHeader(output_pktstream *str, unsigned int pcktType, unsigned int ID);
void writeHeader(output_pktstream *str, unsigned int pcktType, unsigned int ID, bool tlast);


uint32 getPacketid(input_pktstream *w, int index);
uint32 getPacketid(output_pktstream *w, int index);
```

You can use the `writeHeader` API to assemble a packet header with a given packet ID and packet type. Query the packet ID inside the kernel using `getPacketid`. The source row and column are inserted automatically using the coordinates of the AI Engine tile where the API is executes.

The `getPacketid` API allows the compiler assigned packet ID to be queried on the input or output packet stream data structure. The index argument refers to the split or merge branch edge in the graph specification.

`getPacketid`

`writeHeader`

```
void aie_core1(...,output_pktstream *out){
  //Get ID from output pktstream, index=0
  uint32 ID=getPacketid(out,0);
  //Generate header for output
  writeHeader(out,pktType,ID);
......
```

See Explicit Packet Switching for more examples on different graphs and kernel code.

**Note:** Important:

`writeHeader()`

`getPacketid()`
