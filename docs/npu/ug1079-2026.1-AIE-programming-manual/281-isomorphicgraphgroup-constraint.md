---
title: "IsomorphicGraphGroup Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/IsomorphicGraphGroup-Constraint"
toc_id: jLC3haOYKY6qmNx6~165vw
content_id: zXcvDaK37bunsZWBkJa9Fg
---

##### IsomorphicGraphGroup Constraint

You can use the isomorphicGraphGroup constraint to specify isomorphic graphs for use in the stamp and repeat flow.

###### Syntax

```
"isomorphicGraphGroup": {
  "name": string,
   "referenceGraph": <reference graph name>,
   "stampedGraphs": [<stamped graph name list>]
}
```

###### Example

```
  "isomorphicGraphGroup": {
  "name": "isoGroup",
   "referenceGraph": "tx_chain0",
   "stampedGraphs": ["tx_chain1", "tx_chain2", "tx_chain3"]
}
```

###### General Description

You can use the stamp and repeat feature of the AI Engine compiler when the same graph has multiple instances that can be constrained to the same geometry in AI Engines. There are two main advantages to using this feature when the same graph is instantiated multiple times:

- **Small variation in performance:** All graphs have very similar throughput because buffers and kernels are mapped identically with respect to each other. Throughput might not be exactly identical due to differences in routing. However, it much closer than when not using stamping.
- **Smaller run time of AI Engine compiler:** Because the AI Engine compiler only solves a reference graph instead of the entire design, runtime is significantly less than the default flow.

###### Capabilities and Limitations

If required, you can stamp multiple different graphs. For example, if a design contains four instances of a graph called tx_chain and four instances of rx_chain, then both sets of graphs can be independently stamped.

This feature is only supported for designs which have one or more sets of isomorphic graphs, with no interaction between the different isomorphic graph sets. All reference and stamped graphs must contain area group constraints. You must declare identical size area groups for each instance of the graph that needs to be stamped. All area groups must be non-overlapping. For example:

```
"areaGroup": {
   "name": "ant0_cores",
   "nodeGroup": ["tx_chain0*"],
   "tileGroup": ["(0,0):(3,3)"]
},
"areaGroup": {
   "name": "ant1_cores",
   "nodeGroup": ["tx_chain1*"],
   "tileGroup": ["(0,4):(3,7)"]
},
```

**Note:** The node group must contain all node instances in the graphs to be stamped. You can use pattern matching as in shown in the previous example.

You must declare an `isomorphic graph group` in the constraints file that specifies the reference graph and the stamped graphs. For example:

```
"isomorphicGraphGroup": {
  "name": "isoGroup",
   "referenceGraph": "tx_chain0",
   "stampedGraphs": ["tx_chain1", "tx_chain2"]
}
,
```

In this case, the `tx_chain0` graph is the reference and its objects are placed first and stamped to graph `tx_chain1` and `tx_chain2`. Area groups must follow these rules for number of rows: the number of rows for all identical graphs (reference + stamp-able ones) must be the same, and must begin and end at the same parity of row, meaning if the reference graph's tileGroup begins at an even row and ends at an odd row, then all of the stamped graphs must follow the same convention. This limitation occurs because of the mirrored tiles in AI Engine array. In one row, the AI Engine is followed by a memory group and in the next row the memory group is followed by an AI Engine within a tile.

**Recommended:** To add absolute, co-location, or other constraints to your design, only add constraints to the reference graph. All constraints automatically apply to the stamped graphs.

**Note:** Important:
