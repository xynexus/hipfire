---
title: "Colocated Nodes Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Colocated-Nodes-Constraint"
toc_id: g6aL2OQ759rnIldcQnBZSA
content_id: s0GMzSqU7lbaX6KmZQAhVw
---

##### Colocated Nodes Constraint

This constrains a port (thatis, the port buffer) location to be on the same tile as that of one or more kernels. This ensures that other kernels can access the data buffer without requiring a DMA.

###### Syntax

```
"colocated_nodes": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k1.in[0]": {
      "colocated_nodes": ["mygraph.k1"]
    },
    "mygraph.k2.in[0]": {
      "colocated_nodes": ["mygraph.k2"]
    }
  }
}
```
