---
title: "Colocated Nodes Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Colocated-Nodes-Constraint"
toc_id: SCUMGE~SWNtbemY2k08e5A
content_id: B75xviTt6hsSs5~rgBkgTA
---

##### Colocated Nodes Constraint

The colocated nodes constraint requires two or more kernels to be on the same tile and forces sequencing of the kernels in a topological order. It also allows them to share memory buffers without synchronization.

###### Syntax

```
"colocated_nodes": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k2": {
      "colocated_nodes": ["mygraph.k1"]
    }
  }
}
```
