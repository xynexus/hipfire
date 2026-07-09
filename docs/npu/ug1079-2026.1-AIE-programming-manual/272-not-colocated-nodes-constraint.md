---
title: "Not Colocated Nodes Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Not-Colocated-Nodes-Constraint"
toc_id: JCt11X2_Cu07PyhwekIZQg
content_id: tF9TXCG2MEcH2oI1cjaYcw
---

##### Not Colocated Nodes Constraint

This constrains a port (i.e., the port buffer) location to not be on the same tile as that of one or more kernels.

###### Syntax

```
"not_colocated_nodes": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "not_colocated_nodes": ["mygraph.k1"]
    }
  }
}
```
