---
title: "Non-colocated Nodes Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Non-colocated-Nodes-Constraint"
toc_id: 4UjXKxp6a~6Tr6OwYY~Z8w
content_id: Fu9R7z_HeQTYmFghBFzAvw
---

##### Non-colocated Nodes Constraint

This constrains two or more kernels to not be on the same tile.

###### Syntax

```
"not_colocated_nodes": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k2": {
      "not_colocated_nodes": ["mygraph.k1"]
    }
  }
}
```
