---
title: "Colocated Reserved Memories Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Colocated-Reserved-Memories-Constraint"
toc_id: vRhDezkYumcyS_Kmr5qm2A
content_id: nbSyPIj90iQs11cJSo6Sng
---

##### Colocated Reserved Memories Constraint

This constrains a kernel location to be on the same tile as that of one or more stacks. This ensures the kernel can access the stacks without requiring a DMA.

###### Syntax

```
"colocated_reserved_memories": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k2": {
      "colocated_reserved_memories": ["mygraph.k1"]
    }
  }
}
```
