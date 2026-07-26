---
title: "Non-colocated Reserved Memories Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Non-colocated-Reserved-Memories-Constraint"
toc_id: nJPpmqRD4VQNE4QHX3XDYw
content_id: JdmO1pWdU_Yfq0EiLUAmmA
---

##### Non-colocated Reserved Memories Constraint

This constrains a kernel location so that it is not on the same tile as the AI Engine stack memory.

###### Syntax

```
"not_colocated_reserved_memories": [<node list>]
<node list> ::= <node name>[,<node name>...]
<node name> ::= string
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k2": {
      "not_colocated_reserved_memories": ["mygraph.k1"]
    }
  }
}
```
