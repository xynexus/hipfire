---
title: "Colocated Reserved Memories Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Colocated-Reserved-Memories-Constraint"
toc_id: apehMveROrNAx2OTMEZg8Q
content_id: zggCDVe5lllXs4AbDWtQSw
---

##### Colocated Reserved Memories Constraint

This constrains a ports buffer location to be on the same bank as that of one or more stacks.

###### Syntax

```
"colocated_reserved_memories": [<port list>]
<port list> ::= <port name>[, <port name>...]
<port name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "colocated_reserved_memories": ["mygraph.k1"]
    }
  }
}
```
