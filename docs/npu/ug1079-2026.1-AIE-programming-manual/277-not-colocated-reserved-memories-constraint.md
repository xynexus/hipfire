---
title: "Not Colocated Reserved Memories Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Not-Colocated-Reserved-Memories-Constraint"
toc_id: dyFeRoelA1lndJXDuRgeHg
content_id: 9elAThZX8NYEuFZMxM7F4A
---

##### Not Colocated Reserved Memories Constraint

This constrains a ports buffer location to not be on the same bank as that of one or more stacks.

###### Syntax

```
"not_colocated_reserved_memories": [<port list>]
<port list> ::= <port name>[, <port name>...]
<port name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "not_colocated_reserved_memories": ["mygraph.k1"]
    }
  }
}
```
