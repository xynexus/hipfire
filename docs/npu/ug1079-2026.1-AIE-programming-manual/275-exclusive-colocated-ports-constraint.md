---
title: "Exclusive Colocated Ports Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Exclusive-Colocated-Ports-Constraint"
toc_id: IFc~_b9djwH7zLUgnyFttg
content_id: qilZOBptUeymdREo5UlnSQ
---

##### Exclusive Colocated Ports Constraint

This constrains a port buffer location to be exclusively on the same bank as that of one or more other port buffers. This means no other port buffers can be on the same bank.

###### Syntax

```
"exclusive_colocated_ports": [<port list>]
<port list> ::= <port name>[, <port name>...]
<port name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "exclusive_colocated_ports": ["mygraph.k2.out[0]"]
    }
  }
}
```
