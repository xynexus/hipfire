---
title: "Colocated Ports Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Colocated-Ports-Constraint"
toc_id: lefBh5MC3o5lK71wKtK0Ng
content_id: _ldIm8qIiWIGFEeQl_Oquw
---

##### Colocated Ports Constraint

This constrains a ports buffer location to be on the same bank as that of one or more other port buffers. When two double buffers are co-located, this constrains both ping buffers one bank and both pong buffers to another bank.

###### Syntax

```
"colocated_ports": [<port list>]
<port list> ::= <port name>[, <port name>...]
<port name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "colocated_ports": ["mygraph.k2.out[0]"]
    }
  }
}
```
