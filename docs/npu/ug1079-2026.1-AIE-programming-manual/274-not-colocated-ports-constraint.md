---
title: "Not Colocated Ports Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Not-Colocated-Ports-Constraint"
toc_id: AGTQp8jVg8HdbwPaMnwM5Q
content_id: o610fnJNAtRzTWu56i8ubw
---

##### Not Colocated Ports Constraint

This constrains a port buffer location to not be on the same bank as that of one or more other port buffers.

###### Syntax

```
"not_colocated_ports": [<port list>]
<port list> ::= <port name>[, <port name>...]
<port name> ::= string
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.in[0]": {
      "not_colocated_ports": ["mygraph.k2.out[0]"]
    }
  }
}
```
