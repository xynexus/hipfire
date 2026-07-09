---
title: "Shim Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Shim-Constraint"
toc_id: m9Qzzh9IpozA46Yn5V_poA
content_id: qR52_8TY6fK52lG9xGQfRw
---

##### Shim Constraint

AI Engine

**Note:** You cannot place PLIOs in every column. The availability of columns is device dependent. For example, columns 0-5 cannot be used for PLIO for the xcvc1902-vsva2197-2MP-e-S device. Refer to the relevant device data sheet for more information.

###### Syntax

```
"shim": {
  "column": integer,
  "channel": integer (optional)
}
```

###### Example

```
{
  "NodeConstraints": {
    "plioOut1": {
      "shim": {
        "column": 0,
        "channel": 1
      }
    },
    "plioOut2": {
      "shim": {
        "column": 1
      }
    }
  }
}
```
