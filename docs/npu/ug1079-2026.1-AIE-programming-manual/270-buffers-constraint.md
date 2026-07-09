---
title: "Buffers Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffers-Constraint"
toc_id: swjkGDYw25acLPwiKK_t2A
content_id: D~mZLUg2OzEo_3u~rVgUqQ
---

##### Buffers Constraint

This constrains a data buffer to a specific address on a specific tile. You can attach the data buffer to an input, output, or inout port of a kernel or param (for example, a lookup table). The address can be specified in one of three ways:

- Column, row, and offset: where column and row specifies the tile. The the offset address is relative to the tile, starting at zero with a maximum value of 32768 (32k).
- Column, row, and bankId: where the bank ID is relative to the tile and can take values 0, 1, 2, 3.
- Offset: that can be between zero and 32768 (32k) and is relative to the tile allocated by the compiler.

**Note:** You can constrain one or two buffer for a port.

###### Syntax

```
"buffers": [<address>, <(optional) address>]
<address> ::= <offset_address> | <bank_address> | <offset_address>
<tile_address> ::= {
  "column": integer,
  "row": integer,
  "offset": integer
}
<bank_address> ::= {
  "column": integer,
  "row": integer,
  "bankId": integer
}
<offset_address> ::= {
  "offset": integer
}
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k2.out[0]": {
      "buffers": [{
        "column": 2,
        "row": 2,
        "offset": 5632
      }, {
        "column": 2,
        "row": 2,
        "offset": 4608
      }]
    },
    "mygraph.k1.out[0]": {
      "buffers": [{
        "column": 2,
        "row": 3,
        "bankId": 2
      }, {
        "column": 2,
        "row": 3,
        "bankId": 3
      }]
    },
    "mygraph.p1": {
      "buffers": [{
        "offset": 512
      }]
    }
  }
}
```
