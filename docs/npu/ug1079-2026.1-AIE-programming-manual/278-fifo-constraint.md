---
title: "FIFO Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/FIFO-Constraint"
toc_id: n0NvhT9OzavdLT5vH3jpiQ
content_id: 6PKt8GlWVc9X2ioaoIQvBg
---

##### FIFO Constraint

This constrains a FIFO to a specific tile located at a specified column and row within the array. The tile can be an AI Engine tile, memory tile, or interface tile. The bankId is optional; if omitted, the compiler selects an optimal bank.

You can specify the FIFO constraint in one of the following ways:

- Column, row, bankID, offset and size. Where column, row, and bankID specifies the tile. The offset address is relative to the bankID, and the size starting at zero with a maximum value of 8188 32-bit words of a bank.

  Column, row, bankID, offset and size. Where column, row, and bankID specifies the tile. The offset address is relative to the bankID, and the size starting at zero with a maximum value of 8188 32-bit words of a bank.
- Column, row, and bankId, where the bank ID is relative to the tile and can take values 0, 1, 2, or 3.Note: The hardware view is 8 banks of 128-bit width but the software view is 4 banks of 256 width.

The following code block shows the syntax.

```
"PortConstraints": [<fifo list>]
<fifo list> ::= <fifo type>[, <fifo type>...]
<fifo type> ::= <dma_fifos> | <stream_fifos>
<dma_fifos> ::= <aie_tile>
<aie_tile> ::= {
  "fifo_id": string,
  "tile_type": "core",
  "column": integer,
  "row": integer,
  "size": integer,
  "offset": integer,
  "bankId":integer (optional)
}
<stream_fifos> ::= {
  "fifo_id": string,
  "tile_type": "shim",
  "column": integer,
  "row": integer,
  "channel": integer
}
```

The following code shows an example.

```
{
  "PortConstraints": {
    "fifo_locations_records": {
      "dma_fifos": {
        "r1": {
          "tile_type": "core",
           "row": 0,
           "column": 0,
           "size": 16,
           "offset": 8,
           "bankId": 2
        },
        "r2": {
          "tile_type": "core",
          "row": 0,
          "column": 1,
          "size": 16,
          "offset": 9
        },
        "r4": {
          "tile_type": "mem",
          "row": 2,
          "column": 4,
          "size": 16,
          "offset": 6,
          "bankId": 2
        }
      },
      "stream_fifos": {
        "r3": {
          "tile_type": "shim",
          "row": 1,
          "column": 3,
          "channel": 1
        }
      }
    },
    "mygraph.k2.in[0]": {
      "not_colocated_nodes": ["mygraph.k1"],
      "fifo_locations": ["r1", "r2", "r3"]
    },
    "mygraph.k4.in[0]": {
      "fifo_locations": ["r1", "r2", "r4"]
    }
  }
}
```
