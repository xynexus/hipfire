---
title: "Reserved Memory Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Reserved-Memory-Constraint"
toc_id: iBudL8L8YzX7IILQLBj96g
content_id: uytCihmY3G7kf~4osdI93Q
---

##### Reserved Memory Constraint

This constrains the location of system memory (stack and heap) for a kernel to a specific address on a specific tile. You can specify the address in one of two ways:

- Column, row, bankId and offset, where column, row specifies the tile. The bankId is relative to the tile and can be set to either 0, 1, 2, or 3. The offset address is relative to the bankId. It can be set to any value from 0 to 8192 which corresponds to the maximum number of bytes in a bank.
- Column, row, and bankId, where the bankId is relative to the tile and can be set to either 0, 1, 2, or 3.Note: The hardware views of each AI Engine memory comprises eight banks of 128-bit width data. The aiecompiler views of each AI Engine memory comprises of four banks of 256-bit width data.

###### Syntax

```
"reserved_memory": <bank_address>
<bank_address> ::= {
  "column": integer,
  "row": integer,
  "bankId": integer,
  "offset": integer
}
<bank_address> ::= {
  "column": integer,
  "row": integer,
  "bankId": integer
}
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k1": {
      "reserved_memory": {
        "column": 2,
        "row": 1,
        "bankId": 3,
        "offset": 4128
      }
    },
    "mygraph.k2": {
      "reserved_memory": {
        "column": 1,
        "row": 1,
        "bankId": 3
      }
    }
  }
}
```
