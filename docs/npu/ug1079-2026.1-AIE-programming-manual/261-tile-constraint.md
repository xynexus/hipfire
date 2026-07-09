---
title: "Tile Constraint"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Tile-Constraint"
toc_id: NS~~Pzmd0_0jokPNpvsuiQ
content_id: Ry3bGZCJbOnshdJ6SpJFgQ
---

##### Tile Constraint

This constrains a kernel to a specific tile located at a specified column and row within the array. The column and row values are zero based, where the zeroth row is counted from the bottom-most compute processor and the zero-th column is counted from the left-most column.

###### Syntax

```
"tile": {
  "column": integer,
  "row": integer
}
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k1": {
      "tile": {
        "column": 2,
        "row": 1
      }
    }
  }
}
```
