---
title: "Global Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Global-Constraints"
toc_id: F~PdOmsYfrZyFfby6zWGsw
content_id: 9cfd7mviXyA_ti9jNY_goQ
---

#### Global Constraints

The GlobalConstraints section specifies global constraints.

###### Syntax

```
{
  "GlobalConstraints": {
    <constraint>[,
    <constraint>...]
  }
}
<constraint> ::= areaGroup
               | IsomorphicGraphGroup
```

###### Example

```
{
  "GlobalConstraints": {
    "areaGroup": {
      "name": "root_area_group",
      "nodeGroup": ["mygraph.k1", "mygraph.k2"],
      "tileGroup": ["(2,0):(2,3)"],
      "shimGroup": ["0:3"]
    },
    "isomorphicGraphGroup": {
      "name": "isoGroup1",
      "referenceGraph": "clipGraph0",
      "stampedGraphs": ["clipGraph1", "clipGraph2"]
    }
  }
}
```
