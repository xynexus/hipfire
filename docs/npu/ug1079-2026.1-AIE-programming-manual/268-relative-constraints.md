---
title: "Relative Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Relative-Constraints"
toc_id: cVXQTy5kQw9D4edgCwNokw
content_id: B0aYUhSjgF8W2iVQXnpbMw
---

##### Relative Constraints

You can place kernel, PLIO and GMIO objects relative to each other. These types of constraints are called relative constraints. The X and Y offsets indicates the relative column and row offsets between the source node and the relative node. Specify either X offset or Y offset, otherwise it is "don't care".

###### Syntax

```
{
  "NodeConstraints": {
    "<source node>" : {
      "relative_nodes": ["<destination node>:(X=-3,Y=-2)", "<destination node>:(X=1)", "<destination node>:(Y=2)", ...]
    }
  }
}

<source node> ::= string
<destination node> :: string
```

###### Example

```
{
  "NodeConstraints": {
    "gr.k00": {
      "relative_nodes": ["gr.k01:(X=1,Y=1)","gr.k10:(X=2,Y=2)","gr.k11:(X=3,Y=-3)"]
    },
    "DatainA0": {
      "relative_nodes": ["Dataout00:(X=0)","Dataout01:(X=-1)","Dataout10:(X=-2)","Dataout11:(X=3)"]
    },
    "gr.k00": {
      "relative_nodes": ["DatainA0:(X=0)","DatainA1:(X=1)","DatainB0:(X=2)","DatainB1:(X=-3)"]
    },
    "DatainA0": {
      "relative_nodes": ["gr.k00:(X=0)"]
    }
  }
}
```
