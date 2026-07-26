---
title: "Node Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Node-Constraints"
toc_id: yFsBFwPdmMA8pW8scpyonA
content_id: 4JMAFOA02KU~sB2H6XmHGQ
---

#### Node Constraints

The `NodeConstraints` section constrains graph nodes. Constraints are grouped by node, such that one or more constraints can be specified per node.

###### Syntax

```
{
  "NodeConstraints": {
    "<node name>": {
      <constraint>,
      <constraint>,
      ...
    }
  }
}
<node name> ::= string
<constraint> ::= tile
               | shim
               | reserved_memory
               | colocated_nodes
               | not_colocated_nodes
               | colocated_reserved_memories
               | not_colocated_reserved_memories
```

###### Example

```
{
  "NodeConstraints": {
    "mygraph.k1": {
      "tile": {
        "column": 2,
        "row": 1
      },
      "reserved_memory": {
        "column": 2,
        "row": 1,
        "bankId": 3,
        "offset": 4128
      }
    },
    "mygraph.k2": {
      "tile": {
        "column": 2,
        "row": 2
      }
    }
  }
}
```

###### Node Names

You must specify nodes by their fully qualified name, for example: `<graph name>.<kernel name>`.

In the following example, the graph name is `myGraph` and the kernel name is `k1`. The fully specified node name is `myGraph.k1`.

```
class graph : public adf::graph {
private:
  adf::kernel k1;
public:
  my_graph() {
    k1 = kernel::create(kernel1);
    source(k1) = "src/kernels/kernel1.cc";
  }
};
graph myGraph;
```

Whenever this kernel is referenced in the constraints JSON file, name it `myGraph.k1`, as shown in the various examples throughout this document.
