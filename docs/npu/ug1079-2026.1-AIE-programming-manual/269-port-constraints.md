---
title: "Port Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Port-Constraints"
toc_id: hnAjEBbZFyOUERNkfGTNnQ
content_id: qaMRX08jBj3EP9vspB448g
---

#### Port Constraints

The PortConstraints section specifies port constraints. Constraints are grouped by port, allowing you to specify one or more constraints per port.

###### Syntax

```
{
  "PortConstraints": {
    "<port name>": {
      <constraint>[,
      <constraint>...]
    }
  }
}
<port name> ::= string
<constraint> ::= buffers
               | colocated_nodes
               | not_colocated_nodes
               | colocated_ports
               | not_colocated_ports
               | exclusive_colocated_ports
               | colocated_reserved_memories
               | not_colocated_reserved_memories
```

###### Example

```
{
  "PortConstraints": {
    "mygraph.k1.in[0]": {
      "colocated_nodes": ["mygraph.k1"]
    },
    "mygraph.k2.in[0]": {
      "colocated_nodes": ["mygraph.k2"]
    },
    "mygraph.p1": {
      "buffers": [{
        "column": 2,
        "row": 1,
        "bankId": 2
      }]
    }
  }
}
```

###### Port Names

You must specify ports by their fully qualified name: `<graph name>.<kernel name>.<port name>`. In the following example, the graph name is myGraph, the kernel name is k1, and the kernel has two ports named in[0] and out[0] (as specified in kernel1.cc). The fully specified port names are then myGraph.k1.in[0] and myGraph.k1.out[0].

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

Anytime either of these ports are referenced in the constraints JSON file, they must be named myGraph.k1.in[0] and myGraph.k1.out[0], as shown in the various examples throughout this document.
