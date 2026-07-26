---
title: "Logical I/O Ports"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Logical-I/O-Ports"
toc_id: TH1MS~6RlMK3lzKibvnWnw
content_id: GQ~nymhmPyuiVa_5FmhRCw
---

## Logical I/O Ports

When designing a subgraph, it might not be feasible to know beforehand how an I/O port will be connected in the final system. It is probable that a subgraph output will link to an `output_plio` or `output_gmio` port. To overcome this issue, you can define logical I/O ports such as `port<input>`, `port<inout>`, and `port<output>` graph objects. This ensures that you can specify the inputs and outputs of the subgraph. It also enables the end-user to determine how those ports will physically connect in the system. These objects can establish links between kernels within a graph and across levels of hierarchy in your specifications that include platforms, graphs, and subgraphs.

Related reference

port<T>

Port Combinations
