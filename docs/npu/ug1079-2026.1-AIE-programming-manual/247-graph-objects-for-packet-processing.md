---
title: "Graph Objects for Packet Processing"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Graph-Objects-for-Packet-Processing"
toc_id: EdKKFLrOpPq6uB30xNrPdg
content_id: UiI~MJ0ZPNnh7XFiMUeF5Q
---

### Graph Objects for Packet Processing

Use the following predefined object classes in the `adf` namespace to define the connectivity of packet streams.

```
template <int nway> class pktsplit { ... }
template <int nway> class pktmerge { ... }
```

##### Scope

Objects of type `pktsplit<n>` and `pktmerge<n>` can be declared as member variables in a user-defined graph type. That is, inside a class that inherits from `graph`. The template parameter `n` must be a compile-time constant positive integer denoting the n-way degree of split or merge. These objects behave like ordinary nodes of the graph with input and output connections, but are only used for explicit packet routing.

##### Member Functions

```
static pktsplit<nway> & create();
static pktmerge<nway> & create();
```

The static create method for these classes work in the same way as `kernel` create method. The degree of split or merge is already specified in the template variable declaration.

##### Member Variables

```
std::vector<port<input>> in;
```

This variable provides access to the logical inputs of the node. There is only one input for `pktsplit` nodes. For `pktmerge` nodes the i'th index selects the i’th input port.

```
std::vector<port<output>> out;
```

This variable provides access to the logical outputs of the node. There is only one output for `pktmerge` nodes. For `pktsplit` nodes the i'th index selects the i’th output port.
