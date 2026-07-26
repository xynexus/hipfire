---
title: "bypass"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/bypass"
toc_id: 54beiGdBmBbwsb3YiOJWNA
content_id: XhybHeHEGodI4H0Rwjy~MA
---

### bypass

This class is a control flow encapsulator with data bypass. It wraps around an individual node or subgraph to create a bypass data path based on a dynamic control condition. The dynamic control is coded as a runtime parameter port `bp` (with integer value 0 or 1). Dynamic control determines whether:

- The input buffer (or stream) data flows into the kernel encapsulated by the bypass (`bp=0`), or
- The input buffer is directly bypassed into the output buffer (or stream) (`bp=1`).

##### Scope

You can declare `bypass` objects in class scope as member variables in a user-defined graph type. That is, inside a class that inherits from `graph`.

You must initialize `bypass` objects by assignment in the graph constructor.

##### Member Functions

```
static bypass & create( kernel );
```

The static create method creates a bypass object around a given kernel object. The number of inputs and outputs for the bypass is automatically inferred from the kernel’s corresponding ports.
