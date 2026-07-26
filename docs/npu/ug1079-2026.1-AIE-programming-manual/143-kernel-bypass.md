---
title: "Kernel Bypass"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Bypass"
toc_id: ZEmqDHImJ8RfpcZr8NCBNA
content_id: rngjmipRmHqbkYjHGhiEPg
---

## Kernel Bypass

A bypass encapsulator construct executes a kernel conditionally. A runtime parameter controls the bypass.

The bypass runtime control input `bp` : `0` for no bypass and `1` for bypass. In addition to the control parameter, external connections from a bypassed kernel or a graph are redirected. The connections link to the external ports of the bypass construct itself. Internally, the bypass construct is connected to the bypassed kernel or the graph automatically by the compiler.

You can use the `negate` modifier to denote that the input control is negated. By default, the control is synchronous RTP. To use asynchronous RTP, use the `async` modifier.

The following example shows the required coding.

```
input_port control;
input_port in;
output_port out;
bypass b;
kernel k;
k = kernel::create(filter);
dimensions(k.in[0])={32};
dimensions(k.out[0])={32};
...
b = bypass::create(k);
connect<parameter> (control, async(negate(b.bp)));
connect(in, b.in[0]);
connect(b.out[0], out);
```

**Note:** The input and output buffer ports require a one-to-one correspondence (in type and size) for the bypass to work correctly.
