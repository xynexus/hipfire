---
title: "Comparison between Buffer Ports and Windows"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Comparison-between-Buffer-Ports-and-Windows"
toc_id: vgv4xzilZyUtZ42A0tpI0Q
content_id: EJgCschddoxN9x_~c5K~_w
---

## Comparison between Buffer Ports and Windows

In AMD Vitis™ tools, version 2022.1 and previous versions, the memory interface between kernels was called window. Buffer ports are similar to window ports, but differ as outlined below. Buffer ports provide a way for a kernel to operate on a block of core memory.

| Windows | Buffers |
| --- | --- |
| Sizes and margins are in bytes. | Sizes and margins are in samples. |
| Sizes can be defined only in graph using the connect statement. | Sizes can be defined either in graph using dimensions statement, or in kernel signature as compile time constant. |
| Margins can be defined only in graph using the connect statement. | Margins can be defined only in kernel signature as compile time constant. |
| Graph specifies locking mechanism using connect construct. | Kernel signature specifies locking mechanism. |
| C++14 is enough. | Buffer ports use C++17 fold expressions. |
| Support of circular addressing only. | Default addressing is linear. You can set to circular if required. |
| For data access, the iteration operation is tied directly with the action reading or writing API, such as window_readincr(), and window_writeincr(). | A reference pointer or an iterator are supported to perform read or write operations. |
| One-dimensional only | Multi-dimensional. If buffer dimension is omitted in the kernel signature, it defaults to one-dimensional. |

**Note:** Limitations on buffer port size, window size and margin size depend on hardware constraints, hence, there is no difference between buffer port and window in terms of minimum size.
