---
title: "Input and Output Buffers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Input-and-Output-Buffers"
toc_id: hSPhMrsjOSXbaAlJE6sHcg
content_id: M_j7_dvH6JuVPXhf0uMFCw
---

# Input and Output Buffers

Input and output buffers represent a block of data that is stored contiguously on a tile’s physical memory, and that can be used by kernels in a graph. The origin of this data can be either other kernels that produce them, or they can come from the PL through AI Engine array interface. You can allocate the buffer port in the tile’s physical memory where the kernel executes, or in the physical memory of accessible adjacent tiles.

When a kernel has a buffer port on its input side, it waits for the buffer to be fully available before it starts execution. The kernel can access the contents of the buffer port either randomly or in a linear fashion. Conversely the kernel can write a block (frame) of data to the local memory. That block can be used by other kernels after it has finished execution.

When the source of a buffer is a stream, this stream is sliced into contiguous blocks. These blocks are stored one by one into buffers, as illustrated in the following figure.

![cxj1669804546545.png](../assets/084-01-cxj1669804546545-png-df898efd7ee0.png)

*Figure 1. Data Stream Slicing Into Buffers*

The following figure shows an example of a kernel buffer port in local tile memory. Tile (10,0) contains kernel K1, and the input buffer port is allocated in the local memory of the same tile (10,0).

![sxe1678905107272.png](../assets/084-02-sxe1678905107272-png-aa44e6ffde5f.png)

*Figure 2. Kernel Buffer Port in Local Tile Memory*

The following figure shows an example of a kernel buffer port that is allocated in a neighboring tile memory. The kernel k2 is in tile (11,0) and input buffer port at neighbor tile (10,0).

**Note:** For AI Engine devices, a single buffer port is no larger than 32 KB.

![mth1678905155051.png](../assets/084-03-mth1678905155051-png-332144f38222.png)

*Figure 3. Kernel Buffer Port in Neighboring Tile Memory*
