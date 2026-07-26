---
title: "Tiling Parameters Specification"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Tiling-Parameters-Specification"
toc_id: vLRZLrxa6Niu5BielxMvyw
content_id: J_7F6pfEfUypJUqYc6G5uw
---

## Tiling Parameters Specification

The graph specifies tiling parameters and associates them to an input or output port of an AI Engine memory DMA.

This example describes the following:

- six tiles (4x3 samples) that must be written to a 12x8 sample buffer by kernel k1
- six tiles (7x2 samples) that must be read out of the same buffer by kernel k2

![pzw1742999077543.png](../assets/080-01-pzw1742999077543-png-ff6da53c6109.png)

*Figure 1. Write Scheme*

![rvh1742999835350.png](../assets/080-02-rvh1742999835350-png-615cdc24cbad.png)

*Figure 2. Read Scheme*

```
kernel k1, k2;

mygraph()
{
  k1 = kernel::create(func1);
  k2 = kernel::create(func2);

  connect n1(k1.out[0], k2.in[0]);

  read_access(k1.out[0]) = tiling({
    .buffer_dimension={12,8},
    .tiling_dimension={4,3},
    .offset={0,0},
    .tile_traversal = {{.dimension=1, .stride=3, .wrap=2},
      {.dimension=0, .stride=4, .wrap=3}}});

  read_access(k2.in[0]) = tiling({
    .buffer_dimension={12,8},
    .tiling_dimension={7,2},
    .offset={0,0},
    .tile_traversal = {{.dimension=0, .stride=5, .wrap=2},
      {.dimension=1, .stride=2, .wrap=3}}});
};
```

With this construct, the two kernels communicate through two ping-pong buffers. A DMA transfer is automatically added by the AI Engine compiler. The DMA transfer is added between a ping-pong output buffer on the k1 side and a ping-pong input buffer on the k2 side. Within each tile, the system reads or writes data with dimension 0 as the inner loop. Tile selection follows the `tile_traversal` vector specification.

On the k1 side the MM2S DMA accesses the tiles column-wise as per the `tile_traversal` vector starts with dimension 1, which is followed by dimension 0.

On the k2 side the S2MM DMA accesses the tiles row-wise as per the `tile_traversal` vector which starts with dimension 0. The read access overlaps in dimension 0 as the specified stride in this dimension is less than the tile size.

**Note:** ```
Failed to allocate buffer descriptors for TG.G1.mtxin due to insufficient number of available buffer descriptors.
```
