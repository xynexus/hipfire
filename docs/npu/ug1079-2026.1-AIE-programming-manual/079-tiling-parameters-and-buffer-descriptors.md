---
title: "Tiling Parameters and Buffer Descriptors"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Tiling-Parameters-and-Buffer-Descriptors"
toc_id: BVN90mSaCRcX0Zcb_kOFZQ
content_id: KE5ROSU0EXX3AwFsixpUAA
---

## Tiling Parameters and Buffer Descriptors

Buffer descriptors located in AI Engine memory manage DMA transfers.

These buffer descriptors handle 1D and 2D memory addressing, multiple iterations, lock ID, and buffer descriptor chaining.

AI Engine

`tiling_parameters`

```
struct tiling_parameters
{
  std::vector<uint32_t> buffer_dimension;
  std::vector<uint32_t> tiling_dimension;
  std::vector<int32_t> offset;
  std::vector<traversing_parameters> tile_traversal;
  int packet_port_id = -1;
  std::vector<uint32_t> boundary_dimension;
};
```

The members of this structure are:

- **`buffer_dimension`:** Buffer dimensions in the memory element type (for example, AI Engine memory). `buffer_dimension[0]` is contiguous in memory and has the fastest access. If you do not specify this member, the dimensions of the associated memory object are used. The AI Engine memory can access data in the first and second dimensions.
- **`tiling_dimension`:** Tiling dimensions of the data transfer in buffer. The tiling dimension of AI Engine memory can access data in the first and second dimensions.
- **`offset`:** Multidimensional offset with respect to the starting element in the buffer, assuming the buffer dimension is specified.
- **`tile_traversal`:** Vector of `traversing_parameters`.`tile_traversal[i]` represents the i-th loop of inter-tile traversal, where i=0 represents most inner loop and i=N-1 represents most outer loop. `tile_traversal` structure is detailed the section below.
- **`packet_port_id`:** Multiple connections can go through a single port that are previously merged through a `pktmerge` block or split afterward with a `pktsplit` block. This member represents the output port ID of the connected `pktsplit` or the input port ID of the connected `pktmerge`. If this member is set to a specific id, the data transfer only occurs if the incoming or outgoing data block ID matches this ID.

- **`boundary_dimension`:** Real data boundary dimension.

The `tile_traversal` vector is a a key member of the tiling parameter. It describes buffer access. The structure `traversing_parameters` is as follows:

```
struct traversing_parameters
{
  uint32_t dimension;
  uint32_t stride;
  uint32_t wrap;
};
```

The members of this structure are as follows:

- **`dimension`:** The buffer dimension on which this traversing loop applies. It can be the 0 or first dimension. The stride and wrap members of this structure are applied in the specified dimension.
- **`stride`:** Represents the distance in terms of buffer element data type between consecutive inter-tile traversal in this dimension.
- **`wrap`:** Number of tiles to access in this dimension.

When the stride value is lower than the tile size in one or more dimensions, the tiles overlap naturally in that dimension.

**Note:** Important:

- 16-bit data: access as pairs.
- 8-bit data: access as fours.
- 4-bit data: access as eights.
