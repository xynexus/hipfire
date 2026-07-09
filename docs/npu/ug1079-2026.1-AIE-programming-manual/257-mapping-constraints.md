---
title: "Mapping Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Mapping-Constraints"
toc_id: icmHKtTNFohAHps1k1_b~Q
content_id: RzeHRZNrfbz4xMEOfMfj6A
---

### Mapping Constraints

The following functions help to build various types of constraints on the physical mapping of the kernels and buffers onto the AI Engine array.

##### Scope

A constraint must appear inside a user graph constructor.

##### Kernel Location Constructors

```
location_constraint tile(int col, int row)
```

This location constructor points to a specific AI Engine tile located at specified column and row within the AI Engine array. The column and row values are zero-based. The zero'th row is counted from the bottom-most row with a compute processor. The zero'th column is counted from the left-most column. The previously used constructor `proc(col,row)` is now deprecated.

```
location_constraint location<kernel> (kernel&)
```

This constraint provides a handle to the location of a kernel. This means you can constrain it on a specific tile or co-locate it with another kernel using the following assignment operator.

##### Buffer Location Constructors

```
location_constraint address(int col, int row, int offset)
```

This location constructor points to a specific data memory address offset on a specific AI Engine tile. The offset address is relative to that tile starting at zero with a maximum value of 32768 (32K).

```
location_constraint bank(int col, int row, int bankid)
```

AI Engine

**Note:** The hardware view is 8 banks of 128-bit width. The software view is 4 banks of 256 width.

```
location_constraint offset(int offset_value)
```

This location constructor specifies data memory address offset. The offset address is between 0 and 32768 (32K) and is relative to a tile allocated by the compiler.

```
location_constraint location<buffer> (port<T>&)
```

This location constructor provides a handle to the location of a buffer attached to an input, output, or inout port of a kernel. You can use it to constrain the location of the buffer to:

- a specific address
- a specific bank
- be on the same tile as another kernel
- be on the same bank as another buffer

using the following assignment operator. It is an error to constrain two buffers to the same address. This constructor only applies to buffer kernel ports.

```
location_constraint location<stack> (kernel&)
```

This location constructor returns a handle to the AI Engine’s system memory location—stack and heap—where the specified kernel is mapped. This provides a mechanism to constrain the location of the system memory with respect to other buffers used by that kernel.

**Note:** Important:

```
location_constraint location<parameter> (parameter&)
```

This location constructor provides a handle to the location of the parameter array (for example, a lookup table) declared within a graph.

##### Bounding Box Constructor

```
location_constraint bounding_box(int column_min, int row_min, int column_max, int row_max)
```

The bounding box constructor specifies a rectangular bounding box for a graph to be placed in AI Engine tiles. Placement is between columns from `column_min` to `column_max` and rows from `row_min` to `row_max`. You can use multiple bounding box location constraints in an initializer list to specify an irregular shape bounding region.

##### Operator Functions

```
location_constraint& operator=(location_constraint)
```

The operator expresses the equality constraint between two location constructors. It supports expressing various absolute or relative location constraints..

The following example shows how to constrain a kernel to be placed on a specified AI Engine tile.

```
location<kernel>(k1) = tile(3,2);
```

The following template shows how to constrain the location of double buffers attached to a port, placing them on a specific address or a bankid. To constrain the location of the double banks, specify two elements (maximum) in the initializer list. If a DMA engine reads or writes these buffers, they must reside on the same tile.

```
location<buffer>(port1) = { [address(c,r,o) | bank(c,r,id)] , [address(c,r,o) | bank(c,r,id)] };
```

The following template shows how to constrain the location of a parameter lookup table or the system memory of a kernel to be placed on a specific address or a bankid.

```
location<parameter>(param1) = [address(c,r,o) | bank(c,r,id)];

location<stack>(k1) = [address(c,r,o) | bank(c,r,id)];
```

The following example shows how to constrain two kernels to be placed relatively on the same AI Engine. This forces them to be sequenced in topological order and be able to share memory buffers without synchronization.

```
location<kernel>(k1) = location<kernel>(k2);
```

The following example shows how to constrain a buffer, stack, or parameter location to be on the same tile as that of a kernel. This ensures that the other kernel k1 can access the buffer, or parameter array without requiring a DMA.

```
location<buffer>(port1) = location<kernel>(k1);

location<stack>(k2) = location<kernel>(k1);

location<parameter>(param1) = location<kernel>(k1);
```

The following example shows how to constrain a buffer, stack, or parameter location to be on the same bank as that of another buffer, stack, or parameter. When two double buffers are co-located, this constrains the ping buffers to one bank and the pong buffers to another bank.

```
location<buffer>(port2) = location<buffer>(port1);

location<parameter>(param1) = location<buffer>(port1);
```

The following example shows how to constraint a graph to be placed within a bounding box or a joint region among multiple bounding boxes.

```
location<graph>(g1) = bounding_box(1,1,2,2);

location<graph>(g2) = { bounding_box(3,3,4,4), bounding_box(5,5,6,6) };
```

`aie_tile`

`memory_tile`

`shim_tile`

`aie_tile`

`memory_tile`

`shim_tile`

```
location(net0) = { dma_fifo(tile_type0, col0, row0, address0, size0), ss_fifo(tile_type1, col1, row1, fifo_id1), ... }
```

##### Non-Equality Function

```
void not_equal(location_constraint lhs, location_constraint rhs)
```

This function expresses `lhs` ≠ `rhs` for the two `location_constraint` parameters `lhs` and `rhs`. The function allows relative non-collocation constraint to be specified. The `not_equal` buffer constraint only works for single buffers. Do not use this constraint with double buffers.

The following example shows how to specify two kernels, `k1` and `k2`. Do not map these kernels to the same AI Engine.

```
not_equal(location<kernel>(k1), location<kernel>(k2));
```

The following example shows how to specify two buffers, `port1` and `port2`. Do not map these buffers to the same memory bank.

```
not_equal(location<buffer>(port1), location<buffer>(port2));
```

##### Stamp and Repeat Constraint

The following example shows how to constraint a graph to be placed within a bounding box. In this case, the `tx_chain0` graph is the reference and its objects are placed first and stamped to graph `tx_chain1` and `tx_chain2`. The number of rows for all identical graphs (reference plus stamp-able ones) must be the same. They must begin and end at the same parity of row. this means if the reference graph's bounding box begins at an even row and ends at an odd row, then all of the stamped graphs must follow the same convention. This limitation occurs because of the mirrored tiles in AI Engine array. In one row the AI Engine is followed by a memory group. In the next row the memory group is followed by an AI Engine within a tile.

```
location<graph>(tx_chain0) = bounding_box(0,0,3,3);
location<graph>(tx_chain1) = bounding_box(4,0,7,3);
location<graph>(tx_chain2) = bounding_box(0,4,3,7);

location<graph>(tx_chain1) = stamp(location<graph>(tx_chain0));
location<graph>(tx_chain2) = stamp(location<graph>(tx_chain0));
```
