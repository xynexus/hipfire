---
title: "Alignment"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Alignment"
toc_id: XW9rSe7vCB2gvJ9MDjLaBw
content_id: zEtNqXUOB7A3LgctRUcC4Q
---

### Alignment

Applications can load from data memory (DM) into vector registers and store the contents of vector registers into DM. Memory instructions in the AI Engine that operate on vectors have alignment requirements. Therefore, functions are available for both aligned and unaligned accesses.

The following functions are assumed to operate on pointers that meet the alignment requirements for a vector load or store of the size:

- `aie::load_v`
- `aie::store_v`

**Note:** AI Engine

The following functions are assumed to operate on pointers only align to the element of the vector:

- `aie::load_unaligned_v`
- `aie::store_unaligned_v`

For optimal performance, vector load and store must operate on memory that has met the vector operation alignment requirement. Unaligned accesses can incur additional overhead depending on the amount of misalignment.

**Note:** Kernel buffer interfaces ensure that internal buffers have the required alignment for vector loads.

You can use the `alignas` standard C specifier to ensure proper alignment of local memory. In the following example, `reals` aligns to a 16-byte boundary.

```
// align to 16 bytes boundary
// equivalent to "alignas(aie::vector<int16,8>)"
alignas(16) const int16 reals[8] =
       {32767, 23170, 0, -23170, -32768, -23170, 0, 23170};
```

The API has another way to specify vector alignment on a specific vector type, for example:

```
alignas(aie::vector_ldst_align_v<int16, 8>) const int16 reals[8] =
       {32767, 23170, 0, -23170, -32768, -23170, 0, 23170};
```

The API provides a global constant value (`aie::vector_decl_align`) that you can use to align the buffer to a boundary that works for any vector size.

```
alignas(aie::vector_decl_align) static cint16 my_buffer[8]={{0,0},{1,-1},{2,-2},{3,-3},{4,-4},{5,-5},{6,-6},{7,-7}};
```

**Note:** AI Engine

AMD

`aie::vector_decl_align`
