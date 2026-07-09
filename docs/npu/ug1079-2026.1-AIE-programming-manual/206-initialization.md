---
title: "Initialization"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Initialization"
toc_id: YweYKfj5GcFok_dnQLJniQ
content_id: NL3xC509r27QrctZppdPqw
---

### Initialization

The following functions can be used to initialize vector registers as undefined, all 0’s, with data from local memory, or with part of the values set from another register and the remaining part are undefined. Initialization using the `undef_type()` initializer ensures that the compiler can optimize regardless of the undefined parts of the value.

```
v8int32 v;
v8int32 uv = undef_v8int32(); //undefined
v8int32 nv = null_v8int32(); //all 0's
v8int32 iv = *(v8int32 *) reals; //re-interpret "reals" as "v8int32" pointer and load value from it
v16int32 sv = xset_w(0, iv); //create a new 512-bit vector with lower 256-bit set with "iv"
```

In the previous example, vector set intrinsic functions `[T]set_[R]` allow creating a vector where only one part is initialized and the other parts are undefined. Here:

- [T] indicates the target vector register to be set, w for W register (256-bit), x for X register (512-bit), and y for Y register (1024-bit)
- [R] indicates where the source value comes from, v for V register (128-bit), w for W register (256-bit), and x for X register (512-bit)

**Note:** [R] width is smaller than [T] width.

The valid vector set intrinsic functions are, `wset_v`, `xset_v`, `xset_w`, `yset_v`, `yset_w`, and `yset_x`.

The `static` keyword applies to the vector data type as well. If not initialized, the default value is zero and persists across graph run iterations.
