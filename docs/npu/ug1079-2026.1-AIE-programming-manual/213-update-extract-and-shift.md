---
title: "Update, Extract, and Shift"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Update-Extract-and-Shift"
toc_id: aV4HC3cyppb8qBea6TEhkg
content_id: OF9UiMInRNGKxV9BWMMUTQ
---

### Update, Extract, and Shift

Use the `upd_v()`, `upd_w()`, and `upd_x()` intrinsic functions to update 128-bit (v), 256-bit (w), and 512-bit (x) portions of vector registers.

**Note:** The updates overwrite a portion of the larger vector with the new data while keeping the other part of the vector alive. This alive state of the larger vector persists through multiple updates. Keeping too many vectors alive unnecessarily can cause register spillage and reduce performance.

Similarly, you can use `ext_v()`, `ext_w()`, and `ext_x()` intrinsic functions to extract portions of the vector.

Use the `upd_elem()` and `ext_elem()` intrinsic functions to update or extract individual elements. These must be used when loading or storing values that are not in contiguous memory locations and require multiple clock cycles to load or store a vector. In the following example, the 0th element of vector `v1` is updated with the value of `a` - which is 100.

```
int a = 100;
v4int32 v1 = upd_elem(undef_v4int32(), 0, a);
```

Another important use is to move data to the scalar unit and do an inverse or sqrt. The following example, extracts the 0th element of vector `vf` and stores it in the scalar variable `f`.

```
v4float vf;
float f=ext_elem(vf,0);
float i_f=invsqrt(f);
```

You can use the `shft_elem()` intrinsic function to update a vector by inserting a new element at the beginning of a vector and shifting the other elements by one.

`ups`

`upd_hi`

`upd_lo`

```
//From v16int32 to v16acc48
v16int32 v;
v16acc48 acc = upd_lo(acc, ups(ext_w(v, 0), 0));
acc = upd_hi(acc, ups(ext_w(v, 1), 0));
```
