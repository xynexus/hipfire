---
title: "Restrict Qualification"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Restrict-Qualification"
toc_id: IhbTYFBhFR5RurWWnYWHRg
content_id: 0IzKqGuxNBowHagn7IhcuQ
---

## Restrict Qualification

The C standard provides a specific pointer qualifier, `__restrict`. This allows more aggressive compiler optimization by explicitly stating data independence between whatever the pointer references and all other variables. For example:

```
int a; // global variable
void foo(int* __restrict p, int* q)
{
  for (...) { ... *p += a + *q; ...}
}
```

Now the analysis of `foo` can proceed with the knowledge that `*p` does not denote the same object as `*q` and `a`. So, `a` and `*q` can now be loaded once, before the loop.

Currently, the compiler front end does not disambiguate between different accesses to the same array. So, when updating one element of an array, it assumes that the complete array has changed value. The `__restrict` qualifier can be used to override this conservative assumption. This is useful when you want to obtain multiple independent pointers to the same array.

```
void foo(int A[])
{
  int* __restrict rA = A; // force independent access
  for (int i = ...)
  rA[i] = ... A[i];
}
```

In this example, the `__restrict` qualifier allows software pipelining of the loop; the next array element can already be loaded, while the previous one must still be stored. To maximize the impact of the `__restrict` qualifier, the compiler front end, by default, inserts a `chess_copy` operation in the initializer, as if was written:

```
int* __restrict rA = chess_copy(A);
```

This is needed to keep both pointers distinct within the optimizer (for example, no common subexpression elimination). This behavior can be disabled for the AI Engine compiler by means of the option `-mllvm -chess-implicit-chess_copy=false`. So, the `chess_copy` creates two pointers, while `__restrict` informs the compiler not to consider any mutual dependencies between the stores/loads through these pointers. For `__restrict` pointers having a local scope, the mutual independence assumption only holds during the lifetime of the `__restrict` pointer.

Pointers derived from a `__restrict` pointer (such as `rA+1` or through pointer intrinsics) keep the restriction, that is, they are considered to point to the same restricted memory region.

**Note:** `chess_copy`

ASIP Programmer Chess Compiler User Manual

AI Engine Lounge
