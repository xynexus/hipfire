---
title: "Loop Flattening and Unrolling"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Loop-Flattening-and-Unrolling"
toc_id: 5GPlsWNvvUGJq4Y7XPQ5Hw
content_id: 4b0W1COrUsa2go3rV~UK4w
---

### Loop Flattening and Unrolling

Loops can be flattened completely with the `chess_flatten_loop` pragma. Flattening can be useful for small loops that are not optimally automated by the AI Engine compiler.

`chess_loop_count`

```
for(int i=0;i<6;i++) chess_flatten_loop {...}
for(...) chess_loop_count(6) chess_flatten_loop {...}
```

With `chess_unroll_loop(N)`, the loop body can be duplicated `N-1` times, and the loop count is divided by `N`. The loop can also be completely unrolled by `chess_unroll_loop(*)`. The loop is unrolled and rewritten as a repeated sequence of similar independent statements.

`chess_unroll_loop(N)` creates an additional preamble loop. When the loop count is known at compile time, this preamble loop is fully unrolled. However, if the loop bound is not a compile-time constant but is guaranteed to be a multiple of `N`, use `chess_unroll_loop_assuming_multiple(N)` instead. This prevents the extra preamble loop, reducing program memory usage.

Loop flattening occurs in the final scheduling phase such that code generation still uses the loop construct. Unlike loop flattening, loop unrolling duplicates iterations of code, and the duplicated codes can be compiled differently. Unrolling can be used to improve software pipelining of loops, but it can also place a burden on scheduling when the unrolled loop count is large.
