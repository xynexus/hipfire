---
title: "Restrict Keyword"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Restrict-Keyword"
toc_id: ~sAolyRNqrfXoJIeMsm69A
content_id: I0WcdukiiZVUqAnhtgbvnw
---

## Restrict Keyword

The C++ standard provides the compiler extension `__restrict` for the C `restrict` pointer qualifier. It enables more aggressive optimization by stating that pointer aliasing does not cause memory dependencies. The compiler, by default, does not distinguish between different accesses of the same array. Thus, if an array is accessed in the pipeline, the compiler assumes pointers can reference the same location, increasing the interval between loops.

This makes it is essential in some situations to use a `__restrict` keyword to help guide the tool to achieve better performance. If a pointer is created with the `restrict` keyword, it is treated as a new object by the compiler. Pointers with the `restrict` keyword that point to the same location are treated independently by the compiler. The compiler can schedule the pointer access independently, which can impact the order of updates and cause undefined behavior. For detailed information about the concept of the `__restrict` keyword, see Using the Restrict Keyword in AI Engine Kernels.
