---
title: "Inline Keywords"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Inline-Keywords"
toc_id: soEugpzQ46Ch2MQGJ63H2Q
content_id: kfDmW15j0bENg9xwfVqUfQ
---

## Inline Keywords

The AI Engine compiler supports the `always_inline` function attribute to eliminate function call overhead. An `inline` function is a function that expands in the line in which it is called. For larger functions, the overhead of the function call is insignificant compared to the amount of time the function takes to run. However, for smaller functions, the time needed to make the function call is greater than the time needed to execute the function’s code. This overhead occurs for small functions because the execution time of a small function is less than the switching time.

One advantage of using `inline` functions is that there is no function call overhead. Another potential advantage is that the compiler can perform context-specific optimizations which are not possible when the function stands in its own. Some drawbacks of using `inline` functions are as follows:

- Larger register consumption
- Program size increase
- The function cannot be profiled because there are no clear start and stop program counters to check.

The following keywords indicate the usage of the inlining functions to the compiler:

- `inline`: This keyword is an indicator that the AI Engine compiler is free to use a inline substitution or function call for any function marked inline.
- `inline __attribute__((always_inline))`or `__attribute__((always_inline))`: This forces the compiler to inline the function.
- `inline __attribute__((noinline))`or `__attribute__((noinline))`: This forces the compiler to always generate a function call.
