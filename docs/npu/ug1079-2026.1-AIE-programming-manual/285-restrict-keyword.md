---
title: "Restrict Keyword"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Restrict-Keyword"
toc_id: 5wuPW4qgK9rcdIahefZTpQ
content_id: eP35ggZq5_NyKeLLcStgEA
---

## Restrict Keyword

The restrict keyword is mainly used in pointer declarations as a type qualifier for pointers. It does not add any new functionality. It allows you to tell the compiler about a potential optimization. Using `__restrict` with a pointer informs the compiler that the pointer is the only way to access the object pointed at, and the compiler does not need to perform any additional checks.

**Note:** If a programmer uses the restrict keyword and violates the above condition, undefined behavior can occur.

The following is another example with pointers that, by default, have no aliasing.

![mqj1593524314947.png](../assets/285-01-mqj1593524314947-png-3fc5e3b7f326.png)

*Figure 1. No Aliasing Example*

Apply the restrict keyword for performance improvement. The following example shows no memory dependencies with other pointers.

![vqk1593524411519.png](../assets/285-02-vqk1593524411519-png-86909dfaaba5.png)

*Figure 2. No Memory Dependencies with Other Pointers*
