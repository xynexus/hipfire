---
title: "Kernel Code Conversion"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Code-Conversion"
toc_id: dCf9IhS1veKhs0cqL~ZvVA
content_id: hFJR5FRLLrfOy4KKdKQmdA
---

## Kernel Code Conversion

Buffer ports do not support read, read advancing, read decrementing, write, and write advancing APIs.

Access Data From a Buffer Port illustrates multiple methods for accessing data with pointer, iterator, and vector referencing.

If the kernel algorithm uses a vector, use vector referencing with a vector iterator. This approach enables the vector processor and delivers optimal performance.

If the kernel algorithm uses a scalar data type, use iterators to simplify the code. Iterators make the code portable and safer than pointer referencing.
