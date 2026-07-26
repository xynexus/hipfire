---
title: "Alignment"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Alignment"
toc_id: pdaU4He_9RR9DaT7MlIb4A
content_id: cYun7pjvujMtDt~_S4Ul5Q
---

### Alignment

Use the `alignas` standard C specifier to ensure proper alignment of local memory. In the following example, the `reals` aligns to a 16 byte boundary.

```
alignas(16) const int32 reals[8] =
       {32767, 23170, 0, -23170, -32768, -23170, 0, 23170};
       //align to 16 bytes boundary, equivalent to "alignas(v4int32)"
```
