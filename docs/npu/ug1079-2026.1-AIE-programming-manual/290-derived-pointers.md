---
title: "Derived Pointers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Derived-Pointers"
toc_id: 1jiTPwytXTAhKLOWPvivww
content_id: iklp_Dj7Ijzbdt6PkausOw
---

## Derived Pointers

Pointers derived from a restrict pointer are considered restrict pointers and point to the same restricted memory region, as shown in the following example, where `rq2`, derived from `rq1` (defined as a restrict pointer) is also a restrict pointer and points to the same universe.

![bkh1593526856763.png](../assets/290-01-bkh1593526856763-png-3b9fe5361c58.png)

*Figure 1. Pointers to Same Restricted Memory Region*
