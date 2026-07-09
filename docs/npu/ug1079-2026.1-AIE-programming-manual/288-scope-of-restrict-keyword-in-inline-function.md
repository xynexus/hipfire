---
title: "Scope of Restrict Keyword in Inline Function"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Scope-of-Restrict-Keyword-in-Inline-Function"
toc_id: 5I9IDw6~CPTuh1XoXE6N~Q
content_id: xcx~Xs_DmiOppjkmhhKvPQ
---

## Scope of Restrict Keyword in Inline Function

When there are no other accesses within the scope, declaring the restrict pointer has no performance benefits.

![hxa1593525967493.png](../assets/288-01-hxa1593525967493-png-635dba81d0a3.png)

*Figure 1. Working Example with No Performance Benefits*

In a special case, you can have non-aliasing accesses, as in the following example. Here the parent pointer, `p`, is used but points to a different location and therefore this is acceptable.

![ftc1593526496224.png](../assets/288-02-ftc1593526496224-png-8cdc874bed11.png)

*Figure 2. Special Case—Non-aliasing Accesses*
