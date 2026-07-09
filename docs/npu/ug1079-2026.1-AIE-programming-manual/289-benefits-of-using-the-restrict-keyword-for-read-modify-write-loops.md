---
title: "Benefits of Using the Restrict Keyword for Read/Modify/Write Loops"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Benefits-of-Using-the-Restrict-Keyword-for-Read/Modify/Write-Loops"
toc_id: XVsZq6KZpRl1fiqLw4iXEQ
content_id: CRlCZ6ImY9S4zU~qaQk2uw
---

## Benefits of Using the Restrict Keyword for Read/Modify/Write Loops

The following example works without the restrict keyword, but has poor performance.

![ohc1593526603932.png](../assets/289-01-ohc1593526603932-png-14fccbb57a9e.png)

*Figure 1. Example Without Restrict Keyword*

Adding the restrict keyword allows every iteration to access a different location where there is no aliasing between iterations (`__restrict`) and aliasing within iterations preserved by data dependency. The increased parallelization results in improved performance.

![eqg1593526685102.png](../assets/289-02-eqg1593526685102-png-e65488f2d28b.png)

*Figure 2. Add Restrict Keyword*
