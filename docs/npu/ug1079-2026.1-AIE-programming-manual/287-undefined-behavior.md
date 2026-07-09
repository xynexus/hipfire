---
title: "Undefined Behavior"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Undefined-Behavior"
toc_id: XutfMvFQop~OkHH_nns3cg
content_id: jmbBYzaMbK7XEAAuc5Yifg
---

## Undefined Behavior

Using the restrict keyword improves performance as shown in the previous topic. However, inappropriate use causes issues. The `__restrict` child pointers must be used in a different block-level scope than the parent pointers, such as pointer `p` and `q` as shown in the following example.

#### Working Example 1

![nae1593524553845.png](../assets/287-01-nae1593524553845-png-ea0350f37396.png)

*Figure 1. Use of Restrict Keyword*

Use of parent pointers in the same scope might break the `__restrict` contract which produces an undefined behavior, such as pointers `p` and `q` in the following example.

![bqa1593524691697.png](../assets/287-02-bqa1593524691697-png-5f1d67b7593f.png)

*Figure 2. Undefined Behavior*

#### Working Example 2

This can also happen during the `load` operation, as shown in the green text (`return *p;`) in the following figure.

![lvf1593525000429.png](../assets/287-03-lvf1593525000429-png-347005b9c172.png)

*Figure 3. Load Operation*

The undefined behavior occurs when the restrict pointers are used within the same scope, such as pointers `p` and `q` in the following example.

![rfd1593525444617.png](../assets/287-04-rfd1593525444617-png-028f32e26ae5.png)

*Figure 4. Restrict Pointers in Same Scope*

#### Working Example with Inline Function

The following code shows the working inline function call, in which pointer `p` and pointer `q` are used in different scopes.

![rbx1593525551749.png](../assets/287-05-rbx1593525551749-png-30fd5dedd6bd.png)

*Figure 5. Inline Function Calls*

Undefined behavior can occur when you use `restrict` pointers in the same scope. For example, pointers `p` and `q` in the following code cause this issue.

![trw1593525821152.png](../assets/287-06-trw1593525821152-png-15ea6bb23bc2.png)

*Figure 6. Inline Function Calls in Same Scope*
