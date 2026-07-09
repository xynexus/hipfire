---
title: "Strict Aliasing Rule"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Strict-Aliasing-Rule"
toc_id: 4EXMOMeNjUV5rgczMKP2Gg
content_id: 4HGb3MHAvR3sqAX7bgpTnA
---

## Strict Aliasing Rule

The strict aliasing rule dictates that pointers are assumed not to alias if they point to fundamentally different types. `char*` and `void*` are exceptions. These can alias to any other data type. This is shown in the following graphic which shows the object universes and the associated pointers.

![xsi1593467049125.png](../assets/284-01-xsi1593467049125-png-34dbad68823a.png)

*Figure 1. Object Universes*

- **Pointers are associated with a type universe U(T):** T is the template and in the preceding graphic the various templates are shown, including an `int` universe and a `float` universe; there is also a `MyClass` universe per design. Additionally there is a `char` universe that includes all universes by default.
- **Universes do not alias:** Pointer `p` can only point to any address within the `int` universe whereas pointer `q` can only point to any address within the `float` universe. Because of this pointer `p` and pointer `q` cannot be aliased.
- **Derived pointers point to the original universe:** Pointers derived from a restrict pointer are considered restrict pointers and point to the same restricted memory region. See Derived Pointers.
- **`char*` universe contains all universes:** A `char` pointer can point to any variable in all universes.

For two pointers of the same type, as in the following, where both `p` and `q` are `int`, the compiler is conservative and aliasing is applied, resulting in loss of performance.

![wzi1593469524370.png](../assets/284-02-wzi1593469524370-png-41b96dbbcac3.png)

*Figure 2. Loss of Performance*

For two pointers of different types, as in the following example, where `p` is an `int` and `q` is `float`, the compiler applies the strict aliasing rule and an undefined behavior occurs if aliasing exists.

![xdl1593469887584.png](../assets/284-03-xdl1593469887584-png-0191f09712c4.png)

*Figure 3. Two Pointers of Different Types*
