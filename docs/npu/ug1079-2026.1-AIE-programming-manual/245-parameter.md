---
title: "parameter"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/parameter"
toc_id: NS41xs4ulO8KJwlKIG7YkQ
content_id: B_FhagwYGtuspSdHIRLUYA
---

### parameter

The `parameter` class contains two static member functions to allow you to associate globally declared variables with kernels.

##### Member Functions

```
static parameter & array(X)
```

Wrap around any extern declaration of an array to capture the size and type of that array variable.

```
static parameter & scalar(Y)
```

Wrap around any extern declaration of a scalar value (including user defined structs).
