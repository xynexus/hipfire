---
title: "constraint< std::vector<T>>"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/constraint-std-vector-T"
toc_id: ZfDG8hmkaa1qBR7xI~dI~w
content_id: pFnimJyETDK1CSw_YveDSQ
---

### constraint< std::vector<T>>

Use this template class to build vector data constraints on kernels, connections, and ports.

##### Scope

Constraint must appear inside a user graph constructor.

##### Member Function

```
constraint<std::vector<T> > operator=(std::vector<T>)
```

Constraint must appear inside a user graph constructor.

##### Constructors

The default constructor is not used. Instead the following special constructors are used with specific meaning.

```
constraint <std::vector<std::string > >& headers (kernel&)
```

This constraint allows you to specify a set of header files for a kernel that define objects to be shared with other kernels and hence have to be included once in the corresponding `main` program. The kernel source file would instead include an `extern` declaration for that object.
