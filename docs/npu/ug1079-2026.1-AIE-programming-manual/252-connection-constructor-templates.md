---
title: "Connection Constructor Templates"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Connection-Constructor-Templates"
toc_id: 4pnPl6Ba~ejfcu18pprnpw
content_id: _FhEtm5ou_TFOmWKyQvFvw
---

### Connection Constructor Templates

```
connect [name](portA, portB)
```

- `portA` can be a stream port output, a cascade output or an I/O-buffer output.
- `portB` can be a stream port input, a cascade input or an I/O-buffer input

You must connect Cascade ports together, defined as follows:

```
connect [name] (cascade out, cascade in)
```

|  | Stream Input | I/O-Buffer Input | Cascade Input |
| --- | --- | --- | --- |
| Stream Output | Yes | Yes | N/A |
| I/O-Buffer Output | Yes | Yes | N/A |
| Cascade Output | N/A | N/A | Yes |

```
connect [name](portA, portB)
```

Connects between hierarchical ports between different levels of hierarchy.

```
connect [name](parameter, portB)
```

Connects a parameter port to a kernel port.

```
connect [name](LUT, kernel)
```

Connects a LUT parameter array object to a kernel.
