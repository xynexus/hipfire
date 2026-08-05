# The stored `<embed>.coarse.weight` tier is never read

`hipfire-quantize` emits a coarse-Q4 lm_head shortlist tier by default —
`<embed>.coarse.weight`, disabled with `--no-coarse-lmhead` /
`HIPFIRE_NO_COARSE_LMHEAD`. **No runtime code loads it.**

Every occurrence of `coarse` outside the quantizer builds the tier at run time
from the bf16 head instead:

| site | what it does |
|---|---|
| `hipfire-quantize/src/main.rs:468-492` | writes `<stem>.coarse.weight` |
| `runtime/src/llama.rs:71` | `build_lmhead_coarse_bf16(gpu, &w.buf, ...)` — recomputes it |
| `runtime/examples/verify_lmhead_twostage_real.rs:71` | same, in the check |

Nothing resolves the tensor by name. Searched: `grep -rn "coarse" --include=*.rs
crates/` returns only the quantizer, the env docs, and the two `build_*` call
sites.

## Cost

On `Llama-3.2-1B-Instruct--oq4++.hfq` (`hipfire inspect`):

| tier | tensors | size |
|---|---|---|
| `Oq4G256` layer weights | 112 | 494.14 MB |
| `model.embed_tokens.weight` BF16 | 1 | 525.34 MB |
| **`model.embed_tokens.coarse.weight` CoarseQ4Row** | **1** | **131.59 MB** |
| `F16` scales | 112 | 0.66 MB |

131.59 MB of a 986 MB artifact — **13%** — is a tier nothing reads. It is paid
on disk, over the wire, and in load time. On a small model that is the single
largest avoidable line item.

## Two things to decide

1. **Load it instead of rebuilding it.** The two-stage path currently quantises
   the 525 MB bf16 head on first use. The quantizer already did that work
   offline; wiring `lmhead_project` to the stored tensor when present removes a
   first-token stall and makes the emitted tier meaningful.
2. **Or stop emitting it by default.** If the runtime is going to rebuild
   regardless, the default should be `--no-coarse-lmhead` and every artifact
   built so far is carrying dead weight.

Doing neither is the only option that is clearly wrong, and it is the current
default.

## How this surfaced

While trying to confirm a per-token bandwidth claim for llama3.2:1b
(`hipfire-npu` `docs/npu/llama32-1b-npu-bandwidth.md`), the two-stage path would
not activate: TTFT is flat at 59.0 / 60.2 / 56.6 ms for unset / `q4` / `q2`,
where a first-use quantisation of a 525 MB matrix cannot be free. That is a
separate open question — the gate is `w.gpu_dtype == DType::BF16`
(`llama.rs:66`) and why it is false there is not yet established. Chasing it is
what turned up the unread tier.
