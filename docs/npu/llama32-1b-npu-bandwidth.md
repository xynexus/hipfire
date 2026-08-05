# llama3.2:1b — NPU bandwidth, FLM vs hipfire

Same method as `chatglm3-npu-bandwidth.md`: storage read rate with per-file
page-cache eviction, then the NPU weight-streaming ceiling applied to each
artifact's **per-token** footprint.

## Storage (all copied to local btrfs first)

`/srv` is NFS4 from `carbon`, and reading the hipfire artifact there gives
0.12 GB/s cold — 35x slower than FLM's, which lives on local disk. That is the
network, not the artifact. Measured fairly, both local:

| file | size | cold GB/s | warm GB/s |
|---|---|---|---|
| FLM `model.q4nx` | 1.298 G | 5.40 | 23.12 |
| hipfire `Llama-3.2-1B-Instruct--oq4++.hfq` | 0.986 G | 7.18 | 28.90 |
| chatglm3 token set (reference) | 3.231 G | 6.34 | 17.15 |

## Per-token streamed bytes — where the two formats differ

The NPU streaming ceiling (55.5 GB/s, `chatglm3-npu-bandwidth.md`) is a machine
property. What differs per model is how many bytes cross it per token.

**FLM**: 772.3 MB of the 1.298 GB file, documented and manifest-checked in
`flm-layer-dataflow.md` — 113 I8 tensors.

**hipfire oq4++**, from `hipfire inspect`:

| tier | tensors | size | on the per-token path? |
|---|---|---|---|
| `Oq4G256` layer weights | 112 | 494.14 MB | yes, all of it |
| `model.embed_tokens.weight` BF16 | 1 | 525.34 MB | **depends — see below** |
| `model.embed_tokens.coarse.weight` CoarseQ4Row | 1 | 131.59 MB | only with two-stage lm_head |
| `F16` scales | 112 | 0.66 MB | yes |

The embedding is read one 4 KB row per token for the *lookup*. The cost is the
*output projection*, and which tier serves it is a runtime switch:
`HIPFIRE_LMHEAD_TWOSTAGE` selects a coarse-Q4 shortlist then a bf16 rescore of
the top-k rows. **Unset — the default — "the exact full-precision gemv runs"**,
i.e. the full BF16 525 MB matrix, every token.

| configuration | per-token bytes | vs FLM's 772.3 MB |
|---|---|---|
| hipfire as measured (F32 head) | 1545.5 MB | +100% |
| head kept BF16 | 1020.1 MB | +32% |
| **head BF16L3, packed — current default** | **871.1 MB** | **-13%** |

The rescore adds little: top-k rows of bf16 at 4 KB each is well under 1 MB.

### RESOLVED — it was worse than this, and is now better

An earlier revision of this document flagged the figures above as unverified,
because a throughput A/B of `HIPFIRE_LMHEAD_TWOSTAGE` showed no difference and
a diagnostic added to the two-stage branch never fired. Both were real, and the
cause was found:

**The tied lm_head was expanded to F32 at load.** `hfq.rs` dequantised a BF16
embedding to f32 for the tied output projection, so the head sat in VRAM at
1050.7 MB rather than 525.3, and `lmhead_project`'s `w.gpu_dtype == DType::BF16`
gate could never match — which is why the two-stage path was unreachable. Two
independent observations, not inference: the artifact matches `lm_head` zero
times (so the tied branch runs), and that branch sets `gpu_dtype: DType::F32`
unconditionally at both sites.

So the starting point was worse than reported: **1545.5 MB per token, exactly
2.00x FLM**, not 1020.1.

Fixed in `hipfire` across three steps, each measured with byte-identical output:

| step | head | per-token | tg128 |
|---|---|---|---|
| as found | F32, 1050.7 MB | 1545.5 MB | 76.07 |
| stop widening | BF16, 525.3 MB | 1020.1 MB | 89.95 |
| **BF16L3, packed** | **379.8 MB** | **871.1 MB** | **102.41** |

That needed a kernel that did not exist — `gemv_bf16_xf32`, then
`gemv_bf16l3_xf32` (bf16/BF16L3 weight against an f32 activation, f32
accumulate) — plus a dispatch-family entry and decode arms in every arch loader.
LUT3 heads are now resident by default, and the quantizer already steers
gather-shaped tensors to LUT3, so stock artifacts get it with no flag.

The two-stage lm_head shortlist is a separate, still-unused path; it is not what
delivered this.

## Decode ceilings at 55.5 GB/s

| configuration | per-token | ceiling |
|---|---|---|
| **hipfire now (BF16L3 head)** | 871.1 MB | **63.7 tok/s** |
| FLM streamed set | 772.3 MB | 71.9 tok/s |
| FLM at its own measured 46.2 GB/s | 772.3 MB | 59.8 tok/s |
| hipfire as found (F32 head) | 1545.5 MB | 35.9 tok/s |

hipfire moves 13% fewer bytes than FLM but is still behind FLM's ceiling,
because FLM's 772.3 MB excludes tensors hipfire streams every token. The
ordering that matters: hipfire went from 0.50x FLM's ceiling to 0.89x, on bytes
alone.

**None of this is collectible on the NPU yet.** These are GPU measurements and a
bytes argument; hipfire's NPU decode delivers ~10 GB/s effective against the
55.5 available, so it is dispatch-bound, not bandwidth-bound. Fewer bytes is
necessary and not sufficient.

## What this says about the goal

On a 1B model the lm_head is not a rounding error — it is 13% of the parameters
but, at bf16 against 4-bit layers, **51% of the default per-token traffic**.
The two-stage path is the difference between giving away 32% of the bandwidth
budget to FLM and taking back 19%, and it is off unless asked for.

This is a traffic argument, not an end-to-end result. hipfire's decode path
delivers ~10 GB/s effective against this same fabric
(`decoder-layer-npu-scope.md`), so dispatch structure, not bytes, is the
binding constraint today — the format advantage above is headroom that cannot
be collected until that is fixed.
