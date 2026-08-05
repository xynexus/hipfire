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
| hipfire default (exact bf16 head) | **1020.1 MB** | **+32%** |
| hipfire, `HIPFIRE_LMHEAD_TWOSTAGE=q4` | **~626.4 MB** | **-19%** |

The rescore adds little: top-k rows of bf16 at 4 KB each is well under 1 MB.

### UNVERIFIED — the two-stage path did not activate when tested

The table above is derived from `hipfire inspect` and the env documentation. An
attempt to confirm it on hardware did not, and the numbers should be treated as
unconfirmed until it does.

What was tried, and what it showed:

* **Throughput A/B** — `hipfire bench` with and without
  `HIPFIRE_LMHEAD_TWOSTAGE=q4`: 76.07 vs 76.15 tok/s. A 51% cut in per-token
  traffic should not be invisible.
* **Output A/B** — `q4:1` (top-1 shortlist) produced byte-identical text. This
  turned out to be a WORTHLESS control: `llama.rs:60` documents the path as
  "greedy-exact (recall@1 = 1.0 at K=32)", so identical argmax output is the
  design, not evidence of anything.
* **TTFT probe** — the decisive one. `lmhead_project` builds the coarse tier on
  first use via `build_lmhead_coarse_bf16`, quantising the 525 MB bf16 head.
  That cannot be free. Measured TTFT: **59.0 ms unset, 60.2 ms at q4, 56.6 ms at
  q2** — flat. The coarse build is not running, so the two-stage path is **not
  activating** on this artifact.

The gate is `w.gpu_dtype == DType::BF16` (`llama.rs:66`). Why it is false here
is not established. The open possibility that matters is that the runtime may
already serve the projection from the stored `model.embed_tokens.coarse.weight`
(CoarseQ4Row, 131.59 MB) rather than the bf16 tier — in which case the DEFAULT
per-token figure is ~626 MB, not 1020 MB, and the "+32% vs FLM" conclusion is
wrong in the model's favour.

Both rows below therefore rest on an unconfirmed premise. Settling it needs the
runtime to report which tensor backs the head, which no current log emits.

## Decode ceilings at 55.5 GB/s

| configuration | per-token | ceiling |
|---|---|---|
| hipfire, two-stage lm_head | 626.4 MB | **88.6 tok/s** |
| FLM streamed set | 772.3 MB | 71.9 tok/s |
| hipfire default | 1020.1 MB | 54.4 tok/s |
| FLM at its own measured 46.2 GB/s | 772.3 MB | 59.8 tok/s |

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
