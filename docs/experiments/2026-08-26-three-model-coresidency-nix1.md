# Three models co-reside on nix1 — and the daemon's ledger understates GTT by 8.5 GB

Status: measured 2026-08-26 on nix1 (`gfx1103`, Phoenix APU, RDNA3, UMA).
Ceiling: **42.0 GB GTT** (`/sys/class/drm/card0/device/mem_info_gtt_total`);
dedicated VRAM is 0.2 GB, which is why `rocm-smi` is useless here.
Companion: `2026-08-26-two-model-coresidency.md` (halo, 2 models).

## Setup

| worker | artifact | on disk | arch |
|---|---|---|---|
| `wA` | `Qwen3.6-35B-A3B--oq4.hfq` | 17.79 GB | MoE |
| `wB` | `Qwen3-4B--bf16.hfq` | 5.01 GB | qwen3 |
| `wC` | `Llama-3.2-3B-Instruct--bf16.hfq` | 3.99 GB | qwen3 |

All local NVMe (copied from `/srv` and verified byte-for-byte — NFS would
measure the network). Each was probe-loaded alone first; `wB` 14.2 s, `wC` 4.7 s.

## Result: 3 resident, and switching is still a full reload

**Dual (`wA` + `wB`):**

| | cold | switch ×2 |
|---|---|---|
| `wA` | 46.0 s | 26.8 / 27.1 s |
| `wB` | 9.0 s | 9.7 / 8.2 s |

**Triple (`wA` + `wB` + `wC`):**

| | cold | switch ×2 |
|---|---|---|
| `wA` | 29.1 s | 26.7 / 27.2 s |
| `wB` | 8.2 s | 8.4 / 8.3 s |
| `wC` | 7.3 s | 8.4 / 8.3 s |

Ledger after each cold load: **17.77 → 25.27 → 31.25 GB**, final
`resident_workers: 3`.

**The halo finding reproduces exactly, on a second machine and a second arch.**
`wB`'s cold load is 8.2 s and its "switch" is 8.3 s — *identical*. Switching by
re-issuing `load` is a full reload, because `lifecycle.rs` removes the parked
copy and `unload_model`s it before re-reading. `wA`'s 27 s switch against a
29–46 s cold load is the same story with page-cache help.

## The ledger understates real GTT by ~8.5 GB

Sampling `mem_info_gtt_used` every 2 s through the triple run:

```
   0s   0.01 GB      <- idle
  43s  38.61 GB      <- climbing through the three cold loads
  45s  16.69 GB      <- a switch: parked copy freed, then re-read
  71s  39.63 GB
  73s  31.45 GB
 123s  39.76 GB
 143s  39.79 GB      <- peak
```

| measure | value | of 42 GB |
|---|---|---|
| daemon ledger (`total_model_weight_bytes`) | 31.25 GB | 74% |
| **actual `mem_info_gtt_used` peak** | **39.79 GB** | **95%** |

The ledger is 8.5 GB low. It counts model weights; it does not count KV cache,
runtime state, scratch, or the transient buffers a load allocates while
streaming. **Three models is very near this box's ceiling — a fourth would not
fit**, which the ledger's 74% would not have told you.

The sawtooth is the reload behaviour rendered in memory: GTT drops sharply at
each switch as the parked copy is freed, then climbs back as the artifact is
re-read. A true resident swap would show a flat line.

## ⚠️ Correction: the GTT instrument does exist

`2026-08-26-m8-halo-round2-results.md` §3 originally concluded that neither
`rocm-smi` nor `resource_status` measures GTT residency and that "a third
instrument does not exist yet." **That was wrong.**
`/sys/class/drm/card0/device/mem_info_gtt_used` measures exactly this — 0.01 GB
idle, 39.79 GB at peak here — with `mem_info_gtt_total` giving the ceiling.
Every residency claim in the M8 rounds rested on the daemon ledger; this is the
instrument that can check it, and the check shows the ledger low by 8.5 GB.
That doc has been corrected.

## An artifact that is mislabelled, and a guard that catches it

`Qwen2.5-7B-Instruct--bf16.hfq` (the original second model) **cannot be loaded**:

```
refusing to load Qwen2 HFQ through the LLaMA family path: tensor
`model.layers.0.self_attn.q_proj.bias` is present, which means this is a Qwen2
(attention_bias=true) model. The LLaMA loader drops Q/K/V proj bias and would
produce wrong outputs. Current HFQ arch_id = 1. Re-quantise with
`hipfire-quantize --arch-id 7 ...`
```

The artifact is tagged `arch_id = 1` (LLaMA) but is a Qwen2 model. The loader
refuses rather than silently dropping the biases. Worth recording twice over:
the artifact on `/srv` is wrong and needs re-quantising, and this is the failure
mode that *should* be loud — a silent load would have produced plausible-looking
wrong output that no timing measurement would have caught.

## Harness notes

- **Fixed sleeps do not survive a machine change.** The first driver paced with
  55 s sleeps tuned on halo; nix1's 35B load takes ~72 s, so the next request
  went out mid-load and frames interleaved. Replaced with adaptive waits — send,
  then block for `loaded`/`error` with a deadline. That also avoids the
  no-deadline blocking read that deadlocked the halo harness for 68 minutes.
- **Verify a copy before measuring it.** The first attempt ran against a 7B that
  was 5.25 GB of 9.48 GB — `cp` was still going, and the "task completed"
  notification referred to the launcher, not the copy. Sizes are now compared
  against source before use.

## Reproducing

`nres.py` (N-model driver, adaptive waits), `probe_load.py` (per-artifact load
check), plus `dual3.json` / `tri3.json` / `tri4.json` and `gtt.log`, in this
session's scratch directory.

Sample GTT alongside any residency run:

```sh
while true; do printf '%s %s\n' "$(date +%s)" \
  "$(cat /sys/class/drm/card0/device/mem_info_gtt_used)"; sleep 2; done
```
