# The daemon schedules calibration and training, but not quantization or QAT

Status: **OPEN**, 2026-09-03. Operator intent, stated directly: *"the service
should provide everything needed, including eval, calib, quantization, and QAT."*

## Where the surface stands today

`DaemonRequest` (crates/hipfire-daemon-protocol/src/lib.rs) already carries the
forward/backward-pass jobs:

| op | wire tag | status |
|---|---|---|
| KLD evaluation | `kld_eval` | ✅ |
| Calibration | `calibrate` | ✅ one layer per request, session parked by `run_id` |
| Drafter training | — | ✅ `TrainDrafter` |
| LoRA training | `train_lora` | ✅ |
| **Quantization** | — | ❌ **absent** |
| **QAT / norm recovery** | — | ❌ **absent** |

Quantization runs as the standalone `hipfire-quantize` binary. QAT (block-local
RMSNorm recovery) is a `hipfire-train` *example*, `qwen35_norm_recovery`.

## Why the gap has teeth

Both are GPU binaries that do not go through the scheduler, so they contend for
`hip-gpu-0` with the daemon rather than queueing behind it. Concretely, during
the scale study on 2026-09-03:

- The daemon **refuses** rather than waits when the lock is held
  (`FATAL: ... is locked by <pid> hipfire-coexistence calibrate`), so a scoring
  run launched next to a calibration silently produced no rows for 7 of 8
  artifacts — the header printed, the KLD lines did not.
- `hipfire.service` grabs the lock at boot, so every offline build on a machine
  running the server fails until the service is stopped by hand.
- Two independently launched pipelines raced on the same output artifact,
  because nothing arbitrates "one quantize at a time" the way the scheduler
  would.

None of these are bugs in the tools. They are what happens when three GPU
consumers share one advisory lock and only one of them has a queue.

## What the AGENTS.md invariant says to do

> if it runs kernels over model weights it may be scheduled by the daemon; if it
> rewrites bytes between container formats it belongs in `hipfire-coexistence`.

Quantization is **both**, and the split falls naturally:

- **Daemon-shaped (schedulable):** the Hessian/imatrix passes, AWQ scale search,
  LDLQ/OBS error propagation, the QTIP trellis encoder — all kernels over
  weights, all preemptible against serving traffic, all already sharing the
  runtime the daemon owns.
- **Stays offline:** the container write itself (`hfq_out`), index/offset
  bookkeeping, spill, the lossless bf16 recode. Byte translation, not GPU work.

So the shape is a `Quantize` op that schedules the encode passes and hands the
finished tensors to the existing writer — not a second container writer inside
the daemon.

QAT is the easier of the two: it is a forward+backward over captured residuals
plus an AdamW step, which is exactly what `TrainLora` already does. The pieces
(`hipfire_train::qtip_quant` dequant, the recovery loop, `--norm-patch` fold)
exist; what is missing is the request type and a session parked by `run_id` the
way `Calibrate` parks one.

## Interim, until that lands

`hipfire lock status` before every offline GPU step, and stop `hipfire.service`
for the duration of a build campaign. Both are workarounds for the missing queue.
