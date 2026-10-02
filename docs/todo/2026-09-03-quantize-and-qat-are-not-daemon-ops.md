# The daemon schedules calibration and training, but not quantization or QAT

Status: **PARTLY RESOLVED**, 2026-09-04. Quantization and QAT are now service
ops, but as *deferred jobs* the server hands the GPU to, not as preemptible
daemon requests. Operator intent, stated directly: *"the service should provide
everything needed, including eval, calib, quantization, and QAT."*

## Where the surface stands today

`DaemonRequest` (crates/hipfire-daemon-protocol/src/lib.rs) carries the
forward/backward-pass jobs; the server's deferred queue
(crates/hipfire-server/src/deferred_jobs.rs) carries the offline GPU tools:

| op | where | status |
|---|---|---|
| KLD evaluation | daemon `kld_eval` | ✅ |
| Calibration | daemon `calibrate` | ✅ one layer per request, session parked by `run_id` |
| Drafter training | daemon `TrainDrafter` | ✅ |
| LoRA training | daemon `train_lora` | ✅ |
| Induction | job `induct` | ✅ |
| Quantization | job `quantize` | ✅ `hipfire quantize --detach …` |
| QAT / norm recovery | job `qat` | ✅ `hipfire qat --detach …` |

## What was actually wrong, and what fixed it

The original complaint was arbitration, not surface: three GPU consumers shared
one advisory lock and only one had a queue. The queue turned out to already
exist — `~/.hipfire/jobs/deferred`, drained sequentially by the server, with
`hipfire jobs {list,status,watch,cancel}`, HTTP routes and a TUI on top. What
was missing was two job kinds and, more importantly, a way for any of them to
actually get the GPU.

**The load-bearing bug: `hipfire lock run` cannot win against the daemon.** The
daemon takes its `hip-gpu-0` lease in `acquire_resource_lease_or_exit()` — before
HIP init — and holds it for the whole process lifetime. An offline job wrapped in
`hipfire lock run` therefore polls until its 1800s timeout and lands in `failed/`.
`induct --detach` was already broken this way on any machine running the server;
adding two more kinds would have shipped two more broken paths.

The fix is `is_gpu_exclusive` in `deferred_jobs.rs`: before an offline GPU job the
server unloads and stops the daemon (releasing both its VRAM and its lease), runs
the job, then restarts it and lets models reload on the next request. Handing over
the lock without the memory would only trade a lock timeout for an OOM — the
resident model has to go either way. This is the operator's documented manual
workaround made automatic and bounded, and it covers `induct` too.

## What did NOT get built, and why

- **Quantization is not an in-process daemon op.** The doc's original shape —
  schedule the encode passes, hand finished tensors to the existing writer —
  means refactoring `hipfire_quantize::cli::main()`, an 8k-line function that
  reads `std::env::args()` and calls `process::exit` throughout. The subprocess
  form gets the queueing, the arbitration, the status/cancel/logs and the
  one-at-a-time guarantee at a fraction of the risk on the repo's most
  byte-sensitive path. The container write stays offline either way, which is
  what the AGENTS.md line asks for.
- **Serving is DOWN for the duration of a GPU job**, not time-sliced with it.
  The upgrade path is a daemon that can unload, yield its lease, and reclaim it
  across a `run_id` — which needs a state machine that refuses GPU frames while
  yielded, and is worth doing on its own rather than alongside two new job kinds.
  Marked `ponytail:` at the site.
- **QAT is thin.** `hipfire-qat` (was `hipfire-train`'s `qwen35_norm_recovery`
  example, now a real binary) is qwen3.5-only and needs the teacher's residual
  captures dumped to disk by a separate env-var-driven inference run. Its
  measured win is ~0.6% perplexity, and its best-*measuring* variant deploys
  worse (see `project_light_qat_recovery`). It is reachable from the service now;
  it is not yet a thing to point an operator at.

## Still true

Non-daemon GPU binaries run outside the queue do not self-lock, so a hand-run
`hipfire-quantize` still needs `hipfire lock status` first. Going through
`--detach` is what removes that step.
