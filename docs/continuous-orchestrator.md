# Continuous Accelerator Orchestrator

Hipfire's continuous orchestrator coordinates heterogeneous work that shares
local accelerator and memory resources:

- token prefill and decode;
- image generation;
- training;
- maintenance and model-residency transitions.

## Ownership

`hipfire-scheduler` owns policy. `ContinuousWorkScheduler` provides priority
queues, deadline aging, conservative resource admission, cancellation, active
leases, exclusive training, and compatibility-key microbatch selection. It
does not execute work.

The server owns workload lifetime and dispatch. Daemon-backed token work uses
typed `DaemonEngine` operations for `generate_batch_prefill`,
`generate_batch_decode_step`, `reserve_session_state`, and `release_sessions`.
Image and training work enter through the durable deferred-job tree described
in [DEFERRED-JOBS.md](./DEFERRED-JOBS.md).

The daemon owns resident model/session state and the actual GPU/NPU execution.
The existing `hipfire-lock` resource leases remain the only cross-process lock
primitive. Scheduler leases are in-process accounting records, not alternate
lockfiles.

## Microbatch Contract

A workload is microbatchable only when its class supports batching and the
caller supplies an identical compatibility key. The key must encode every
property that affects one shared runtime invocation:

- model worker and accelerator device;
- architecture, quant, precision, and recurrent-state kinds;
- operation and shape;
- image sampler, schedule, step count, and geometry where applicable.

Resource requests are summed conservatively across a selected batch. Training
and maintenance work are singleton. Training is exclusive and waits for active
leases to drain.

## Current Integration State

Implemented:

- heterogeneous scheduler policy and lease accounting;
- token and image microbatch selection policy;
- typed daemon batch-prefill, decode-step, state-reservation, and session-release
  transactions;
- continuous crash-recoverable image, HTTP, and training job polling.

Still to connect:

- one server-owned payload dispatcher that feeds all request classes through
  `ContinuousWorkScheduler`;
- token streaming fan-out from batch decode results to individual HTTP clients;
- cross-request SDAPI batch assembly and result fan-out;
- cooperative training checkpoints/preemption rather than exclusive run-to-end
  commands;
- unified cancellation, deadlines, and health metrics across live and durable
  jobs.

Until those connections land, existing chat requests still use the legacy
prefill turn gate and deferred jobs execute sequentially even though the shared
policy can form compatible batches.
