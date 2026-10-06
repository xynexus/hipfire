# Move the daemon-socket CLI tools to the HTTP API

Status: done 2026-10-07 (see "Done" at the end). Decided 2026-10-06: these tools are deprecated in their current form.

## Why

`hipfire serve` spawns its inference worker with `--listen`, so the worker serves
`~/.hipfire/daemon.sock` beside its stdio owner (serve). Three tools attach to that
socket and speak the daemon protocol directly:

| tool | where |
|---|---|
| `hipfire chat` | `crates/hipfire-cli/src/commands/chat.rs` |
| `hipfire bench` | `crates/hipfire-cli/src/commands/bench.rs` |
| eval daemon executor | `crates/hipfire-eval/src/executor_daemon.rs` |

Their requests name no worker, so they acted on whichever model was active: the
running swarm's. A `bench` reset wiped the swarm's in-flight batch sessions, and an
attached load cleared every steer and left a second copy of the model resident
(review finding #5).

Since the change that added this file, a shared worker **refuses** a socket client's
request that names no `worker_key_id` or `model`. Read-only kinds are exempt: ping,
inventory, model_registry, and the status and trace queries. See
`refuse_unscoped` in `crates/hipfire-daemon/src/lib.rs`. The tools therefore fail
while serve runs, and still work against a worker with no serve beside it.

## To do

- `hipfire chat`: send `/v1/chat/completions` (streaming) to the running serve.
  Spawn a private worker only when no serve is up.
- `hipfire bench`: decide what it measures through serve.
  - Throughput through the HTTP routes, which is what users get.
  - Or an admin-gated bench route on serve that runs `bench_prefill` on a named
    worker, sequenced with the batch runner rather than racing it.
- eval daemon executor: use the HTTP API, or name the worker it loaded (its own
  `worker_key_id`) on every request.
- Once all three are ported, delete their socket paths. `refuse_unscoped` stays, as
  the guard for any client that bypasses serve.

## Not a fix

Do not relax `refuse_unscoped` to get a tool working again. The refusal is the
protection; the port is the fix.

## Done (2026-10-07)

None of the three attaches to the shared socket any more. The adapter's
client side of it (`connect`, `attach_or_spawn`, `shared_daemon_listening`,
`SocketTransport`) is gone. The worker still listens beside serve, and
`refuse_unscoped` still guards that door.

- **`hipfire chat`.** With serve up, it streams `/v1/chat/completions`.
  Otherwise it spawns a private worker.
- **`hipfire bench`.**
  - With serve up, it measures through `/v1/chat/completions`, streamed: a
    streamed request reports full timings and runs on its own.
  - A per-sample nonce at the head of the prompt replaces the
    `/admin/runtime/reset` between samples, which wiped every other client's
    sessions.
  - Otherwise it spawns a private worker, which keeps the exact `bench_prefill`.
- **`hipfire eval`.**
  - With serve healthy, the smoke and speed batteries go through it, streamed.
  - The smoke row that reset the server between turns now checks that the same
    greedy request answers the same with another in between.
  - The batteries that drive a worker directly (quality/KLD, cask, profile,
    vision) spawn a private one. While serve holds the worker that fails on the
    `daemon.pid` flock and is reported as a failed row.

Still open: on the batched path, a non-streamed request's `timings` carry
token counts only. The batch runner's `DoneEvent` sets the rates and TTFT to
`None`. The tools stream to avoid it; a client that does not stream gets no
throughput numbers.
