# Deferred Jobs

Hipfire continuously executes local deferred jobs while `hipfire serve` is
running. This is intended for heavy image, HTTP, and training work that should
survive client disconnects and server restarts.

The startup runner scans:

```text
~/.hipfire/jobs/deferred/queued/*.json
~/.hipfire/jobs/deferred/running/*.json
```

Jobs are claimed by moving them into `running/`, executed sequentially, then
moved into `done/` or `failed/`. Each terminal state also gets a
`*.result.json` record. The runner keeps polling for newly queued files after
the startup drain. Existing `running/*.json` files are retried on startup,
which handles a previous server crash.

Set `HIPFIRE_DEFERRED_JOBS=0` to disable the runner. Set
`HIPFIRE_DEFERRED_POLL_MS` to control the polling interval (default 250 ms,
clamped to 10 ms through 60 seconds). Command jobs can be disabled separately
with `HIPFIRE_DEFERRED_ALLOW_COMMANDS=0`.

## Image Generation

`sdapi_txt2img` and `sdapi_img2img` use the same request bodies as the matching
SDAPI routes.

```json
{
  "id": "daily-image",
  "kind": "sdapi_txt2img",
  "body": {
    "model": "Qwen-Image",
    "prompt": "a compact workbench with a small AMD inference rig",
    "steps": 20,
    "width": 1024,
    "height": 1024,
    "save_images": true,
    "send_images": false
  }
}
```
## HTTP Jobs

`http_post` supports these in-process endpoints:

- `/sdapi/v1/txt2img`
- `/sdapi/v1/img2img`
- `/v1/chat/completions`
- `/v1/responses`

Streaming is forced off for deferred jobs.

```json
{
  "id": "batch-summary",
  "kind": "http_post",
  "endpoint": "/v1/responses",
  "body": {
    "model": "Qwen3.5-8B--mq4.hfq",
    "input": "Summarize yesterday's local benchmark notes."
  }
}
```

## Training Commands

`training_command` launches a no-shell argv vector and writes status into the
existing admin training-run tree under `~/.hipfire/training/runs/<run_id>/`.
Stdout and stderr are written under `~/.hipfire/jobs/deferred/logs/`.

```json
{
  "id": "pflash-ssm-train",
  "kind": "training_command",
  "run_id": "pflash-ssm-train",
  "cwd": "/home/sadara/hipfire",
  "env": {
    "HIPFIRE_PFLASH_DAEMON_LABELS": "/home/sadara/.hipfire/training/labels/pflash.jsonl"
  },
  "argv": [
    "cargo",
    "run",
    "-p",
    "hipfire-train",
    "--release",
    "--example",
    "ssm_drafter_train"
  ]
}
```
