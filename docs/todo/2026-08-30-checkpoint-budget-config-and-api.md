# TODO — `resident_checkpoint_max` as real config, and readable by clients

Status: **not started.** Small. Partly built already — see "What exists" before
writing anything.

## Why

A client constructing prompts as layered prefixes (`prefix1`, `prefix1+prefix2`,
`prefix1+prefix3`) needs to know how many reuse boundaries it can afford. Each
boundary costs a checkpoint, because Qwen3.5/3.6 is 48 of 64 layers linear
attention whose recurrent state is not positionally truncatable — reuse across a
boundary requires a checkpoint captured *at* it. So `resident_checkpoint_max` is
not an internal tuning knob; it is the **budget that decides a client's prompt
structure**. Corrode's `docs/harness-architecture.md` §3.2 layers context
project → turn → role against exactly this number, and today has to hardcode 4.

## What exists

- The value is already env-configurable: `HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS`
  (or legacy `HIPFIRE_SERVER_PREFILL_STATE_CACHE_MAX`), default 4, clamped
  `[0, 64]` — `hipfire-scheduler/src/lib.rs:1207`.
- It is already reported on `/health` under `state_cache`
  (`server_state_cache_health_json`, `lib.rs:1334`).

## What's missing

1. **It is not a config key.** `resident_checkpoint_max` and `state_cache` appear
   zero times in `docs/config-schema.md`, and the env var is absent from
   `docs/env-vars.md` — it is an undocumented env-only knob. Promote it to a
   first-class key with `global`/`model`/`runtime` scope like the `ngram_spec*`
   family, so it can be set in `config.json` and per model. A big-context project
   and a chat workload want different budgets, and model residency already varies
   per model.

2. **Clients can't read it.** `/health` is an operator endpoint. A client sizing
   its prefix layers needs the checkpoint budget as an advertised **capability** —
   `/v1/models` or a small capabilities route — alongside anything else required
   to construct a prompt correctly. Asking a client to scrape `/health` to decide
   how to build prompts is the wrong contract.

3. **The reported numbers are partly literals.** `server_state_cache_health_json`
   hardcodes `resident_checkpoints: 0`, `daemon_prefix_hash: false`,
   `semantic_boundary_checkpoints: false`. A client cannot distinguish the
   *configured* max from what is *actually available now*, so both should be
   reported and both should be live. This overlaps
   `docs/bugs/2026-08-30-prefix-state-cache-never-engages.md` — until that lands
   the honest answer for available checkpoints is 0, and saying so is better than
   reporting a budget the client cannot spend.
