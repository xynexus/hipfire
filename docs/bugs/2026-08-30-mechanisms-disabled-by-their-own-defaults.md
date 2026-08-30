# Five complete mechanisms, each disabled by its own default

Status: recorded 2026-08-30 on nix1. Four fixed (PRs #390, #391, #393), one open
(see `2026-08-30-prefix-state-cache-never-engages.md`). This file is the pattern,
not a sixth bug — it exists so the next one is found by a gate instead of by
someone noticing a number that should not be zero.

## The shape

Each of these was fully implemented, correct, and reachable. Each did nothing,
silently, because one default or literal short-circuited it. None produced an
error, a warning, or a metric that looked wrong — the only outward sign was an
absence.

| mechanism | what disabled it | how it surfaced |
|---|---|---|
| `kv_cache: "auto"` | `non_auto_value` mapped `"auto"` -> `None` before it became a load param, leaving the match's `auto` arm dead | fp32 KV everywhere; the `auto` arm unreachable from config |
| `model_overrides` | exact-key lookup vs a `--`/`-` artifact-name mismatch | six MiniCPM entries inert; config said `q8`, runtime logged `kvarn` |
| n-gram spec-decode settings | `load_config_bundle()` re-read config with no CLI layer and no model tag | per-model `ngram_spec` unreachable |
| batch-prefill routing | `batch_envelope_ok` read `HIPFIRE_DFLASH_DRAFT`, unset for a sibling-discovered drafter | every request to a DFlash model failed the whole batch cycle |
| `scheduler_vram_budget_bytes` | default `0`, and `effective_limit` reads `0` as UNLIMITED | no eviction, no module residency, parked workers grew until OOM |
| `semantic_boundary_checkpoints` | hardcoded `false` at its only assignment | prefix cache never populated; still open |

## Why nothing caught it

Three properties recur:

**Zero and "auto" both mean "unset", and "unset" was implemented as "no limit".**
`effective_limit(0, _) -> None` is a reasonable local decision that becomes
"admit everything, evict nothing" three layers up. `non_auto_value("auto") ->
None` is the same move on a string.

**The disabled path had no observable.** A budget of 0 produces no log line. An
override that matches no key produces no diagnostic. A load param that is never
populated is indistinguishable from one whose value happens to be the default.
`prefix_hash_preflight_requests: 0` and `state_cache entries: 0` are the only
evidence the prefix cache is dark, and both read as "idle".

**Reporting sometimes actively lied.** `cache_write_tokens` is
`prompt_tokens - cached_tokens`, so with caching off it equals the prompt length
and reads as "31 tokens cached". `ModelWorkerRuntimeView` hardcodes
`max_resident_workers: 1` while the top level reported 2 resident. A metric that
is wrong in the reassuring direction is worse than a missing one.

## What would have caught all of them

A gate that walks the config schema and, for each field, asserts something reads
it and that a non-default value changes observable behaviour. Concretely:

* every `field!` in `schema.rs` has at least one reader outside the config crate;
* every field the daemon consumes travels through `ModelLoadParams` (the rule
  written into AGENTS.md by #390), so `model_overrides` can reach it;
* a value that means "no limit" is spelled as such and logged once at startup
  when it is in force, rather than being the silent default.

The startup diagnostics added in #390 (`report_config_diagnostics`) cover the
first class — unknown keys, out-of-domain values, missing paths — and would have
caught the `kv_cache` and `model_overrides` cases at boot. They do not cover "this
knob is set to the value that turns the feature off", which is what the remaining
three were.

## The cheap next step

`scheduler_vram_budget_bytes` now logs when it derives a budget (#393). Doing the
same for every policy-bearing default — one line at startup naming the value in
force and whether it disables the feature — would convert this whole class from
"noticed by accident" to "stated on every boot".
