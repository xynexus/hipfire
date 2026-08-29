# TUI / web config editing — findings and scope (2026-08-29)

Written while wiring the n-gram spec-decode config knobs, which surfaced the
limitation. Nothing here is n-gram-specific: it affects most of the config
surface.

## The finding

**55 of 82 config fields cannot be edited in the TUI.** Everything that is not a
`bool` or an `enum` is display-only.

The advanced list itself is fully schema-driven — `ConfigState` builds
`advanced_rows` from `build_config_editor_snapshot_from_paths`
(`crates/hipfire-config/src/editor.rs:121`), so a new `field!` entry shows up in
the TUI with no TUI change at all. Display is not the problem; **input is**.

`edit_row` (`crates/hipfire-tui/src/hipfire/config.rs:197`) resolves a new value
three ways:

1. `default_model` — special-cased, takes the Models-tab selection;
2. `cycle_row_value` (`:493`) — toggles a `bool`, or steps through
   `row.choices` for an `enum`;
3. otherwise → `Err("{label} does not support cycling")` (`:220`).

There is no text or numeric entry anywhere on that screen. So every `String`,
`Path`, `U8/U16/U32/U64`, `I32`, `F64` and `Json` field is unreachable: paths,
directories, sizes, thresholds, model names.

Reproduce the count:

```sh
cargo run --release -p hipfire-cli -- gen-config-schema --format json \
  --output docs/config-schema.json
python3 -c "
import json; rows=json.load(open('docs/config-schema.json'))
stuck=[f['key'] for f in rows if f['type']['kind'] not in ('bool','enum')]
print(len(stuck), 'of', len(rows)); print('\n'.join(sorted(stuck)))"
```

## What already works (do not rebuild these)

- **Schema is the single source of truth.** `config_schema()` drives the TUI
  rows, `GET /admin/config/schema`, and the generated docs.
- **The write path takes arbitrary values.** `apply_config_edit`
  (`crates/hipfire-config/src/editor.rs:140`) accepts any `serde_json::Value`
  and does its own validation. The TUI is the only thing restricting input to
  cycling — the layer beneath it is already general.
- **HTTP is complete.** `GET /admin/config/schema`
  (`crates/hipfire-server/src/lib.rs:62`), `GET`/`PATCH /admin/config/editor`
  (`:70`, handlers at `crates/hipfire-server/src/routes/admin.rs:163/180/188`).
  `PATCH` sets any field to any value today.

## What is missing

1. **TUI input mode.** An editing state on the config screen: enter on a
   non-cyclable row opens a buffer seeded with the current value, typing edits
   it, enter commits via `apply_config_edit`, escape cancels. Validation already
   exists below, so the TUI only needs to collect a string and parse it to the
   row's `ConfigType`. This one change unlocks all 55 fields.
2. **No web config UI exists at all.** `hipfire-admin-ui` is access + usage
   only; `hipfire-web-ui` (277 lines) has no config surface. The endpoints are
   there and unused — a settings page is a client-side job, not a server one.

## Notes that will bite

- `EASY_KEYS` (`config.rs:16`) is a hand-picked list of five. It is a curation
  layer over `advanced_rows`, not a separate source — adding a field to the
  schema does *not* put it in Easy, and it does not need to.
- A field's `editable_global` flag already gates writes; respect it rather than
  adding a second rule.
- Sentinel values matter for a UI. `ngram_store_root` accepts `ram`/`none`/`off`
  as well as an empty string precisely because a settings UI cannot distinguish
  "unset" from "deliberately RAM-only". Any field where blank carries meaning
  wants the same treatment.
- `ConfigMutability::LoadTime` fields only take effect on the next model load;
  the TUI already renders that via `impact_label`, so surface it on commit too.
