# `kv_cache`: `fp32` was unwritable, and the daemon's fallback reads a raw env var

Status: **FIXED 2026-09-02.** Closes GitHub issue #386.

## CORRECTION to the first version of this document

This file previously claimed "`kv_cache` never reaches the loader — every model
runs KVarN 4-bit". **That was wrong**, and the wrong version is on `master`
(merged in #406). The correction:

`load_model`'s `kv_mode_override` argument is filled from
`protocol_load.params.kv_cache` (`handlers/lifecycle.rs:245`), and
`ModelLoadParams::from_hipfire_config` fills THAT from the resolved
`config.kv_cache`. So on the server path the setting works. Verified directly:

    params.kv_cache = kvarn8   ->  loader reports "K 8b"
    params.kv_cache = fp32     ->  loader reports "KV cache: fp32"
    no param, no env           ->  "K 4b" (the auto default)

What misled me: driving `hipfire daemon` over its stdin protocol with a load
frame that carries no `kv_cache` param. There is no config resolution on that
path, so it falls through to `std::env::var("HIPFIRE_KV_MODE")`. Setting
`HIPFIRE_KV_CACHE` — the env LAYER for the config key — therefore did nothing,
and every "KV mode" comparison I ran that way was silently the same mode. The
measurements I took through `HIPFIRE_KV_MODE` are unaffected and stand.

## The real defects

**1. `fp32` was missing from the enum — FIXED.**

    values: ["auto", "q8", "asym2", "asym3", "asym4", "kvarn2", "kvarn", "kvarn4", "kvarn8"]

The loader accepts `fp32`, and the deprecation message names it as one of the two
supported families ("kvarn ... and unquantized (fp32)"). But it was not a legal
config value, so the one exact, non-lossy mode — the reference any KV comparison
needs — could only be reached through the undocumented `HIPFIRE_KV_MODE`. Added.

**2. Deprecated values pass config validation then fail at load — ALREADY
HANDLED, and now visible.** `deprecated_kv_diagnostics` has emitted the
migration warning (naming `HIPFIRE_KV_ALLOW_DEPRECATED=1`) all along. What was
missing is that the daemon DISCARDED config diagnostics
(`load_config_bundle().config` drops `.diagnostics`), so the warning never
reached an operator. Fixed in #406; verified that `kv_cache=q8` now logs
"is DEPRECATED and the model loader will refuse it" at load.

The deprecated members are deliberately kept in the enum. Removing them would
make an existing config fail domain validation, and this area's rule is that a
bad value is WARNED about, not rejected.

**3. The daemon's stdin fallback consulted an undocumented env var — FIXED.**
`load.rs` fell back to `std::env::var("HIPFIRE_KV_MODE")` only. `kv_cache` is the
schema key, so its env-layer spelling is `HIPFIRE_KV_CACHE` — what an operator
reads in `docs/config-schema.md` and reasonably expects to work. Setting the
documented variable did nothing on this path while the undocumented one worked.
That cost real time: an entire sweep of "different KV modes" in this session was
silently one mode, and a correct conclusion was retracted on the strength of it.

Both spellings are now accepted, with `HIPFIRE_KV_MODE` still winning when both
are set so nothing relying on it changes. Verified:

    HIPFIRE_KV_CACHE=kvarn8                        -> K 8b
    HIPFIRE_KV_MODE=kvarn8                         -> K 8b
    HIPFIRE_KV_CACHE=kvarn2 HIPFIRE_KV_MODE=kvarn8 -> K 8b (MODE wins)

The daemon-direct path still has no resolved config by design, so this is a
fallback rather than a config read — env stays a layer, not a bypass.

## Issue #386's ask

> Make the enum and the loader agree.

Done. `fp32` is a legal config value, deprecated modes warn at config-resolution
time and the warning now reaches the operator, and both env spellings select the
mode. No known divergence remains between the enum and the loader.
