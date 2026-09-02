# `kv_cache` never reaches the loader — every model runs KVarN 4-bit

Status: **OPEN — diagnosed, not fixed.** Found 2026-09-02 while asking why
KVarN-8 was not behaving like KVarN-8.

## The defect

`hipfire_serving_core::load::load_model` resolves the KV mode like this
(`load.rs:846`):

    let mut kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    if kv_mode.is_empty() || kv_mode == "auto" {
        kv_mode = "kvarn".to_string();
    }

The fallback is a raw `std::env::var("HIPFIRE_KV_MODE")` — **not the `kv_cache`
schema field**. Nothing in the daemon load path fills `kv_mode_override` from
config, so `kv_cache` is inert: the mode always resolves to `auto` and then to
plain `"kvarn"`, whose width is 4.

The comment immediately below it reads "ask for it explicitly (`kv_cache=fp32`)
if you want it" — naming the setting that does not arrive.

## Evidence

300 greedy tokens, same prompt, AR only. Setting the documented config key
changes nothing; every run is 4-bit and byte-identical:

    HIPFIRE_KV_CACHE=kvarn2   -> "K 4b"   77ab36b1ad5a569f
    HIPFIRE_KV_CACHE=kvarn4   -> "K 4b"   77ab36b1ad5a569f
    HIPFIRE_KV_CACHE=kvarn8   -> "K 4b"   77ab36b1ad5a569f
    HIPFIRE_KV_CACHE=fp32     -> "K 4b"   (silently ignored, no warning)

`HIPFIRE_KVARN_BITS` is equally inert on this path, because the width comes from
`kvarn_bits_from_mode(kv_mode)` and `kv_mode` has already been flattened to
`"kvarn"`.

Through the lever the loader actually reads, everything works:

    HIPFIRE_KV_MODE=kvarn2  -> "K 2b"   25bffaa8e7d819a3
    HIPFIRE_KV_MODE=kvarn4  -> "K 4b"   77ab36b1ad5a569f
    HIPFIRE_KV_MODE=kvarn8  -> "K 8b"   86f354c73ef92541
    HIPFIRE_KV_MODE=fp32    -> fp32     c27c9545c599d88d

So the plumbing below `kv_mode` is correct — `kvarn_bits_from_mode` maps
`kvarn8 -> 8` and the constructor threads it. Only the resolution of `kv_mode`
itself is wrong.

## Why it matters beyond the setting being dead

This is what made the spec/AR losslessness question so hard to answer. Every
comparison across "different KV modes" was in fact the same mode, so the widths
could not order correctly, the numbers looked non-physical, and a correct
conclusion was retracted on the strength of them. See
`2026-09-01-spec-decode-not-output-equivalent-to-ar.md`, where fp32 KV turns out
to make speculative decode byte-identical to AR — a result that was unreachable
while `kv_cache=fp32` silently did nothing.

`fp32` is also the mode the deprecation warning calls one of the two supported
families ("hipfire is retiring KV storage down to two families: kvarn and
unquantized (fp32)"), while the `kv_cache` schema enum still lists the deprecated
`q8`/`asym*` and **omits `fp32` entirely**. So the one mode with a correctness
argument behind it cannot be named in config at all.

Setting a deprecated value at least warns. Setting `fp32` is accepted in silence
and ignored — the failure mode this repo's config rules single out.

## Fix

1. Fill `kv_mode_override` from the resolved `kv_cache` setting on the daemon
   load path, so config reaches the loader; keep `HIPFIRE_KV_MODE` as the
   env LAYER, not as the only source.
2. Add `fp32` to the `kv_cache` enum and drop or mark the deprecated members, so
   the schema matches what the loader accepts.
3. Reject an unknown/unsupported `kv_cache` value loudly rather than falling
   through to `auto`.
