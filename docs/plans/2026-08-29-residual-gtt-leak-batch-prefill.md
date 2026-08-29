# GOAL: find the ~95 MB/request GTT leak in the qwen3.5 batch-prefill path

Open as of 2026-08-29. Two larger session leaks were fixed in `4db8dd954`
(release now frees; the unusable Final checkpoint is no longer minted), taking
the 27B from ~1100 MB/request to ~91 MB/request. **This document is about the
remainder.** It is not urgent — a 30 GB host now survives ~100 requests instead
of ten — but it is a real, linear leak and it is well characterised, so the next
person should not have to re-derive any of this.

Measured on nix2 / gfx1103 / 30 GB UMA, `Qwen3.8-27B--oq4.25++` (arch 5).

## Reproduce

Plain `/v1/chat/completions`, short prompt, few tokens out, sampling
`MemAvailable` between requests:

```
req  1  avail=11396 MB
req  7  avail=10887 MB
req 14  avail=10218 MB      -> ~91 MB/request, dead linear, never plateaus
```

`resident_sessions` stays 0 and `runtime_session_bytes` stays 0 throughout, so
nothing in hipfire's own accounting reflects it.

## Established facts — do not re-test these

1. **Batch-prefill path only.** With `HIPFIRE_SERVER_PREFILL_BATCH=0` memory is
   perfectly flat over 12 requests (12449 → 12610 MB, drifting *up*), at an
   identical ~4 s/request. Flip the switch and the leak returns. This is the
   single most useful bisect: it isolates the leak to the batch path and gives a
   known-good control.
2. **Fixed per request; not decode-scaled.** 95.0 MB/req at `max_tokens=1`
   versus 96.8 MB/req at 19 tokens generated. Allocated once per request
   regardless of work done.
3. **Driver/GTT, not process memory.** Over 10 requests `MemAvailable` fell
   983 MB while the summed RSS of every hipfire process moved **+1 MB**. So it is
   `hipMalloc`'d device memory (which on a UMA APU comes out of the same pool as
   everything else but never appears in RSS), not a host-side `Vec`/map/telemetry
   growth.
4. **hipfire's own accounting stays flat.** `resident_vram_bytes` holds at
   15.48 GB (the weights), `runtime_state_bytes` at ~1182 MB (the active
   session). Whatever grows is not tracked by the worker memory view — which is
   itself worth fixing, since it makes the leak invisible from `/health`.

## Ruled out

- **Session alloc/free asymmetry.** `qwen35_allocate_session_state` allocates
  exactly KV cache + DeltaNet state + logits; `Qwen35RequestSessionState::free_gpu`
  frees exactly those three. Balanced.
- **Incomplete DeltaNet free.** `DeltaNetState::free_gpu` frees all four of its
  tensor vectors (`s_matrices`, `s_scales`, `conv_states`, `s_ef_residual`), and
  the struct holds no other tensor fields.
- **A failed downcast silently skipping the recurrent free.**
  `impl RecurrentMixerState for DeltaNetState`'s `into_any` returns `self`, not
  `Box::new(self)`, so `downcast::<DeltaNetState>()` succeeds.
- **`SessionCursor`.** Host-only (`usize` + `Vec<u32>`).
- **The deferred-free mailbox.** Hypothesis was that `OwnedTensor::drop` only
  queues buffers and `qwen35_prefill.rs` never calls `reclaim_pending()` (true —
  it doesn't). **Tested: adding a `gpu.reclaim_pending()` at the end of
  `run_generate_batch_prefill_serial_qwen35` changed nothing** — still ~91 MB/req.
  The change was reverted. Do not re-try this alone.

## Suspects, in the order worth checking

1. **Per-session forward scratch on the SerialReference path.** With a single
   request, `qwen35_prefill_suffix_batch` picks `SerialReference`
   (`non_empty.len() < 2`), which routes to `qwen35_prefill_active_session` →
   `forward_scratch` (`crates/hipfire-arch-qwen35/src/qwen35/mod.rs:891`). The
   non-batch path prefills the *legacy* session, reusing whatever scratch it
   already has, while the batch path works against a freshly allocated
   per-request session. That asymmetry matches fact 1 exactly. Check whether
   scratch is cached per session/model and whether a new session forces a fresh
   allocation that is never freed.
2. **Pool bucket mismatch.** `Pool::free` keys free lists by power-of-2 bucket
   (`crates/hipfire-rdna/src/pool.rs:87`). If a per-request allocation lands in a
   bucket nothing reuses, the pool grows monotonically: `MemAvailable` falls,
   RSS does not move, and hipfire's residency accounting stays flat — all three
   observed facts. **`Pool` already tracks `total_new` and `total_allocated`
   (`pool.rs:20-22`) and neither is exposed anywhere.** Surfacing them (health or
   a diag frame) is the cheapest next diagnostic and would settle suspects 1 and
   2 immediately: growing `total_new` means real new allocations, flat
   `total_new` with falling `MemAvailable` means something outside the pool.
3. **Something in the batch-only protocol steps.** `reserve_session_state`
   (`crates/hipfire-daemon/src/handlers/sessions.rs:91`) takes a reservation in
   `generic_state_arena` with a TTL. It looks like host-side bookkeeping and so
   should not explain a GTT-side leak, but nothing confirms the reservations are
   ever released.

## Suggested next step

Expose `Pool::total_new` / `total_allocated` (and the per-bucket free-list
lengths) and re-run the 14-request loop. That one number splits the remaining
search space in half without any guessing, and the fields already exist.

Related: xynexus/hipfire#385 (the two fixed leaks), #384 (the health counters
that are hardcoded zeros — the same blind spot that hid this one).
