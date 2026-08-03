# hipfire-hub: a reliable HuggingFace downloader

Status: scoped, not started.

## Goal

A Rust-native HuggingFace downloader for hipfire.

## Context

The model store is a RAID0 array with no redundancy. This is deliberate —
models are re-downloadable, so the array trades redundancy for speed — but it
makes this tool the recovery mechanism for ~2.6 TB of single-copy data. The bar
is not "usually works": a download that silently produces a wrong file is worse
than one that fails loudly.

One shard of `Qwen3-30B-A3B-Thinking-2507` is already unreadable — a hard I/O
error ~2.0 GB into `model-00006-of-00016.safetensors`, with the other fifteen
shards clean. Repairing exactly that file is the first real test.

## Placement

New crate `crates/hipfire-hub`, surfaced through the `hipfire-coexistence`
binary. Per AGENTS.md this is external-ecosystem interop: it must not be linked
into the daemon, server, or runtime hot path. Rust only.

## Dependencies

Use what the workspace already has: `reqwest` 0.12 (`default-features = false`,
features `["stream", "json", "rustls-tls"]`), `tokio`, `sha2`, `futures-util`,
`indicatif`. Net-new dependencies require justification.

**Do not add Xet support** (`xet-client`, `hf-xet`, `shardline-xet-core`). It
was evaluated and rejected on three grounds:

- it forces a second crypto stack (`aws-lc-rs`/`aws-lc-sys`) with no `ring`
  escape hatch in `xet-client`'s features, against a workspace standardised on
  rustls+ring, in a repo that treats portability as a design constraint;
- it forces a second `reqwest` major version (0.13 against the workspace's
  0.12), so two full HTTP/hyper/TLS stacks in one binary;
- its chunk-level dedup gains close to nothing for this corpus. Sibling
  finetunes share only their tokenizer blob, not weights — measured:
  `LFM2.5-1.2B-Instruct` and `-Thinking` share exactly one 4.6 MB blob
  (`2e27bba0…`) and no weight bytes. Xet's real win is iterative re-upload of
  the *same* model, which is a training workflow, not a fetch workflow.

`shardline-xet-core` is architecturally the better shape — it reimplements only
the CAS structures and pulls no crypto or HTTP stack (37 net-new crates vs 47,
no aws-lc) — but at 310 downloads, v1.3.0 against HF's 1.5.4, and days old from
a third party, it is not something to stake a recovery path on. A subtly wrong
`MerkleHash` would not fail loudly; it would compute a wrong content hash.

Everything needed here is achievable with HTTP + SHA-256.

## Design

**Integrity anchor.** The SHA-256 HF exposes via `X-Linked-Etag` (the LFS oid).
The local cache already records it — see line 2 of
`.cache/huggingface/download/<file>.metadata`. For small non-LFS files fall back
to size + ETag, and record which check was applied rather than implying a
stronger guarantee than was made.

**Atomic commit.** Stream → incremental SHA-256 → `fsync` → compare digest →
`rename` into place. A file becomes visible at its final path only once proven
correct. Never splice a partial file, never trust size alone.

**Layout.** HF cache-compatible: `blobs/<sha256>`, `snapshots/<rev>/<path>`
symlinks, `refs/main`. A drop-in for existing tooling, and it provides the
file-level dedup that actually exists — skip the transfer entirely when the blob
is already present.

**Resume.** HTTP `Range` against `<blob>.part`, with persisted offset and
partial hash state so resuming does not reread from zero. On digest mismatch,
discard and restart that file.

**Preflight.** Resolve the file list and sizes, sum them, compare against free
space with a margin, and refuse before writing anything; abort on a floor
mid-run. A previous bulk job filled the filesystem to 100% and stalled the
machine — the check that would have prevented it is cheap.

**Concurrency.** Bounded parallel files (default ~4), one stream per file.
Sequential-per-file keeps resume simple; parallelism across shards is where the
throughput is.

**Retries.** Jittered backoff, distinguishing retryable (5xx, timeout,
connection reset) from fatal (401/403/404). Auth via `HF_TOKEN` for gated repos.

**Concurrent-run safety.** PID-scoped `.part` files. If a mutex is needed, use
`hipfire-lock` — do not add a new lock primitive.

## CLI

```
hipfire-coexistence hub fetch  --repo <org/name> [--revision <sha|main>]
                               [--include <glob>] [--dest <dir>]
hipfire-coexistence hub verify --repo <org/name> [--revision ...]
hipfire-coexistence hub repair --repo <org/name>
```

`verify` and `repair` are the ones that matter on a no-redundancy array.

## Verification

This is the deliverable, not the code.

- Fault injection, not happy paths: truncated response mid-stream, wrong digest,
  connection dropped mid-transfer, ENOSPC, stale or corrupt `.part`. Each must
  fail loudly and leave no file at its final path.
- Prove a corrupted download is **rejected**, not merely that a good one
  succeeds.
- End to end: `repair` on `Qwen3-30B-A3B-Thinking-2507` must detect exactly the
  one bad shard, re-fetch only that file, and leave the other fifteen untouched.

The weighting is deliberate. Every integrity bug found while building the
archive tooling hid behind a clean-looking success: a `u32` chunk-offset
overflow corrupted tensors while the converter reported
`Max quant error: 0.00000000`, and a one-directional verify reported
"62 files byte-identical" while 43 source files were missing from the archive.
Tests that only assert success reproduce that failure mode.

## Out of scope

Uploads. Xet. Datasets (phase two — different API shape). Any coupling to the
inference path.
