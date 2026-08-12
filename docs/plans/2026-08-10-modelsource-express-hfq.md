# Plan: make `ModelSource` able to express HFQ

Scoped 2026-08-10, after wiring `.hfa` into `hipfire-quantize` (PR #242) hit the
same wall from the other side.

## STATUS (2026-08-10): COMPLETE — P0–P4 landed, `.hfq` and `.hfa`

| phase | commit | evidence |
|---|---|---|
| P0 Cow-yielding `tensor()` | `1442898a7` | safetensors stays `Cow::Borrowed`, asserted by pointer identity |
| P1 real `ModelSource for HfqFile` | `1442898a7` | `modelsource_parity`: 320/320 tensors identical vs source dir, 1.5 GB |
| P2 shared bf16 decode | `e389c03e7` | tiny-affected gate: same 8 pre-existing failures, no new |
| P3 generic `layer_stream` | `46b0ba196` | source manifest fingerprint unchanged (`fnv64:bf8adfccd57b73c5`) |
| P4 `.hfq` calibrate source | `c620c79a8` | `compare-calibration`: 375 tensors / 277,502,460 values, `max_abs_error 0.0` |
| P4 `.hfa` calibrate source | this branch | same oracle, same numbers — archive vs restored directory |

Two corrections to this plan, found by building it:

- **No `TensorDesc` was needed.** `TensorInfo` already carries `quant_type` with
  an explicit "For HFQ: the quant_type byte" contract. The only real obstacle
  was the BORROW — the trait hands out `&TensorInfo` and HFQ stores
  `HfqTensorInfo` — solved by a lazy mirror of the index.
- **P3 was not the risky step.** `begin`/`step`/`finish` and
  `source_manifest_identity` already took `&dyn ModelSource`; the entire
  safetensors binding was two struct fields. The risk was actually in P4, where
  the trait's borrowed-only accessor silently paired PACKED bytes with LOGICAL
  metadata — a wrong-size read that the dry-run planner cannot catch because it
  never touches a payload.

### `.hfa` as a calibrate source — DONE

Route 1, as the plan preferred, once #242 landed and freed the file to move.
`hfa.rs` now lives in `hipfire-quant-format` — the crate that already declares
itself the contract between the writer (`hipfire-quantize`) and every reader
(`hipfire-runtime`), which is exactly the boundary an archive reader sits on. It
needed only `std` + `hipfire_primitives`, both already reachable there.
`hipfire-quantize` re-exports it, so existing `hipfire_quantize::hfa::` paths
still resolve.

`SafetensorsSource` gained the archive backend, the same shape #242 gave
`SafetensorsFile`: `Mmap` borrows mapped pages, `Archive` decodes owned bytes
out of the `.hfa`. Both share one `index_shard_tensors` helper, because the
archive stores each shard's safetensors header VERBATIM — the tensor table is
identical either way and only payload retrieval differs.

Three things only a REAL calibration surfaced, none visible to a dry run:

- The archive carries `tokenizer.json` inside it, so there is no file beside it
  for `tokenizer_json_path` to point at. It is folded into the metadata under
  the `tokenizer` key — the same place an `.hfq` keeps it, reusing the fallback
  P4 already added.
- `tensor_storage` cannot return `None`: `TensorLoadPlan` requires it for
  physical-alias detection and byte-length validation. Archive shards get a
  synthetic `<archive>#<shard>` path with logical offsets, which preserves both.
- `source_shard_identity` opened that synthetic path with `fs::metadata` before
  reaching any format arm. The archive arm had to move ABOVE it, and identifies
  a shard by its header (`hfa_shard_header_hash`) — the same class as the other
  arms: hash the index, never the payload.

## Why it is worth doing

It is the same blocker three times over: the 122B/397B restore cost, the
per-arch decode duplication, and `.hfq` not being a calibration source. One
owned-or-borrowed view type retires all three.
