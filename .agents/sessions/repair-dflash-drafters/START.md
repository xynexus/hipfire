# Session: repair the six unusable DFlash drafters

**Blocked on:** nothing. This is plain work with a proven recipe.
**Est:** one session. **Value:** unblocks every DFlash / spec-decode measurement
on five families that currently cannot load a drafter at all.

## Objective

Every `*-DFlash--bf16.hfq` on `/srv/hipfire/models` is tagged **`arch 1`
(qwen2)** instead of `arch 20`, so `DflashConfig::from_hfq` rejects all seven and
no DFlash path can load them. Produce working arch-20 drafters for the six that
are still broken, verified by output identity.

Done when: each repaired drafter drives `dflash_spec_demo` and produces a draft
stream you have compared against a reference — see the bar below.

## Why now

Speculative decode cannot be measured on Qwen3.5-9B, Qwen3.5-122B,
Qwen3.5-397B, Qwen3.6-35B-A3B, gemma-4-26B or gemma-4-31B, because the drafter
will not load. That is five families and the two largest models on the box.

## They are MISLABELLED, not structurally different — proven

Diffed a broken artifact against one freshly built by `dflash_convert` from the
same upstream checkpoint:

| | broken (`/srv`) | freshly converted |
|---|---|---|
| tensors | 58, incl. `fc.weight`, `hidden_norm.weight` | 58, identical names |
| geometry | hidden 5120, 5 layers, 32/8 heads, head_dim 128 | identical |
| `config.block_size` | 16 | 16 |
| `config.dflash_config` | `mask_token_id`, `target_layer_ids` | identical |
| top-level `dflash` | **absent** | present |
| arch id | 1 | 20 |

Everything the loader needs is already in the broken file. **No HF source
required** — which matters, because only one of the seven has a matching source
on `/srv`.

## First moves

Recipe and field mapping: `docs/help/hfq-container-surgery.md`.

```sh
SRC=/srv/hipfire/models/<Name>-DFlash--bf16.hfq   # READ-ONLY, never write here
W=~/.hipfire/models/_repair && mkdir -p "$W"

hipfire inspect "$SRC" --json        # build dflash.json from config + config.dflash_config
hfq rearch   "$SRC" "$W/s1.hfq" --arch-id 20
hfq meta-set "$W/s1.hfq" "$W/out.hfq" --key dflash --value-file dflash.json --json
```

`--json` is **required** — without it the block is stored as a JSON *string* and
the load fails with the same opaque `parse DflashConfig`, naming no field.

## The verification bar

Loading is not the bar. Compare the **draft token stream** against a reference on
the same target and prompt:

```sh
dflash_spec_demo --target <target.hfq> --draft <drafter.hfq> \
  --prompt "..." --max 32 --ddtree-batched --ddtree-budget 12 --ddtree-topk 1
```

The 27B repair produced a token-for-token identical stream and identical
`decode_tau` (10.6667) against its `dflash_convert` build. Where no reference
exists, at minimum assert the stream is non-degenerate and `decode_tau > 1`.

## Traps

- **`/srv` is read-only.** Shared NFS, often the only copy. Copy or write elsewhere.
- **`rearch` rewrites the payload, so on-disk codecs are lost.** `Bf16Huff` comes
  out plain `BF16`: 2.29 GB -> 3.46 GB on the 27B. Functionally identical, larger.
  Decide whether to re-apply the codec before publishing anywhere.
- `num_target_layers` is informational (`from_hfq` never reads it) but is
  recoverable: `max(target_layer_ids) + 3`.
- Only Qwen3.6-27B has a matching HF source; the rest **must** be repaired rather
  than reconverted.

## Then

Update `docs/todo/DECISIONS-PENDING.md` item 8 — it is written as a
delete-or-lose decision and is now stale. This work retires it.
