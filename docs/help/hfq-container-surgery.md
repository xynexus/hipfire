# `hfq` — container surgery without re-quantizing

`hfq` edits the **container** of a `.hfq` artifact: its header arch id, its
metadata JSON, and which tensors it carries. It never touches weight values, so
these operations are cheap and lossless relative to the quantizer — the whole
point is to fix an artifact you cannot or do not want to rebuild.

`docs/CLI.md` is generated from clap and lists `hfq` as a single line. This page
covers the flags that page does not.

```
hfq list      <file>
hfq verify    <file>
hfq extract   <in> <out> --tensor <pat>...
hfq meta-get  <file> [--key <k>]
hfq meta-set  <in> <out> --key <k> (--value <v> | --value-file <f>) [--json]
hfq rearch    <in> <out> --arch-id <id>
```

Every mutating form is `<in> <out>` — it writes a new file and never edits in
place. That matters on `/srv`, which is a shared mount and must be treated as
read-only: copy first, or write the output elsewhere.

---

## `rearch --arch-id <id>`

Rewrites the architecture id in **both** the container header and
`metadata.arch_id`, copying every tensor through unchanged.

```sh
hfq rearch broken.hfq fixed.hfq --arch-id 20
# set header arch_id and metadata.arch_id to 20 (58 tensors copied) → fixed.hfq
```

Use it when an artifact's weights are right but its family tag is wrong, so the
loader refuses it or routes it to the wrong architecture.

⚠️ **It rewrites the payload, so on-disk codecs are not preserved.** A `Bf16Huff`
artifact comes out as plain `BF16`. Measured on a 27B drafter: 2.29 GB → 3.46 GB,
the same 1.51x the Huffman codec was buying. Functionally identical, larger on
disk. If size matters, re-apply the codec afterwards rather than assuming it
survived.

## `meta-set --key <k> (--value <v> | --value-file <f>) [--json]`

Inserts or replaces one top-level metadata key.

```sh
# a string value (the original use case: a jinja chat template)
hfq meta-set model.hfq model+tmpl.hfq --key chat_template --value-file tmpl.jinja

# a STRUCTURED value — note --json
hfq meta-set in.hfq out.hfq --key dflash --value-file dflash.json --json
```

**`--json` is not optional when the value is an object or array.** Without it the
value is stored as a JSON *string*, and a consumer doing `meta["dflash"]
.get("num_hidden_layers")` gets `None` from a string rather than a field from an
object. The failure is silent at write time and opaque at read time — the drafter
loader reports only `parse DflashConfig`, naming no field.

It is opt-in rather than "parse it if it looks like JSON" so that a template
which happens to be valid JSON cannot silently change type.

Check the result with `meta-get`, which prints the value's actual JSON type:

```sh
hfq meta-get out.hfq --key dflash
```

---

## Worked example: repairing a mislabelled DFlash drafter

Every `*-DFlash--bf16.hfq` on `/srv/hipfire/models` is tagged `arch 1` (qwen2)
instead of `arch 20`, so `DflashConfig::from_hfq` rejects all of them and no
DFlash path can load them. They are **mislabelled, not structurally different** —
verified by diffing one against a drafter freshly built by `dflash_convert` from
the same upstream checkpoint:

| | broken (`/srv`) | freshly converted |
|---|---|---|
| geometry | hidden 5120, 5 layers, 32/8 heads, head_dim 128 | identical |
| tensors | 58, incl. `fc.weight`, `hidden_norm.weight` | 58, identical names |
| `config.block_size` | 16 | 16 |
| `config.dflash_config` | `mask_token_id`, `target_layer_ids` | identical |
| top-level `dflash` | **absent** | present |
| arch id | 1 | 20 |

Two things are wrong, and fixing only the arch id is not enough: the loader reads
its geometry from a **top-level `dflash` object** that these artifacts lack.
Everything that object needs is already present under `config` and
`config.dflash_config`, so no upstream checkpoint is required.

```sh
SRC=/srv/hipfire/models/Qwen3.6-27B-DFlash--bf16.hfq   # read-only: never write here
W=~/.hipfire/models/_repair && mkdir -p "$W"

# 1. build the dflash block FROM THE BROKEN FILE'S OWN metadata
hipfire inspect "$SRC" --json > "$W/meta.json"
#    map config.{hidden_size,num_hidden_layers,num_attention_heads,
#                num_key_value_heads,head_dim,intermediate_size,vocab_size,
#                rms_norm_eps,rope_theta,block_size}
#    plus config.dflash_config.{mask_token_id,target_layer_ids}
#    into  $W/dflash.json   (see the field list in `DflashConfig::from_hfq`)

# 2. fix the arch tag
hfq rearch "$SRC" "$W/step1.hfq" --arch-id 20

# 3. add the block the loader actually reads
hfq meta-set "$W/step1.hfq" "$W/repaired.hfq" --key dflash \
    --value-file "$W/dflash.json" --json
```

`num_target_layers` is informational (`from_hfq` never reads it) but is derivable
if you want it faithful: `build_target_layer_ids` uses
`end = num_target_layers - 3`, so `max(target_layer_ids) + 3` recovers it — 61
gives 64 on the 27B, matching a freshly converted drafter.

**Verify by output, not by loading.** A repaired artifact that loads can still be
subtly wrong; compare its draft stream against a known-good drafter on the same
target and prompt:

```sh
dflash_spec_demo --target <target.hfq> --draft <drafter.hfq> \
  --prompt "..." --max 32 --ddtree-batched --ddtree-budget 12 --ddtree-topk 1
```

The repair above produced a **token-for-token identical** draft stream and an
identical `decode_tau` (10.6667) against the `dflash_convert` build. That is the
bar: identical tokens, not merely "it loaded".

## When to reach for `dflash_convert` instead

Container surgery only moves what is already in the file. If the tensors
themselves are wrong — a different architecture, a different quantization, a
genuinely missing weight — rebuild from the upstream checkpoint:

```sh
dflash_convert --input /srv/huggingface/models--<org>--<repo>/snapshots/<rev>/ \
               --output ~/.hipfire/models/<Name>--dflash.bf16.hfq
```

Only one of the seven artifacts above has a matching local HF source, which is
why the metadata repair matters: it recovers the other six without one.
