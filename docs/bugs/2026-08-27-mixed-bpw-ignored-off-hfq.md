# `--mixed-bpw` is silently ignored unless the input is an `.hfq`

Status: **FIXED 2026-08-30** — the silent no-op is gone; `--mixed-bpw` now
REFUSES a non-`.hfq` input (and a malformed value, which used to `.ok()` into
`None` and read as "flag not passed"). Threading the promoter into the source
pipeline is still open, and is the better fix; see "Fix" below. Opened
2026-08-27. Summary entry in `BUGS.md`.

## The defect

`--mixed-bpw <target>` is the per-tensor Oq4→Oq8 promoter
(`hipfire-quantize/src/main.rs:8472`), but it is threaded only into
`run_hfq_source_pipeline` (`:5817`, consumed at `:5886`) — the `.hfq` → `.hfq`
path. A safetensors directory or an `.hfa` archive goes through the source
pipeline, which never reads it. **No warning, no error, no diagnostic line.**

## Reproduce

Full recipe in `tests/tiny-moe-mixed-gate.sh`.

    hipfire-quantize --emit-fixture qwen3_5_moe_indexed --out $W/src --seed 42
    hipfire-quantize --input $W/src --output $W/anchor.fp16.hfq --format fp16 --arch-id 6
    tiny_quant_probe collect --arch qwen3_5_moe_indexed --model $W/anchor.fp16.hfq \
        --out $W/calib.hfq --len 128

    # SOURCE input: promoter silent, 64 uniform Oq4G256 experts
    hipfire-quantize --input $W/src --output $W/a.hfq --format oq4.25++ \
        --arch-id 6 --hessian $W/calib.hfq --mixed-bpw 4.25

    # HFQ input, otherwise identical:
    #   "mixed-bpw 4.5000: promoted 14 of 82 tensors to oq8++"
    hipfire-quantize --input $W/anchor.fp16.hfq --output $W/b.hfq --format oq4.25++ \
        --arch-id 6 --hessian $W/calib.hfq --mixed-bpw 4.5

Also note `oq4.25++` ALONE never produces a mixed artifact: that flag is
within-tensor magnitude tiering ("3 sparse W8 overlays/group, bulk W4"), a
different mechanism from per-tensor promotion. Confusing the two costs an
afternoon.

## `.hfa` input itself is fine

Worth stating plainly, because the above reads as "`.hfa` is broken" and it is
not. `hfa::is_hfa` (`main.rs:8821`) detects it, `HfaArchive::open` consumes it IN
PLACE (no 244 GB restore), and each shard's safetensors header is stored verbatim
so the tensor table, dtypes and `data_offsets` match a restored file's. Confirmed
on the 122B's own 180 GB archive:

    Architecture: qwen3_5_moe (id=6)
      MoE detected — will split 3D expert tensors per-expert before quantization.
    HFA input: 39 shard(s), read in place (no restore)

(`/srv/hipfire/models/qwen3-moe.hfa` fails, but on `model_type qwen3_moe` not
being registered — an arch-registration gap, not an archive one.)

## Consequence

A mixed-precision artifact CANNOT be built directly from the source archive. It
must go source → `.hfq` → re-quantize, and that path selects a different, larger
tensor set than the source path: on the tiny fixture it picked up
`mlp.shared_expert.down_proj [256, 128]`, K=128, which the runtime then refuses
outright — "256-wide FWHT rotation requires K % 256 == 0 and K > 0". So an
HFQ-re-quantized artifact can contain a tensor that hard-errors on first forward.

That specific arm (`HfqInputFormat::OqPlusCompact`) is guarded as of `224acb1cb`,
falling back to Q8 like its `Oq2`/`Oq3`/`Oq6`/`Mq3`/`Mq6` siblings. The `Oq4` and
`Oq8|Oq8Plus` arms still lack the same guard, and their comments claim ragged K
is zero-padded into a final 256-wide group — a claim the runtime check
contradicts. No failure was reproduced there, so they were left alone.

Better fix available: a K%128==0 tensor could take `OqPlusCompactG128` (qt 52)
and keep ~4 b/w instead of dropping to Q8. That needs the group to become a
per-tensor choice; today it arrives run-wide (`opus_group`).

## Fix

Either thread `mixed_bpw_target` into the source pipeline, or reject
`--mixed-bpw` loudly when the input is not an `.hfq`. The silent no-op is the
worst of the three.

**Done 2026-08-30: the refusal half.** `crates/hipfire-quantize/src/cli.rs`
rejects `--mixed-bpw` on a non-HFQ input, next to the identical
`--tensor-format` guard that was already there, and the error names the
`source -> .hfq -> re-quantize` route plus the `--mixed-bpw` vs `oq4.25++`
distinction this document warns about. The argument parser no longer swallows a
missing or unparseable value.

`tests/tiny-moe-mixed-gate.sh` asserts the refusal before its GPU steps, so a
regression to silence fails the gate. It uses `--format oq4` deliberately: the
`++` formats demand `--hessian` earlier in the same function, so an `oq4.25++`
probe would refuse for an unrelated reason and assert nothing.

**Still open: threading `mixed_bpw_target` into the source pipeline**, which is
what makes a mixed artifact buildable directly from `.hfa`/safetensors and
avoids the tensor-set difference (and the `K % 256` hard error) that the
`source -> .hfq -> re-quantize` detour introduces.
