# R65 — compact W4 projection to BF16 attention staging

R65 keeps R63's byte-identical W4 compute functions, producer-native 8 KiB
activation records, and compact QKV `.rdna2.hfp`. Each completed 24x96 f32
accumulator tile is converted on-core into three padded 24x32 BF16 records.
Mutable output DMA scatters those records directly into the five-role raw
staging ABI already consumed by R29's headnorm/RoPE packers.

The raw record is 10 KiB: 4 KiB projected BF16 values, 4 KiB cos/sin, and
2 KiB norm/epsilon parameters. R65 overwrites only the projected-value prefix;
the runner seeds and verifies every attention-tail byte. There is no immutable
tensor-block reorder in the kernel. The local signed-nibble/lane handling stays
inside the unchanged R15 W4 compute function.

This is deliberately one bandwidth-first step short of fused headnorm/RoPE:
it proves the reduced-width shared-BO boundary independently. R66 can attach
the existing R29 pack loop to the same staging layout without returning f32 QKV
to the host.

```bash
export R63_QKV_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp"
./build_r65.sh
python r65_run.py
```

Acquire the repository `hipfire lock` before running on hardware.

## Result

Three locked fresh-process runs, each with two warmups and three timed
submissions, pass all 327,680 projected BF16 values bit-for-bit under the AIE
`floor` rounding mode. All 1,474,560 preseeded cos/sin/norm tail bytes remain
unchanged and every padding record remains zero.

| metric | median | range |
|---|---:|---:|
| raw-runtime NPU time | 0.487964 ms | 0.485649-0.490328 ms |
| host call | 0.551481 ms | 0.507620-0.567781 ms |
| useful projection rate | 1.0315 TOPS | 1.0265-1.0364 TOPS |

The largest R65 core text is 9,280 bytes, below the 16 KiB AIE2P program
limit. R66 should first validate the pack-only consumer and distribute K and V
across separate column sets if naive one-context fusion exceeds that limit.

The first oracle used NumPy's round-to-nearest BF16 cast and was rejected after
the first value differed by one BF16 ULP (`18.5` versus `18.625`). R15
deliberately leaves the AIE in `floor` mode; a bit-exact floor oracle passes.
The harness also does not rewrite the host mapping of the output BO between
warmed raw-runtime submissions: doing so exposed stale sentinel bytes on later
commands. The initial preseed plus final tail check still proves that the NPU
never writes the attention-tail region, and matches the resident-BO usage being
measured.

Durable rows:
`../results/r65-w4-bf16-raw-attention-20260713.csv`.
