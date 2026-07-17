# Qwen3 embedding image ABI

The Rust runtime and compiled AIE2P images meet at
`hipfire.full_embedding_encoder.v1`. An image is admitted only when its
`manifest.json` has both this ABI and an exact cache key match.

## Cache layout

Images live under:

```text
$HIPFIRE_NPU_IMAGE_CACHE/embedding/
  aie2p-qwen3-h1024-l28-qh16-kvh8-d128-i3072-oq8+-s512-b8/
    manifest.json
    final.xclbin
    insts.bin
```

If `HIPFIRE_NPU_IMAGE_CACHE` is unset, the root defaults to
`~/.cache/hipfire/npu`. There is no nearest-shape, nearest-batch, legacy-name,
or GPU fallback for an NPU-only artifact.

```json
{
  "schema": "hipfire.npu_embedding_image.v1",
  "runtime_abi": "hipfire.full_embedding_encoder.v1",
  "key": {
    "npu_architecture": "aie2p",
    "model_geometry": {
      "architecture": "qwen3",
      "hidden_size": 1024,
      "num_hidden_layers": 28,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 3072
    },
    "quant_format": "oq8+",
    "sequence_bucket": 512,
    "dispatch_batch": 8
  },
  "xclbin": "final.xclbin",
  "instructions": "insts.bin"
}
```

## Kernel arguments

The command packet passes four buffers in this order:

1. padded BF16 hidden rows shaped `[batch, bucket, hidden]`;
2. the resident encoder-weight blob described below;
3. little-endian `u32` real token lengths shaped `[batch]`;
4. final F32 embeddings shaped `[batch, output_dimensions]`.

The host performs tokenization and embedding-table lookup. The image owns all
transformer layers, final RMSNorm, last-real-token pooling, and L2
normalization. Real lengths are semantic: the graph must use them to block
padding and cross-document attention. Qwen attention is causal within each
segment. A padded row must never be selected for pooling.

## Encoder-weight blob

The little-endian blob begins with a 64-byte header:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `[u8; 8]` | `HFENCB01` |
| 8 | `u32` | version, currently 1 |
| 12 | `u32` | descriptor count |
| 16..40 | `6 * u32` | hidden, layers, query heads, KV heads, head dim, intermediate |

Descriptors begin at byte 64 and are 48 bytes each:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `u16` | role |
| 4 | `u32` | layer, or `u32::MAX` for the final norm |
| 8 | `u32` | HFQ quant type |
| 12 | `u32` | rank |
| 16,20 | `u32` | first two logical dimensions |
| 24 | `u32` | quant group size |
| 32 | `u64` | aligned payload offset |
| 40 | `u64` | payload byte length |

Payloads start at the next 64-byte boundary and each payload is padded to a
64-byte boundary. Roles are:

| Role | Tensor |
|---:|---|
| 1 | final RMSNorm |
| 2 | input RMSNorm |
| 3,4,5,6 | Q, K, V, output projections |
| 7,8 | per-head Q/K RMSNorm |
| 9 | post-attention RMSNorm |
| 10,11,12 | gate, up, down projections |

An AWQ scale sidecar uses the corresponding projection role ORed with
`0x8000`. Projection payloads must be OQ8 qt=35 or row-padded OQ8 qt=43;
vectors are F16, F32, or BF16. The blob never contains `lm_head.weight`, KV
state, or generation scratch.

The 0.6B geometry is the first admission target. The 4B and 8B models use the
same ABI and differ only through the exact geometry key and blob contents.

## Segmented-attention component image

The independently verifiable causal-attention component uses schema
`hipfire.npu_segmented_attention_image.v1`. It is not accepted by the full
encoder resolver above. Build it with:

```text
python3 tools/npu/build_qwen3_segmented_attention.py \
  --bucket 128 --batch 2 --query-heads 16 --output /tmp/qwen3-attention
```

Its three command arguments are packed BF16 Q with replicated real-length
trailers, packed BF16 K/V, and packed BF16 output. The same layout supports 16
or 32 query heads over eight KV heads at head dimension 128. The Rust
`NpuSegmentedAttention` wrapper owns canonical-to-physical packing and output
unpacking. This component image proves causal masking, padding exclusion, and
document isolation, but it must not be installed or reported as a
`hipfire.full_embedding_encoder.v1` image.
