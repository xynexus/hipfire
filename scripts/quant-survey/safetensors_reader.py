"""
bf16/f16/f32-aware safetensors reader for layer-by-layer streaming.

Lifted from docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/expert_absmax_stats.py
with these changes:
  - Generalized to handle dense + MoE checkpoints with both stacked-3D and
    split-2D expert tensor naming.
  - Returns f32 ndarrays (NOT f16/bf16) for all consumers; downstream
    quant_ops.quantize_mq4g256_fwht expects float32.
  - Layer-iterator API so we can stream large models without holding more
    than one layer's tensors at a time.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Direct safetensors file reader (bypasses the Python safetensors lib for bf16
# because safe_open(framework="numpy") raises "data type 'bfloat16' not
# understood" on bf16 tensors. We read the file format directly:
#   - First 8 bytes: u64 LE header size N
#   - Bytes 8..8+N: UTF-8 JSON {tensor_name: {dtype, shape, data_offsets: [s, e]}, "__metadata__": {...}}
#   - Bytes 8+N+s .. 8+N+e: raw tensor data
# ---------------------------------------------------------------------------

# Bytes per element by safetensors dtype string.
_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F16": 2, "BF16": 2, "U16": 2, "I16": 2,
    "F32": 4, "U32": 4, "I32": 4,
    "F64": 8, "U64": 8, "I64": 8,
}


@dataclass
class _ShardIndex:
    """Cached parse of one safetensors shard's header."""
    path: Path
    data_origin: int  # absolute byte offset where data section begins (8 + header_len)
    tensors: dict[str, dict]  # tensor_name -> {"dtype": str, "shape": list[int], "data_offsets": [s, e]}


_INDEX_CACHE: dict[Path, _ShardIndex] = {}


def _shard_index(path: Path) -> _ShardIndex:
    """Parse the header of a safetensors shard once. Cached by path."""
    cached = _INDEX_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    raw = json.loads(header_bytes.decode("utf-8"))
    tensors = {k: v for k, v in raw.items() if k != "__metadata__"}
    idx = _ShardIndex(path=path, data_origin=8 + header_size, tensors=tensors)
    _INDEX_CACHE[path] = idx
    return idx


def _read_tensor_bytes(idx: _ShardIndex, key: str) -> tuple[bytes, str, tuple[int, ...]]:
    """Pull raw bytes for a single tensor from its shard. Returns (bytes, dtype_str, shape)."""
    meta = idx.tensors[key]
    s, e = meta["data_offsets"]
    nbytes = e - s
    with open(idx.path, "rb") as f:
        f.seek(idx.data_origin + s)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise IOError(f"short read on {key} from {idx.path}: wanted {nbytes} got {len(raw)}")
    return raw, meta["dtype"], tuple(meta["shape"])


def _bytes_to_f32(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Cast raw safetensors bytes to a float32 numpy array of the given shape."""
    expected = int(np.prod(shape)) * _DTYPE_BYTES[dtype]
    if len(raw) != expected:
        raise IOError(f"size mismatch: dtype {dtype} shape {shape} expects {expected} bytes, got {len(raw)}")
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float32, copy=True)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "BF16":
        # bf16 is the top 16 bits of an f32; left-shift u16 by 16 and view as f32.
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        f32_bits = (u16 << 16).astype(np.uint32)
        return f32_bits.view(np.float32).reshape(shape).copy()
    if dtype == "F64":
        return np.frombuffer(raw, dtype=np.float64).astype(np.float32).reshape(shape)
    raise NotImplementedError(f"unsupported safetensors dtype for f32 cast: {dtype}")


def _list_tensor_keys(idx: _ShardIndex) -> list[str]:
    """All tensor keys in a shard."""
    return list(idx.tensors.keys())


def _read_tensor_f32(idx: _ShardIndex, key: str) -> np.ndarray:
    """Direct-read replacement for safe_open(framework='numpy').get_tensor(key).
       Works for bf16 (which safetensors numpy backend cannot handle).
    """
    raw, dtype, shape = _read_tensor_bytes(idx, key)
    return _bytes_to_f32(raw, dtype, shape)


# ---------------------------------------------------------------------------
# Snapshot resolution + tensor enumeration
# ---------------------------------------------------------------------------

def find_hf_snapshot(hf_cache: Path, repo: str) -> Path:
    """Find the largest snapshot under the HF cache for `repo` (e.g.
    'Qwen/Qwen3.5-9B'). Picks the snapshot dir with the most bytes.
    """
    repo_dir = hf_cache / f"models--{repo.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"no snapshots under {snapshots_dir}; is the model downloaded?")
    snaps = [s for s in snapshots_dir.iterdir() if s.is_dir() and any(s.iterdir())]
    if not snaps:
        raise FileNotFoundError(f"no populated snapshots under {snapshots_dir}")
    return max(snaps, key=lambda p: sum((p / f).stat().st_size for f in p.iterdir() if (p / f).is_file()))


def list_safetensors_shards(snapshot: Path) -> list[Path]:
    """Both naming conventions in the wild:
       - model-NNNNN-of-NNNNN.safetensors          (Qwen3.6, modern)
       - model.safetensors-NNNNN-of-NNNNN.safetensors  (Qwen3.5)
    Plus single-file 'model.safetensors' for small models. Returns sorted list.
    """
    shards = sorted(snapshot.glob("model-*.safetensors"))
    shards += sorted(snapshot.glob("model.safetensors-*.safetensors"))
    if not shards:
        single = snapshot / "model.safetensors"
        if single.exists():
            shards = [single]
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {snapshot}")
    return shards


def read_config(snapshot: Path) -> dict:
    """Load the model's config.json. Required to know n_layers, num_experts."""
    with open(snapshot / "config.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tensor name parsing
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


@dataclass(frozen=True)
class TensorRef:
    name: str           # full tensor name as stored in safetensors
    layer_idx: int      # extracted layer index, -1 if not a layer-bound tensor
    expert_idx: int     # extracted expert index, -1 if dense or non-MoE
    projection: str     # "q_proj"/"k_proj"/"v_proj"/"o_proj"/"gate_proj"/"up_proj"/"down_proj"/
                        #   "gate_up_proj"/"router"/"shared_expert.gate"/"shared_expert.up"/...
                        #   "" if not a projection-shape tensor
    is_stacked_3d: bool  # True if shape[0] is the expert axis (3D-packed MoE storage)


def _classify_projection(name: str) -> str:
    """Return a coarse projection label for outlier-survey grouping.

    Returns "" for non-projection tensors (norms, embeddings, etc.).
    Order matters: shared_expert.* and router checks come BEFORE the
    generic projection match because shared_expert.down_proj would
    otherwise classify as "down_proj".
    """
    last = name.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]
    last_seg = last.rsplit(".", 1)[-1]

    # 1. shared_expert.* takes precedence over plain projection names.
    if "shared_expert" in name and "shared_expert_gate" not in name:
        if last_seg == "gate":
            return "shared_expert.gate"
        if last_seg == "gate_proj":
            return "shared_expert.gate_proj"
        if last_seg == "up_proj":
            return "shared_expert.up_proj"
        if last_seg == "down_proj":
            return "shared_expert.down_proj"
    if last_seg == "shared_expert_gate":
        return "shared_expert_gate"

    # 2. router gate (`mlp.gate.weight` outside of any expert sub-tree).
    if last_seg == "gate" and ".mlp." in name and ".experts." not in name and ".shared_expert" not in name:
        return "router"

    # 3. Standard projections (dense + per-expert MoE).
    for p in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj", "gate_up_proj"):
        if last_seg == p:
            return p

    return ""


def parse_tensor_name(name: str) -> TensorRef:
    layer_match = _LAYER_RE.search(name)
    layer_idx = int(layer_match.group(1)) if layer_match else -1
    expert_match = _EXPERT_RE.search(name)
    expert_idx = int(expert_match.group(1)) if expert_match else -1
    projection = _classify_projection(name)
    # 3D-stacked detection: if name has ".experts." but NO numeric expert index
    # (e.g. "model.layers.5.mlp.experts.gate_up_proj.weight"), it's stacked.
    is_stacked_3d = ".experts." in name and expert_idx == -1
    return TensorRef(
        name=name,
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        projection=projection,
        is_stacked_3d=is_stacked_3d,
    )


def all_tensor_refs(snapshot: Path) -> list[TensorRef]:
    """Walk every shard and return TensorRefs for every tensor in the model.
    Cheap (header-only); does not load tensor data.
    """
    refs = []
    for shard in list_safetensors_shards(snapshot):
        idx = _shard_index(shard)
        for key in _list_tensor_keys(idx):
            refs.append(parse_tensor_name(key))
    return refs


# ---------------------------------------------------------------------------
# Streaming iteration
# ---------------------------------------------------------------------------

@dataclass
class TensorBatch:
    """One tensor's data alongside its parsed reference. Yielded by stream_tensors().

    For 3D-stacked MoE tensors (shape [n_experts, M, K]), `data` is the full
    3D array; the consumer is expected to iterate the leading axis to get
    per-expert 2D matrices.
    """
    ref: TensorRef
    data: np.ndarray
    shape: tuple[int, ...]
    dtype_str: str


def stream_tensors(snapshot: Path, projections: list[str] | None = None) -> Iterator[TensorBatch]:
    """Stream every tensor in the model one at a time as f32. Memory cost
    is O(largest tensor), not O(model). Tensors not matching `projections`
    (if provided) are skipped without loading.

    Order across shards is shard-stable but NOT sorted by layer; consumers
    that need layer-ordered output must collect-and-sort.
    """
    proj_set = set(projections) if projections else None
    for shard in list_safetensors_shards(snapshot):
        idx = _shard_index(shard)
        for key in _list_tensor_keys(idx):
            ref = parse_tensor_name(key)
            if proj_set is not None and ref.projection not in proj_set:
                continue
            data = _read_tensor_f32(idx, key)
            yield TensorBatch(
                ref=ref,
                data=data,
                shape=tuple(data.shape),
                dtype_str="float32 (cast from safetensors)",
            )


def stream_layer_tensors(snapshot: Path, projections: list[str] | None = None) -> Iterator[tuple[int, list[TensorBatch]]]:
    """Higher-level: yields (layer_idx, [TensorBatch...]) tuples in ASCENDING
    layer order. Streams ONE LAYER AT A TIME via a two-pass strategy:

    Pass 1 (cheap): enumerate `(shard_path, tensor_key, TensorRef)` for every
    tensor without loading data. Costs O(n_tensors) ref records.

    Pass 2 (per-layer load): for each layer in ascending index order, open
    each shard that holds at least one of that layer's tensors, load only
    those tensors as f32, and yield. Resident memory per yield is O(one
    layer's tensors) — bounded by the largest layer in the model, NOT by
    the model size. For a 122B-A10B with ~10 GB per layer, peak resident
    is O(10 GB) regardless of total checkpoint size.

    Tensors with layer_idx == -1 (embed_tokens, lm_head, model.norm) are
    yielded under layer_idx -1 first by virtue of sort order on the int
    keys.

    Caveat: this opens each shard multiple times (once per layer that
    intersects it). The OS page cache absorbs most of the cost, but
    expect ~N_LAYERS shard reopens for a checkpoint with N_LAYERS
    spanning all shards. Acceptable for the survey use case; if it
    becomes a hot path, switch to a shard-ordered single-pass variant
    that buffers the current layer until the layer index changes.
    """
    # Pass 1: enumerate without loading. Cache shard indices once.
    layer_to_locations: dict[int, list[tuple[Path, str, TensorRef]]] = {}
    proj_set = set(projections) if projections else None
    for shard in list_safetensors_shards(snapshot):
        idx = _shard_index(shard)
        for key in _list_tensor_keys(idx):
            ref = parse_tensor_name(key)
            if proj_set is not None and ref.projection not in proj_set:
                continue
            layer_to_locations.setdefault(ref.layer_idx, []).append((shard, key, ref))

    # Pass 2: per-layer load + yield. Group locations by shard within
    # each layer to minimize file reopens within a single yield.
    for layer_idx in sorted(layer_to_locations.keys()):
        batches: list[TensorBatch] = []
        by_shard: dict[Path, list[tuple[str, TensorRef]]] = {}
        for shard, key, ref in layer_to_locations[layer_idx]:
            by_shard.setdefault(shard, []).append((key, ref))
        for shard, items in by_shard.items():
            idx = _shard_index(shard)  # cached
            for key, ref in items:
                data = _read_tensor_f32(idx, key)
                batches.append(TensorBatch(
                    ref=ref,
                    data=data,
                    shape=tuple(data.shape),
                    dtype_str="float32 (cast from safetensors)",
                ))
        yield layer_idx, batches
        # Drop refs to data so GC can reclaim before the next layer loads.
        del batches
        del by_shard


# ---------------------------------------------------------------------------
# Self-test (light, requires no model)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Test parsers without needing a real safetensors file."""
    cases = [
        ("model.embed_tokens.weight", -1, -1, "", False),
        ("model.layers.5.self_attn.q_proj.weight", 5, -1, "q_proj", False),
        ("model.layers.12.mlp.down_proj.weight", 12, -1, "down_proj", False),
        ("model.layers.7.mlp.gate.weight", 7, -1, "router", False),
        ("model.layers.3.mlp.shared_expert.down_proj.weight", 3, -1, "shared_expert.down_proj", False),
        ("model.layers.3.mlp.shared_expert_gate.weight", 3, -1, "shared_expert_gate", False),
        ("model.layers.5.mlp.experts.42.gate_up_proj.weight", 5, 42, "gate_up_proj", False),
        ("model.layers.5.mlp.experts.42.down_proj.weight", 5, 42, "down_proj", False),
        ("model.layers.5.mlp.experts.gate_up_proj.weight", 5, -1, "gate_up_proj", True),
        ("model.layers.5.mlp.experts.down_proj.weight", 5, -1, "down_proj", True),
        ("model.norm.weight", -1, -1, "", False),
        ("lm_head.weight", -1, -1, "", False),
    ]
    fail = 0
    for name, want_layer, want_expert, want_proj, want_3d in cases:
        ref = parse_tensor_name(name)
        if (ref.layer_idx, ref.expert_idx, ref.projection, ref.is_stacked_3d) != \
           (want_layer, want_expert, want_proj, want_3d):
            print(f"  FAIL {name}: got {ref}, want layer={want_layer} "
                  f"expert={want_expert} proj={want_proj!r} 3d={want_3d}")
            fail += 1
    if fail:
        print(f"[self-test] {fail} parser cases failed")
        return 1
    print(f"[self-test] {len(cases)} parser cases PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
