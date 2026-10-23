"""Loader for `routing_trace_qwen4exp` output (format: the README beside the
results, `benchmarks/results/2026-10-20-qwen4exp-routing-trace/README.md`).

    from routing_trace import load
    tr = load("/path/to/trace")
    tr.routed[s, t, l]     # the k experts token t of stream s used at layer l
    tr.alt_ids[s, t, l]    # top_n experts by router prob (first k == routed)
    tr.alt_p[s, t, l]      # their softmax-over-all-experts probability
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Trace:
    layers: int
    k: int
    experts: int
    n_tok: int
    groups: list  # per stream
    ids: list  # per stream document id
    routed: np.ndarray  # u16 [S, T, L, k]
    alt_ids: Optional[np.ndarray]  # u16 [S, T, L, top_n]
    alt_p: Optional[np.ndarray]  # f16 [S, T, L, top_n]
    tokens: Optional[np.ndarray]  # u32 [S, T]


def load(path) -> Trace:
    d = Path(path)
    m = json.loads((d / "meta.json").read_text())
    L, K, E, T = m["layers"], m["k"], m["experts"], m["n_tok"]
    docs = [json.loads(l) for l in open(d / "docs.jsonl")]
    S = len(docs)

    def arr(name, dtype, *shape):
        f = d / name
        if not f.exists():  # traces from before alternatives/tokens were logged
            return None
        n = int(np.prod(shape))
        return np.fromfile(f, dtype, count=n).reshape(shape)

    N = m.get("top_n", K)
    return Trace(
        L, K, E, T,
        [x["group"] for x in docs],
        [x["id"] for x in docs],
        arr("routing.bin", np.uint16, S, T, L, K),
        arr("alt_ids.bin", np.uint16, S, T, L, N),
        arr("alt_p.bin", np.float16, S, T, L, N),
        arr("tokens.bin", np.uint32, S, T),
    )
