#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""DSpark heads on the XDNA1 NPU (DFlash Phase E, steps 2-4).

Runs the DSpark head epilogue -- confidence head + markov head (embedding gather,
[vocab, rank] GEMV, argmax) + confidence-threshold truncation -- through the
proven int8 projection path (`oq_gemm_design.int_matmul`, per-row symmetric int8
exactly as `dflash_body_npu._proj_row` does it), and validates against the f32
golden in `dspark_ref.py`.

Three tiers are compared, which is what makes the gate honest:

  f32    -- `dspark_ref.run_heads`, the reference (weights dequantized to f32).
  int8   -- host simulation of the SAME per-row int8 quantization the NPU uses.
            Separates quantization error (expected, real) from kernel error
            (must be zero).
  npu    -- the NPU kernels. Must agree with `int8` bit-for-bit in the integer
            GEMM; any drift beyond f32 rescale rounding is a kernel bug.

Both a free-running chain (NPU argmax feeds the next slot's embedding gather,
the real inference behaviour) and a teacher-forced pass (each slot fed the
reference's `out_ids[i]`) are reported: free-running cascades a single argmax
flip into every later slot, so teacher-forced is what isolates per-slot error.

Shape constraints of the int8 GEMM kernel, both hit here:
  * `m % (4*r) == 0` with r=4, plus `M % m == 0` and `(M // m)` even -> the
    confidence head's `[1, 1280]` proj must be zero-padded to `[32, 1280]`
    (row 0 is the real one); 31/32 of its rows are waste.
  * `n % (2 * t) == 0` with t=8       -> the activation batch N must be a
    multiple of 16, so a single 1-row GEMV is padded to 16 columns and column 0
    is read back. 15/16 of the MACs are waste; see the cost note in `main`.

NOTE: this sidecar is qwen3-0.6b (hidden 1024), NOT the Qwen3.5-9B DFlash body
validated in Phases A-D. This validates the head kernels, not an end-to-end 9B
DSpark run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oq_gemm_design as design  # noqa: E402  (self-bootstraps the NPU env)
from dspark_ref import (  # noqa: E402
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_SIDECAR,
    DsparkHeads,
    rmsnorm,
    run_heads as run_heads_f32,
    synthetic_inputs,
)

# Activation batch width the int8 kernel accepts (n % (2*t) == 0, t = 8).
NPU_BATCH = 16
# Minimum padded row count for the A operand. The micro-kernel needs
# m % (4*r) == 0 with r=4, so m >= 16, and `_tiles_for` additionally needs
# M % m == 0 with (M // m) even -> M must be a multiple of 32.
CONF_PROJ_ROWS = 32


def quantize_row_symmetric(x_f32: np.ndarray, bits: int = 8):
    """Per-ROW symmetric int8 over the whole K. Identical to the body path's
    `dflash_body_npu.quantize_row_symmetric`."""
    qmax = (1 << (bits - 1)) - 1
    absmax = np.abs(x_f32).max(axis=1)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    q = np.round(x_f32 / scale[:, None]).clip(-qmax, qmax).astype(np.int8)
    return q, scale


@dataclass
class QuantWeights:
    """Per-row int8 head weights, plus their NPU-resident handles."""

    w2_q: np.ndarray  # int8 [vocab, rank]
    w2_s: np.ndarray  # f32  [vocab]
    conf_q: np.ndarray  # int8 [CONF_PROJ_ROWS, hidden+rank] (row 0 real)
    conf_s: np.ndarray  # f32  [CONF_PROJ_ROWS]
    w2_dev: tuple | None = None
    conf_dev: tuple | None = None


def quantize_heads(heads: DsparkHeads) -> QuantWeights:
    w2_q, w2_s = quantize_row_symmetric(heads.markov_w2.astype(np.float32))
    proj = np.zeros((CONF_PROJ_ROWS, heads.confidence_proj.shape[1]), np.float32)
    proj[0] = heads.confidence_proj[0]
    conf_q, conf_s = quantize_row_symmetric(proj)
    return QuantWeights(w2_q=w2_q, w2_s=w2_s, conf_q=conf_q, conf_s=conf_s)


def _pad_batch(vec: np.ndarray) -> np.ndarray:
    """[K] -> [NPU_BATCH, K] with the real row at index 0, rest zero."""
    out = np.zeros((NPU_BATCH, vec.shape[0]), np.float32)
    out[0] = vec
    return out


def gemv_int8_host(w_q, w_s, x: np.ndarray) -> np.ndarray:
    """Host int8 simulation of one GEMV: exactly what the NPU path computes."""
    qx, sx = quantize_row_symmetric(x.reshape(1, -1))
    acc = qx[0].astype(np.int64) @ w_q.astype(np.int64).T  # exact int
    return (w_s * float(sx[0])) * acc.astype(np.float32)


@dataclass
class Counters:
    dispatches: int = 0
    npu_seconds: float = 0.0
    per_op: dict = field(default_factory=dict)

    def record(self, op: str, seconds: float) -> None:
        self.dispatches += 1
        self.npu_seconds += seconds
        slot = self.per_op.setdefault(op, [0, 0.0])
        slot[0] += 1
        slot[1] += seconds


def gemv_int8_npu(dev, w_s, x: np.ndarray, counters: Counters, op: str) -> np.ndarray:
    """One NPU GEMV against a resident per-row-int8 weight.

    `dev` is `(A_t, M, K)` from `design.upload_int8`; the activation is padded to
    NPU_BATCH columns and column 0 is read back.
    """
    qx, sx = quantize_row_symmetric(_pad_batch(x.astype(np.float32)))
    A_t, M, K = dev
    start = time.perf_counter()
    C, _tile = design.matmul_npu_resident(A_t, M, K, qx)  # int32 [M, NPU_BATCH]
    counters.record(op, time.perf_counter() - start)
    return (w_s * float(sx[0])) * C[:, 0].astype(np.float32)


@dataclass
class HeadRun:
    out_ids: np.ndarray
    tokens: np.ndarray
    confidence: np.ndarray
    survival: np.ndarray
    confident_len: int
    bias_rows: np.ndarray  # [block, vocab] markov bias per slot (for cosine)
    counters: Counters


def run_heads_quant(
    heads: DsparkHeads,
    qw: QuantWeights,
    x_head: np.ndarray,
    logits: np.ndarray,
    prev_token: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    backend: str = "npu",
    conf_backend: str = "host",
    forced_ids: np.ndarray | None = None,
    markov_topk: int = 256,
) -> HeadRun:
    """The DSpark head loop with the markov (and optionally confidence) GEMV in
    int8 -- on the NPU (`backend="npu"`) or host-simulated (`backend="int8"`),
    or the CPU top-k shortlist (`backend="topk"`, f32, `markov_topk` candidates).

    `forced_ids`, when given, is the reference `out_ids` chain: each slot's
    embedding gather uses `forced_ids[i]` instead of this run's own argmax
    (teacher forcing), isolating per-slot error from the sequential cascade.
    `conf_backend` selects where the tiny 1x1280 confidence GEMV runs -- "f32",
    "int8", or "npu" -- so its accuracy cost and its dispatch cost can be
    measured separately from the markov head.
    """
    cfg = heads.cfg
    block, hidden = x_head.shape
    counters = Counters()
    normed = rmsnorm(x_head, heads.stage_norm, cfg.rms_norm_eps)

    out_ids = np.full(block + 1, prev_token, dtype=np.int64)
    confidence = np.zeros(block, dtype=np.float32)
    bias_rows = np.zeros((block, heads.vocab), np.float32)
    proj_f32 = heads.confidence_proj[0].astype(np.float32)
    bias_scalar = float(heads.confidence_bias[0])

    for i in range(block):
        token = int(forced_ids[i]) if forced_ids is not None else int(out_ids[i])
        emb = heads.markov_w1[token].astype(np.float32)  # embedding gather

        if cfg.enable_confidence:
            hidden_i = normed[i] if cfg.confidence_uses_normed else x_head[i]
            concat = np.concatenate([hidden_i.astype(np.float32), emb])
            if conf_backend == "f32":
                conf = float(np.dot(proj_f32, concat))
            elif conf_backend == "int8":
                conf = float(gemv_int8_host(qw.conf_q, qw.conf_s, concat)[0])
            else:
                conf = float(
                    gemv_int8_npu(qw.conf_dev, qw.conf_s, concat, counters, "confidence")[0]
                )
            confidence[i] = conf + bias_scalar

        if backend == "topk":
            # CPU top-k shortlist: the markov bias only ever decides an ARGMAX, so
            # compute it for just the top-`markov_topk` candidates of the base
            # logits instead of all 151936 rows. Exact whenever the winner is in
            # that shortlist — measured 100% (incl. free-running) down to k=16 with
            # the real trained markov_w2. Turns a 156 MB/slot streaming GEMV into a
            # k-row gather: 74.2 -> 8.7 ms/block on ONE CPU thread (vs 180 ms on the
            # NPU). The residual cost is the O(vocab) top-k SELECTION, not the dot —
            # if the target's sampler already yields a top-k, this drops to ~us.
            lg_i = logits[i].astype(np.float32)
            cand = np.argpartition(lg_i, -markov_topk)[-markov_topk:]
            bias_c = heads.markov_w2[cand].astype(np.float32) @ emb  # [k]
            bias_rows[i, cand] = bias_c  # only the shortlist is defined
            out_ids[i + 1] = int(cand[int(np.argmax(lg_i[cand] + bias_c))])
        else:
            if backend == "npu":
                bias_v = gemv_int8_npu(qw.w2_dev, qw.w2_s, emb, counters, "markov")
            else:
                bias_v = gemv_int8_host(qw.w2_q, qw.w2_s, emb)
            bias_rows[i] = bias_v
            out_ids[i + 1] = int(np.argmax(logits[i].astype(np.float32) + bias_v))

    survival = 1.0 / (1.0 + np.exp(-confidence.astype(np.float32)))
    confident_len = block
    for i in range(block):
        if survival[i] < conf_threshold:
            confident_len = i
            break
    confident_len = max(confident_len, 1)

    return HeadRun(
        out_ids=out_ids,
        tokens=out_ids[1:],
        confidence=confidence,
        survival=survival,
        confident_len=confident_len,
        bias_rows=bias_rows,
        counters=counters,
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if denom == 0 else float(np.dot(a, b) / denom)


def report_tier(name: str, ref, run: HeadRun, block: int) -> dict:
    """Compare one tier against the f32 reference and print the honest gate."""
    conf_cos = cosine(ref.confidence, run.confidence)
    conf_max = float(np.max(np.abs(ref.confidence - run.confidence)))
    bias_cos = [cosine(ref_row, run_row) for ref_row, run_row in zip(ref.bias_rows, run.bias_rows)] \
        if hasattr(ref, "bias_rows") else []
    id_match = int(np.sum(ref.tokens[:block] == run.tokens[:block]))
    flips = [i for i in range(block) if ref.tokens[i] != run.tokens[i]]
    trunc_ok = ref.confident_len == run.confident_len

    print(f"  [{name}]")
    print(f"    confidence cosine   {conf_cos:.9f}   max|delta| {conf_max:.3e}")
    if bias_cos:
        print(f"    markov bias cosine  min {min(bias_cos):.9f}  mean {np.mean(bias_cos):.9f}")
    print(f"    out_ids exact       {id_match}/{block}" + (f"  FLIPPED slots {flips}" if flips else ""))
    print(f"    truncation          ref={ref.confident_len} run={run.confident_len} "
          f"{'MATCH' if trunc_ok else 'MISMATCH'}")
    return dict(conf_cos=conf_cos, id_match=id_match, flips=flips, trunc_ok=trunc_ok)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 6, 7])
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.4, 0.45, DEFAULT_CONF_THRESHOLD],
        help="conf-threshold sweep; exercises several truncation points",
    )
    parser.add_argument("--conf-backend", default="host", choices=["f32", "int8", "npu"])
    parser.add_argument("--skip-npu", action="store_true", help="int8 host tiers only")
    args = parser.parse_args()

    heads = DsparkHeads(Path(args.sidecar))
    block = heads.cfg.block_size
    prev_token = heads.cfg.noise_token_id
    print(f"sidecar {args.sidecar}")
    print(f"hidden={heads.hidden} vocab={heads.vocab} rank={heads.cfg.markov_rank} "
          f"block={block} conf_uses_normed={heads.cfg.confidence_uses_normed}")

    t0 = time.perf_counter()
    qw = quantize_heads(heads)
    print(f"per-row int8 quantization of head weights: {time.perf_counter() - t0:.2f}s")

    if not args.skip_npu:
        t0 = time.perf_counter()
        qw.w2_dev = design.upload_int8(qw.w2_q)
        qw.conf_dev = design.upload_int8(qw.conf_q)
        print(f"NPU-resident upload (markov_w2 {qw.w2_q.nbytes / 1e6:.1f} MB): "
              f"{time.perf_counter() - t0:.2f}s")

    conf_backend = "f32" if args.conf_backend == "host" else args.conf_backend
    all_flips: list = []
    all_trunc_ok = True
    total = Counters()

    for seed in args.seeds:
        x_head, logits = synthetic_inputs(heads, seed)
        ref = run_heads_f32(heads, x_head, logits, prev_token)
        # Attach markov bias rows to the f32 reference for the cosine gate.
        ref.bias_rows = np.stack([
            heads.markov_w2.astype(np.float32) @ heads.markov_w1[ref.out_ids[i]].astype(np.float32)
            for i in range(block)
        ])

        print(f"\n=== seed {seed} ===")
        print(f"  reference out_ids {list(map(int, ref.tokens))}")

        sim = run_heads_quant(heads, qw, x_head, logits, prev_token,
                              backend="int8", conf_backend=conf_backend,
                              forced_ids=ref.out_ids)
        report_tier("int8 host sim, teacher-forced", ref, sim, block)

        if not args.skip_npu:
            npu_tf = run_heads_quant(heads, qw, x_head, logits, prev_token,
                                     backend="npu", conf_backend=conf_backend,
                                     forced_ids=ref.out_ids)
            r = report_tier("NPU, teacher-forced", ref, npu_tf, block)
            all_flips += [(seed, i) for i in r["flips"]]
            # NPU vs the int8 host sim: the integer GEMM must agree exactly.
            gemm_delta = float(np.max(np.abs(sim.bias_rows - npu_tf.bias_rows)))
            id_agree = int(np.sum(sim.tokens == npu_tf.tokens))
            print(f"    NPU vs int8 sim     max|delta| {gemm_delta:.3e}  "
                  f"out_ids {id_agree}/{block}"
                  f"{'  <- kernel bug' if id_agree != block else ''}")

            npu_free = run_heads_quant(heads, qw, x_head, logits, prev_token,
                                       backend="npu", conf_backend=conf_backend)
            report_tier("NPU, free-running chain", ref, npu_free, block)
            for op, (count, secs) in npu_free.counters.per_op.items():
                total.per_op.setdefault(op, [0, 0.0])
                total.per_op[op][0] += count
                total.per_op[op][1] += secs
                total.dispatches += count

            # Truncation gate across the threshold sweep (same confidence values,
            # several firing positions).
            for threshold in args.thresholds:
                def trunc(surv):
                    length = block
                    for i in range(block):
                        if surv[i] < threshold:
                            length = i
                            break
                    return max(length, 1)
                ref_len, npu_len = trunc(ref.survival), trunc(npu_tf.survival)
                ok = ref_len == npu_len
                all_trunc_ok &= ok
                print(f"    truncation @ {threshold:.2f}: ref={ref_len} npu={npu_len} "
                      f"{'ok' if ok else 'MISMATCH'}")

    if not args.skip_npu:
        print("\n=== NPU cost (free-running chain) ===")
        for op, (count, secs) in sorted(total.per_op.items()):
            print(f"  {op:12s} {count:4d} dispatches  {secs * 1e3:8.1f} ms total  "
                  f"{secs / count * 1e3:7.2f} ms/dispatch")
        print(f"  markov head is {block} dispatches/block "
              f"(sequential: out_ids[i+1] depends on slot i's argmax)")
        print(f"\nARGMAX FLIPS vs f32 reference: {all_flips if all_flips else 'none'}")
        print(f"TRUNCATION sweep: {'all match' if all_trunc_ok else 'MISMATCH present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
