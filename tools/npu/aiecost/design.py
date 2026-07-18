#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Design-space search: LLM problem -> ranked candidate schedules.

The model predicts a hand-written ScheduleSpec. That answers "how fast is this
schedule?" but not "which schedule should I build?", which is the question that
actually matters before writing a kernel. This turns an LLM-shaped problem into
the candidate schedules that could implement it, rejects the unbuildable ones
against live hardware limits, and ranks the rest.

Two problem shapes cover the large-LLM hot path:

  GemmProblem  — QKV / O / FFN projections. M tokens x K in-features x N out.
                 Weight dtype is the lever: C4 showed int8 x int4 is native at
                 512 MACs/VMAC vs 256 for int8 x int8, so OQ4/MQ4 weights get 2x
                 the compute rate — *if* the kernel is compute-bound at all.

  DecodeAttnProblem — one decode step against a KV cache. This is the
                 bandwidth-shaped half of the LLM: the whole cache is read once
                 per token, so KV bit-width is a direct multiplier on the bytes
                 that must cross the wire. hipfire's KVarN runs at 2/4/8 bits
                 (`kv.rs` asserts bits in {2,4,8}, head_dim in {128,256}).

The point of ranking both on one model: the ideal design differs by regime, and
the regime is not obvious in advance. A 2x compute win is worth nothing on a
feed-bound schedule, and halving KV bytes is worth nothing on a compute-bound
one. The model says which you are in.

Usage:
    python -m aiecost.design gemm --m 256 --k 768 --n 1280
    python -m aiecost.design gemm --m 256 --k 4096 --n 4096 --weight-bits 4
    python -m aiecost.design attn --context 4096 --kv-heads 8 --head-dim 128
    python -m aiecost.design kv-sweep --context 4096      # kvarn 2 vs 4 vs 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiecost import calib, model  # noqa: E402
from aiecost.spec import ScheduleSpec  # noqa: E402
from aiecost.target import resolve_target  # noqa: E402

# Native mmul shapes per operand-type pair, from C4's ISA probe. A shape that is
# not native costs extra accumulator registers for no throughput gain, so the
# search only enumerates shapes that fill their VMAC.
NATIVE_SHAPES = {
    ("int8", "int8"): [(4, 8, 8)],
    ("int8", "int4"): [(4, 16, 8)],
}

_BITS = {"int8": 8, "int4": 4, "bf16": 16}

# Core efficiency for a MULTI-CORE tiled GEMM: a 2-D SURFACE over output-tile
# area (m*n) and tile count per core. Both drive it: a bigger tile amortises the
# per-tile DMA + accumulator + acquire/release overhead; more tiles amortise the
# per-dispatch fill/drain. Fitted to a whole_array grid on npu1 (4 cols, i8,
# tile 32/48/64 x problem 128..2304 + 2048^3 anchors, 13 points):
#
#   eff = SURF_CMAX * area/(area + SURF_A) * n^2/(n^2 + SURF_N^2)
#
# The amortization is SQUARED in n: a dense low-tile-count sweep showed the drop
# is much steeper than n/(n+N) (which over-predicted +540% at n=4). Squared form:
# mean |err| 22%, max 80%, well-behaved for n>=16. tile 64 is the max buildable
# output tile for i32 (96/128 fail placement).
SURF_CMAX = 0.800
SURF_A = 9775.0
SURF_N = 23.5
GEMM_TILE = 64  # max buildable square output tile (i32 out); L1-capped on npu1
GEMM_K_TILE = 64  # k-reduction tile


def eff_surface(out_area: int, n_tiles_per_core: float) -> float:
    n = max(n_tiles_per_core, 1.0)
    return SURF_CMAX * (out_area / (out_area + SURF_A)) * (n * n / (n * n + SURF_N * SURF_N))

# KVarN record: codes pack 8/bits per byte, plus fp16 per-channel scale+zp and
# fp16 per-token s_col. Mirrors kvarn_record_bytes_bits() in hipfire-kvquant.
def kvarn_record_bytes(r_dim: int, c_dim: int, bits: int) -> int:
    cpb = 8 // bits
    return -(-(r_dim * c_dim) // cpb) + r_dim * 2 * 2 + c_dim * 2


@dataclass
class GemmProblem:
    """A projection: [M,K] activations x [K,N] weights -> [M,N]."""

    m: int
    k: int
    n: int
    dtype_a: str = "int8"
    dtype_b: str = "int8"
    out_bytes_per_elem: int = 4  # int32 accumulator

    @property
    def total_macs(self) -> int:
        return self.m * self.k * self.n

    def weight_bytes(self) -> int:
        return self.k * self.n * _BITS[self.dtype_b] // 8

    def act_bytes(self) -> int:
        return self.m * self.k * _BITS[self.dtype_a] // 8

    def candidates(self, target) -> list[ScheduleSpec]:
        out = []
        shapes = NATIVE_SHAPES.get((self.dtype_a, self.dtype_b), [])
        # Fixed max-buildable L1 tile; efficiency comes from the 2-D surface
        # (tile area x tile-count-per-core), so it is computed per column count.
        t_tile = GEMM_TILE
        out_area = t_tile * t_tile
        for cols in _column_options(target):
            cores = cols * (target.compute_cores // target.compute_columns)
            n_tiles_pc = (-(-self.m // t_tile) * -(-self.n // t_tile)) / cores
            eff = eff_surface(out_area, n_tiles_pc)
            for (mm, mk, mn) in shapes:
                macs_per_call = mm * mk * mn
                calls_total = -(-self.total_macs // macs_per_call)
                calls_per_core = -(-calls_total // cores)
                # Weights stream once; activations are broadcast to every column.
                wire = self.weight_bytes() + self.act_bytes() * cols
                # Real L1 tile footprint: A+B double-buffered + C accumulator.
                stage = int((t_tile * GEMM_K_TILE * _BITS[self.dtype_a] / 8 + GEMM_K_TILE * t_tile * _BITS[self.dtype_b] / 8) * 2
                            + t_tile * t_tile * self.out_bytes_per_elem)
                out.append(
                    ScheduleSpec(
                        name=f"gemm-{self.dtype_a}x{self.dtype_b}-c{cols}-t{t_tile}(eff{eff:.2f})",
                        columns=cols,
                        cores=cores,
                        wire_bytes_in=wire,
                        output_bytes=self.m * self.n * self.out_bytes_per_elem,
                        dma_tasks_live=max(1, self.k // mk),
                        bds_per_core=4,
                        locks_per_core=4,
                        fifo_depth=2,
                        mmul_calls_per_core=calls_per_core,
                        mmul_shape=(mm, mk, mn),
                        dtype_a=self.dtype_a,
                        dtype_b=self.dtype_b,
                        local_stage_bytes=stage,
                        core_efficiency=eff,
                        host_pack_bytes=self.act_bytes(),
                        host_deblock_bytes=self.m * self.n * self.out_bytes_per_elem,
                        n_bos=3,
                    )
                )
        return out


@dataclass
class DecodeAttnProblem:
    """One decode step, one layer, against a KVarN-quantized KV cache.

    Scores = q . K^T over head_dim, then out = scores . V. Both read the whole
    cache, so bytes scale with context x kv_heads x head_dim x bits.
    """

    context: int
    kv_heads: int = 8
    head_dim: int = 128
    kv_bits: int = 4
    dtype_a: str = "int8"

    def kv_bytes(self) -> int:
        # K and V, one KVarN record per (kv_head): [head_dim channels, context tokens]
        per_head = kvarn_record_bytes(self.head_dim, self.context, self.kv_bits)
        return 2 * self.kv_heads * per_head

    @property
    def total_macs(self) -> int:
        # q.K^T and scores.V, per head.
        return 2 * self.kv_heads * self.head_dim * self.context

    def candidates(self, target) -> list[ScheduleSpec]:
        # KVarN codes are 4-bit only when kv_bits == 4; 2-bit has no native MMUL
        # (C4 found mmul_8_4 is the narrowest family), so a 2-bit cache must be
        # widened to int4 or int8 in-core before it can feed a VMAC.
        dtype_b = "int4" if self.kv_bits == 4 else "int8"
        shapes = NATIVE_SHAPES.get((self.dtype_a, dtype_b), [])
        out = []
        for cols in _column_options(target):
            cores = cols * (target.compute_cores // target.compute_columns)
            for (mm, mk, mn) in shapes:
                macs_per_call = mm * mk * mn
                calls_total = -(-self.total_macs // macs_per_call)
                calls_per_core = -(-calls_total // cores)
                stage = (mm * mk + mk * mn) * 2 * 2 + mm * mn * 4
                out.append(
                    ScheduleSpec(
                        name=f"attn-kvarn{self.kv_bits}-ctx{self.context}-c{cols}",
                        columns=cols,
                        cores=cores,
                        wire_bytes_in=self.kv_bytes(),
                        output_bytes=self.kv_heads * self.head_dim * 2,  # bf16 out
                        dma_tasks_live=max(1, self.context // 64),
                        bds_per_core=4,
                        locks_per_core=4,
                        fifo_depth=2,
                        mmul_calls_per_core=calls_per_core,
                        mmul_shape=(mm, mk, mn),
                        dtype_a=self.dtype_a,
                        dtype_b=dtype_b,
                        local_stage_bytes=stage,
                        core_efficiency=eff_surface(GEMM_TILE * GEMM_TILE, 64),  # movement-bound; moot
                        host_pack_bytes=0,  # cache is already resident device-side
                        host_deblock_bytes=self.kv_heads * self.head_dim * 2,
                        n_bos=3,
                    )
                )
        return out


@dataclass
class DrafterProblem:
    """One draft-token forward through a DSpark/DFlash speculator body.

    The runtime DSpark drafter (crates/hipfire-arch-llama/src/dspark_body.rs:76)
    is a 5-layer dense Qwen3 block: dim=4096, FFN=12288, GQA 32 q / 8 kv x 128.
    Each draft token is one full forward at M=1 (autoregressive) — so it is the
    DECODE regime: weights stream once per token, arithmetic intensity ~1, and
    the schedule is movement-bound, not compute-bound.

    The design question this answers is not "how fast is one GEMM" but "does the
    WHOLE body fit a latency/energy budget on the NPU" — and the dominant lever
    turns out to be dispatch FUSION, because every op pays the C1 dispatch floor
    (~t_submit) and a 5-layer body is ~40 ops. Unfused, the floor alone is
    ~40 x 155 us = 6 ms/token; fused into one dispatch it is paid once. Energy
    excludes the floor (rate-dependent), so it sums cleanly either way — the
    fusion lever moves latency, not tok/J.
    """

    n_layers: int = 5
    dim: int = 4096
    inter: int = 12288
    n_heads: int = 32
    n_kv: int = 8
    head_dim: int = 128
    context: int = 512  # drafter block/context length (small)
    weight_bits: int = 8  # DSpark sidecar is Q8F16; 4 = OQ4/MQ4
    kv_bits: int = 4

    def _dtype_b(self) -> str:
        return "int4" if self.weight_bits == 4 else "int8"

    def layer_ops(self, target) -> list[tuple[str, ScheduleSpec]]:
        """The per-layer op list (one representative candidate each): 4 attn
        projections + 3 FFN projections + 1 block-attention step. M=1 decode."""
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv * self.head_dim
        db = self._dtype_b()
        projs = [
            ("q_proj", self.dim, q_dim),
            ("k_proj", self.dim, kv_dim),
            ("v_proj", self.dim, kv_dim),
            ("o_proj", q_dim, self.dim),
            ("gate_proj", self.dim, self.inter),
            ("up_proj", self.dim, self.inter),
            ("down_proj", self.inter, self.dim),
        ]
        # M=1 decode is feed-bound on weight bytes, so a real drafter kernel is
        # designed to saturate the feed: use all shim input streams (8 => 30.8
        # GB/s), not the one-per-column default (4 => 15.4 GB/s). This is the
        # single biggest lever after weight bit-width.
        shim = _shim_streams(target)
        ops: list[tuple[str, ScheduleSpec]] = []
        for label, k, n in projs:
            # Fastest candidate = the one the model would pick; use full columns.
            cand = _fastest(GemmProblem(1, k, n, "int8", db).candidates(target))
            if cand:
                cand.feed_streams = shim
                # The tiled-GEMM efficiency surface is fit on square 64x64 output
                # tiles; an M=1 GEMV has a 1x64 output strip, so the surface
                # collapses to ~0.0004 and inflates a ~1 us compute to ~2 ms. That
                # is wrong: decode issues its few MACs back-to-back and then waits
                # on weight streaming. Model it at saturated efficiency so t_feed
                # (the real, physical limiter for M=1) dominates.
                cand.core_efficiency = 1.0
                ops.append((label, cand))
        attn = _fastest(DecodeAttnProblem(self.context, self.n_kv, self.head_dim, self.kv_bits).candidates(target))
        if attn:
            ops.append(("block_attn", attn))
        return ops


def _fastest(cands: list[ScheduleSpec]) -> ScheduleSpec | None:
    """The candidate a speed-ranked search would build (most columns first)."""
    return max(cands, key=lambda s: s.columns) if cands else None


def _shim_streams(target) -> int:
    """Concurrent shim input streams (feed ceiling) for the target; 8 on npu1."""
    try:
        dev = target.iron_device()
        shims = list(dev.get_shim_tiles())
        return dev.get_num_connections(shims[0], False) * len(shims)
    except Exception:
        return 8


def _column_options(target) -> list[int]:
    opts, c = [], 1
    while c <= target.compute_columns:
        opts.append(c)
        c *= 2
    if target.compute_columns not in opts:
        opts.append(target.compute_columns)
    return opts


# Differences below this are not real: E1's energy marginals reproduce to ~±6-9%,
# and the device-span gate is ±30%. Treating a 1% gap as a trade-off would claim a
# distinction the model cannot support — the same error the validator made when it
# scored predicted TIES as misorderings.
TOL = 0.02


def _better(a: float, b: float) -> bool:
    """a beats b by more than tolerance."""
    return a < b * (1.0 - TOL)


def _worse(a: float, b: float) -> bool:
    return a > b * (1.0 + TOL)


def pareto_front(rows: list[tuple]) -> set[str]:
    """Names of candidates not dominated on BOTH time and energy, within tolerance.

    A candidate is dominated if another is no worse on either axis and clearly
    better on one. Everything surviving is a real trade — a policy call (latency
    vs battery) the model cannot settle. Ranking on time alone hides exactly
    these; but calling a sub-tolerance difference a trade invents one.
    """
    front = set()
    for s, p in rows:
        dominated = any(
            not _worse(q.device_s, p.device_s)
            and not _worse(q.energy_j, p.energy_j)
            and (_better(q.device_s, p.device_s) or _better(q.energy_j, p.energy_j))
            for t, q in rows
            if t.name != s.name
        )
        if not dominated:
            front.add(s.name)
    return front


def rank_and_render(specs: list[ScheduleSpec], key: str, device: str, header: str,
                    objective: str = "pareto") -> str:
    rows = [(s, model.predict(s, key, device)) for s in specs]
    ok = [(s, p) for s, p in rows if p.buildable and p.admissible]
    bad = [(s, p) for s, p in rows if not (p.buildable and p.admissible)]

    lines = [header, ""]
    if not ok:
        lines.append("  no buildable+admissible candidate:")
        for s, p in bad[:3]:
            lines.append(f"    {s.name}: {'; '.join(p.build_errors or p.missing)}")
        return "\n".join(lines)

    key_fn = {"speed": lambda sp: sp[1].device_s, "energy": lambda sp: sp[1].energy_j}.get(
        objective, lambda sp: sp[1].device_s
    )
    ok.sort(key=key_fn)
    front = pareto_front(ok)
    fastest = min(ok, key=lambda sp: sp[1].device_s)
    cheapest = min(ok, key=lambda sp: sp[1].energy_j)

    lines.append(f"  {'candidate':<38} {'device':>10} {'energy':>9} {'limiter':>9} {'AI':>6} {'E-bound':>9}  flags")
    for s, p in ok:
        ebound = "movement" if p.energy_terms.get("movement", 0) > p.energy_terms.get("compute", 0) else "compute"
        flags = []
        if s.name == fastest[0].name:
            flags.append("FASTEST")
        if s.name == cheapest[0].name:
            flags.append("LOWEST-E")
        if s.name not in front:
            flags.append("dominated")
        lines.append(f"  {s.name:<38} {p.device_s * 1e6:>9.1f}u {p.energy_j * 1e3:>8.3f}m {p.limiter:>9} "
                     f"{p.arithmetic_intensity:>6.1f} {ebound:>9}  {' '.join(flags)}")
    if bad:
        lines.append("")
        for s, p in bad:
            why = p.build_errors[0] if p.build_errors else f"uncalibrated: {p.missing[0]}"
            lines.append(f"  REJECTED {s.name}: {why}")

    # The trade, stated explicitly — but only if there IS one. Ranking on a single
    # objective buries a real trade; announcing a sub-tolerance one invents it.
    lines.append("")
    dt = cheapest[1].device_s / fastest[1].device_s
    de = fastest[1].energy_j / cheapest[1].energy_j if cheapest[1].energy_j else 1.0
    if fastest[0].name == cheapest[0].name:
        lines.append(f"  NO TRADE: {fastest[0].name} is both fastest and lowest-energy.")
    elif de <= 1.0 + TOL:
        lines.append(f"  NO MEANINGFUL TRADE: the lowest-energy candidate saves only {(de - 1) * 100:.1f}% "
                     f"(under the {TOL * 100:.0f}% tolerance) while costing {dt:.2f}x the time.")
        lines.append(f"  => take the fastest, {fastest[0].name}. Energy is flat across these candidates.")
    else:
        lines.append("  TRADE-OFF — the objectives pick DIFFERENT schedules:")
        lines.append(f"    tok/s -> {fastest[0].name:<32} {fastest[1].device_s * 1e6:8.1f} us  {fastest[1].energy_j * 1e3:7.3f} mJ")
        lines.append(f"    tok/J -> {cheapest[0].name:<32} {cheapest[1].device_s * 1e6:8.1f} us  {cheapest[1].energy_j * 1e3:7.3f} mJ")
        lines.append(f"    cost of choosing tok/J: {dt:.2f}x slower for {de:.2f}x less energy")
    lines.append(f"  Pareto front ({len(front)} of {len(ok)}, {TOL * 100:.0f}% tol): {', '.join(sorted(front))}")

    best_s, best_p = ok[0]
    lines.append("")
    lines.append(f"  breakdown of the {objective}-optimal candidate: {best_s.name}")
    for k, v in sorted(best_p.terms.items(), key=lambda kv: -kv[1]):
        mark = "  <== limiter" if k == best_p.limiter else ""
        lines.append(f"    {k:<10} {v * 1e6:9.2f} us{mark}")
    for a in best_p.advice:
        lines.append(f"    ADVICE: {a}")
    return "\n".join(lines)


def render_drafter(prob: DrafterProblem, target, key: str, device: str,
                   draft_len: int, gpu_verify_us: float | None) -> str:
    ops = prob.layer_ops(target)
    preds = [(label, model.predict(s, key, device)) for label, s in ops]
    bad = [(label, p) for label, p in preds if not (p.buildable and p.admissible)]

    lines = []
    lines.append(f"DSpark/DFlash drafter body on {target.key} — {prob.n_layers}-layer, dim={prob.dim}, "
                 f"FFN={prob.inter}, GQA {prob.n_heads}/{prob.n_kv}x{prob.head_dim}")
    lines.append(f"  int8 x int{prob.weight_bits} weights, kvarn{prob.kv_bits}, ctx={prob.context}, "
                 f"M=1 decode (one draft token = one full forward)")
    lines.append("")
    if bad:
        lines.append("  NON-ADMISSIBLE ops (cannot model the body):")
        for label, p in bad:
            why = (p.build_errors or p.missing or ["?"])[0]
            lines.append(f"    {label}: {why}")
        return "\n".join(lines)

    # Per-op: device_s already includes one dispatch floor (t_submit). Split it
    # out so we can model FUSION — a fused kernel pays the floor once, not per op.
    lines.append(f"  {'op':<12} {'device':>10} {'work':>10} {'floor':>9} {'energy':>9} {'limiter':>9} {'AI':>6}")
    work_sum = 0.0
    floor_max = 0.0
    energy_sum = 0.0
    for label, p in preds:
        floor = p.terms.get("t_submit", 0.0)
        work = p.device_s - floor
        work_sum += work
        floor_max = max(floor_max, floor)
        energy_sum += p.energy_j
        eb = "mv" if p.energy_terms.get("movement", 0) > p.energy_terms.get("compute", 0) else "cmp"
        lines.append(f"  {label:<12} {p.device_s * 1e6:>9.1f}u {work * 1e6:>9.1f}u {floor * 1e6:>8.1f}u "
                     f"{p.energy_j * 1e3:>8.4f}m {p.limiter:>7}/{eb} {p.arithmetic_intensity:>6.1f}")

    n_ops = len(preds) * prob.n_layers
    unfused_tok = sum(p.device_s for _, p in preds) * prob.n_layers
    # Fused-per-layer: one floor per layer dispatch; work is sequential (data-dep).
    fused_layer_tok = (floor_max + work_sum) * prob.n_layers
    # Fused-whole-body: one floor for the entire forward.
    fused_body_tok = floor_max + work_sum * prob.n_layers
    energy_tok = energy_sum * prob.n_layers  # floor excluded from energy — fusion-invariant

    lines.append("")
    lines.append(f"  per DRAFT TOKEN ({prob.n_layers} layers x {len(preds)} ops = {n_ops} ops):")

    def _row(name: str, t: float) -> str:
        toks = 1.0 / t if t else 0.0
        toks_j = 1.0 / energy_tok if energy_tok else 0.0
        return (f"    {name:<22} {t * 1e6:>10.1f} us/tok  {toks:>8.0f} tok/s   "
                f"{energy_tok * 1e3:>8.3f} mJ/tok  {toks_j:>7.0f} tok/J")

    lines.append(_row("unfused (floor/op)", unfused_tok))
    lines.append(_row("fused per-layer", fused_layer_tok))
    lines.append(_row("fused whole-body", fused_body_tok))
    floor_tax = (unfused_tok - fused_body_tok) / unfused_tok * 100 if unfused_tok else 0.0
    lines.append(f"  => dispatch-floor tax, unfused vs fused-body: {floor_tax:.0f}% of latency is the "
                 f"C1 floor paid {n_ops}x. Energy is identical (floor excluded).")

    lines.append("")
    lines.append(f"  speculative window (draft_len={draft_len}, autoregressive, fused-body):")
    window_s = fused_body_tok * draft_len
    window_j = energy_tok * draft_len
    lines.append(f"    {window_s * 1e6:>10.1f} us   {window_j * 1e3:>8.3f} mJ  to draft {draft_len} tokens")
    if gpu_verify_us is not None:
        verify_s = gpu_verify_us * 1e-6
        hidden = window_s <= verify_s
        lines.append(f"    GPU verify budget (input): {gpu_verify_us:.0f} us for the {draft_len + 1}-token verify pass")
        if hidden:
            lines.append(f"    => NPU draft ({window_s * 1e6:.0f} us) FITS under the verify pass: drafting is FREE "
                         "(fully hidden), and it runs on the NPU's energy budget, not the GPU's.")
        else:
            over = window_s / verify_s
            lines.append(f"    => NPU draft ({window_s * 1e6:.0f} us) EXCEEDS the verify pass by {over:.2f}x: it is "
                         "on the critical path. Shorten draft_len, fuse harder, or the split loses to GPU-only draft.")
        lines.append("    (GPU verify time is an INPUT — design.py has no GPU calibration; supply it from a GPU bench.)")

    lines.append("")
    lines.append("  caveats: M=1 GEMMs charge only USEFUL macs (tile padding for a 1-row output is not")
    lines.append("  modelled, but decode is movement-bound so t_core is not the limiter anyway); LM/markov")
    lines.append("  /confidence heads and embed lookup are excluded (small vs the body); fusion model treats")
    lines.append("  per-op work as sequential (data-dependent q->attn->o->ffn) sharing one dispatch floor.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--objective", default="pareto", choices=["pareto", "speed", "energy"],
                   help="which objective to sort by; the trade-off and Pareto front are always reported")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gemm", help="rank schedules for a projection GEMM")
    g.add_argument("--m", type=int, default=256)
    g.add_argument("--k", type=int, default=768)
    g.add_argument("--n", type=int, default=1280)
    g.add_argument("--weight-bits", type=int, default=8, choices=[4, 8])

    a = sub.add_parser("attn", help="rank schedules for one decode-attention layer")
    a.add_argument("--context", type=int, default=4096)
    a.add_argument("--kv-heads", type=int, default=8)
    a.add_argument("--head-dim", type=int, default=128, choices=[128, 256])
    a.add_argument("--kv-bits", type=int, default=4, choices=[2, 4, 8])

    b = sub.add_parser("batch-sweep", help="batch size vs tok/s and tok/J — where the energy regime flips")
    b.add_argument("--k", type=int, default=768)
    b.add_argument("--n", type=int, default=1280)
    b.add_argument("--weight-bits", type=int, default=8, choices=[4, 8])
    b.add_argument("--out-bits", type=int, default=32, choices=[16, 32],
                   help="accumulator output width. 32 = int32; 16 = bf16. This is a LEVER: the output\nscales with batch and comes to dominate the byte count, capping arithmetic intensity.")
    b.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256, 512])

    s = sub.add_parser("kv-sweep", help="KVarN 2/4/8-bit head-to-head on decode attention")
    s.add_argument("--context", type=int, default=4096)
    s.add_argument("--kv-heads", type=int, default=8)
    s.add_argument("--head-dim", type=int, default=128, choices=[128, 256])
    s.add_argument("--layers", type=int, default=32, help="scale the per-layer answer to a whole model")

    d = sub.add_parser("drafter", help="DSpark/DFlash speculator body — per-draft-token latency, tok/J, fusion tax")
    d.add_argument("--layers", type=int, default=5, help="drafter body layers (DSpark runtime = 5)")
    d.add_argument("--dim", type=int, default=4096)
    d.add_argument("--inter", type=int, default=12288)
    d.add_argument("--heads", type=int, default=32)
    d.add_argument("--kv-heads", type=int, default=8)
    d.add_argument("--head-dim", type=int, default=128, help="drafter head_dim (tiny config uses 64)")
    d.add_argument("--context", type=int, default=512, help="drafter block/context length")
    d.add_argument("--weight-bits", type=int, default=8, choices=[4, 8], help="8 = DSpark Q8F16 sidecar; 4 = OQ4/MQ4")
    d.add_argument("--kv-bits", type=int, default=4, choices=[2, 4, 8])
    d.add_argument("--draft-len", type=int, default=4, help="speculative tokens drafted per verify cycle")
    d.add_argument("--gpu-verify-us", type=float, default=None,
                   help="GPU verify-pass time (us) to test the NPU-draft||GPU-verify pipelining condition")

    args = p.parse_args()
    target = resolve_target(args.device)
    key = calib.current_key(args.device) if _key_takes_device() else calib.current_key()

    if args.cmd == "gemm":
        prob = GemmProblem(args.m, args.k, args.n, "int8", "int4" if args.weight_bits == 4 else "int8")
        hdr = (f"GEMM {args.m}x{args.k}x{args.n} int8 x int{args.weight_bits} on {target.key} "
               f"({target.compute_columns} cols, {target.compute_cores} cores)\n"
               f"  {prob.total_macs / 1e6:.1f} M MACs, weights {prob.weight_bytes() / 1024:.0f} KiB")
        print(rank_and_render(prob.candidates(target), key, args.device, hdr, args.objective))
        return 0

    if args.cmd == "attn":
        prob = DecodeAttnProblem(args.context, args.kv_heads, args.head_dim, args.kv_bits)
        hdr = (f"decode attention, 1 layer, ctx={args.context} kv_heads={args.kv_heads} "
               f"head_dim={args.head_dim} kvarn{args.kv_bits} on {target.key}\n"
               f"  KV bytes/layer {prob.kv_bytes() / 1024:.0f} KiB, {prob.total_macs / 1e6:.1f} M MACs")
        print(rank_and_render(prob.candidates(target), key, args.device, hdr, args.objective))
        return 0

    if args.cmd == "batch-sweep":
        # Batching is the lever that crosses the energy break-even. For a
        # projection, weights are read ONCE and reused for every token in the
        # batch, so arithmetic intensity ~= M. At M=1 a GEMM is pure movement; at
        # M > ~183 (int8) arithmetic finally dominates its energy.
        #
        # NOTE the asymmetry that decides LLM design: batching amortises WEIGHTS
        # but NOT the KV cache — every sequence carries its own KV, so attention
        # stays movement-bound at any batch size while the projections do not.
        ratio_name = f"byte_mac_energy_ratio_int8_{'int4' if args.weight_bits == 4 else 'int8'}"
        consts = calib.load(key)
        breakeven = consts[ratio_name].value if ratio_name in consts else None
        print(f"batch sweep — {args.k}x{args.n} projection, int8 x int{args.weight_bits}, "
              f"{'bf16' if args.out_bits == 16 else 'int32'} out, on {target.key}")
        if breakeven:
            print(f"  energy break-even: AI > {breakeven:.0f} MACs/byte before arithmetic dominates\n")
        print(f"  {'batch':>6} {'AI':>7} {'E-bound':>9} {'device':>10} {'energy':>9} "
              f"{'tok/s':>9} {'tok/J':>8} {'us/tok':>8} {'uJ/tok':>8}")
        base = None
        for m in args.batches:
            prob = GemmProblem(m, args.k, args.n, "int8", "int4" if args.weight_bits == 4 else "int8",
                               out_bytes_per_elem=args.out_bits // 8)
            rows = [(s_, model.predict(s_, key, args.device)) for s_ in prob.candidates(target)]
            ok = sorted([r for r in rows if r[1].buildable and r[1].admissible], key=lambda sp: sp[1].device_s)
            if not ok:
                print(f"  {m:>6}  no admissible candidate")
                continue
            _, p = ok[0]  # fastest; per-token metrics are what serving cares about
            eb = "movement" if p.energy_terms.get("movement", 0) > p.energy_terms.get("compute", 0) else "compute"
            tok_s = m / p.device_s
            tok_j = m / p.energy_j if p.energy_j else 0.0
            # base is the FIRST batch swept, not necessarily 1 — label it honestly.
            gain = f"  ({tok_j / base:.1f}x tok/J vs B={args.batches[0]})" if base else ""
            print(f"  {m:>6} {p.arithmetic_intensity:>7.1f} {eb:>9} {p.device_s * 1e6:>9.1f}u "
                  f"{p.energy_j * 1e3:>8.3f}m {tok_s:>9.0f} {tok_j:>8.0f} "
                  f"{p.device_s / m * 1e6:>8.2f} {p.energy_j / m * 1e6:>8.2f}{gain}")
            if base is None:
                base = tok_j
        print("\n  Batching amortises WEIGHTS, so per-token cost falls hard on both axes. But it does")
        print("  NOT amortise the OUTPUT: that scales with the batch and comes to dominate the byte")
        print("  count (80% at B=2048, int32, 1 col), capping AI. Batching ALONE cannot cross the")
        print("  break-even — only bf16 output + minimal activation replication gets there.")
        print("  It also does NOT amortise the KV cache — each sequence has its own — so attention")
        print("  stays movement-bound at any batch size. Batch the projections; the KV read is irreducible.")
        return 0

    if args.cmd == "drafter":
        prob = DrafterProblem(
            n_layers=args.layers, dim=args.dim, inter=args.inter, n_heads=args.heads,
            n_kv=args.kv_heads, head_dim=args.head_dim, context=args.context,
            weight_bits=args.weight_bits, kv_bits=args.kv_bits,
        )
        print(render_drafter(prob, target, key, args.device, args.draft_len, args.gpu_verify_us))
        return 0

    # kv-sweep: the KVarN question, answered per-layer then scaled to the model.
    print(f"KVarN bit-width sweep — decode attention on {target.key}")
    print(f"  ctx={args.context} kv_heads={args.kv_heads} head_dim={args.head_dim} layers={args.layers}\n")
    print(f"  {'kvarn':>6} {'KV KiB/layer':>13} {'device/layer':>13} {'limiter':>9} {'model ms/token':>15} {'tok/s':>8}")
    base = None
    for bits in (8, 4, 2):
        prob = DecodeAttnProblem(args.context, args.kv_heads, args.head_dim, bits)
        cands = prob.candidates(target)
        rows = [(s, model.predict(s, key, args.device)) for s in cands]
        ok = sorted([r for r in rows if r[1].buildable and r[1].admissible], key=lambda sp: sp[1].device_s)
        if not ok:
            print(f"  {bits:>6}  no admissible candidate")
            continue
        s, pr = ok[0]
        per_model = pr.device_s * args.layers
        tok_s = 1.0 / per_model if per_model else 0.0
        speed = f"  ({base / pr.device_s:.2f}x vs kvarn8)" if base else ""
        print(f"  {bits:>6} {prob.kv_bytes() / 1024:>13.0f} {pr.device_s * 1e6:>12.1f}u {pr.limiter:>9} "
              f"{per_model * 1e3:>15.2f} {tok_s:>8.1f}{speed}")
        if bits == 8:
            base = pr.device_s
    print("\n  Caveat: attention only. Excludes the projections, and assumes the KV cache is")
    print("  already device-resident (host_pack_bytes=0). Per-layer dispatch pays the C1 floor")
    print("  once per layer — fusing layers into one dispatch is a separate lever.")
    return 0


def _key_takes_device() -> bool:
    import inspect

    return "device" in inspect.signature(calib.current_key).parameters


if __name__ == "__main__":
    sys.exit(main())
