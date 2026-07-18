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
# tile 32/48/64 x problem 768/1536/2304 + 2048^3 anchors):
#
#   eff = SURF_CMAX * area/(area + SURF_A) * n/(n + SURF_N)
#
# mean |err| 20% overall, ~10% in the design-relevant large-problem regime; it
# over-predicts at very low tile count (n<40). tile 64 is the max buildable
# output tile for i32 output (96/128 fail placement). This SUPERSEDES the earlier
# flat 0.209, which was right only at one operating point.
SURF_CMAX = 0.600
SURF_A = 5391.0
SURF_N = 23.3
GEMM_TILE = 64  # max buildable square output tile (i32 out); L1-capped on npu1
GEMM_K_TILE = 64  # k-reduction tile


def eff_surface(out_area: int, n_tiles_per_core: float) -> float:
    n = max(n_tiles_per_core, 1.0)
    return SURF_CMAX * (out_area / (out_area + SURF_A)) * (n / (n + SURF_N))

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
