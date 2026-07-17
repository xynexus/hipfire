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
        for cols in _column_options(target):
            cores = cols * (target.compute_cores // target.compute_columns)
            for (mm, mk, mn) in shapes:
                macs_per_call = mm * mk * mn
                calls_total = -(-self.total_macs // macs_per_call)
                calls_per_core = -(-calls_total // cores)
                # Weights stream once; activations are broadcast to every column.
                wire = self.weight_bytes() + self.act_bytes() * cols
                # One A-tile + one B-tile double-buffered, plus the C accumulator.
                stage = (mm * mk + mk * mn) * 2 * 2 + mm * mn * 4
                out.append(
                    ScheduleSpec(
                        name=f"gemm-{self.dtype_a}x{self.dtype_b}-c{cols}-mmul{mm}.{mk}.{mn}",
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


def rank_and_render(specs: list[ScheduleSpec], key: str, device: str, header: str) -> str:
    rows = [(s, model.predict(s, key, device)) for s in specs]
    ok = [(s, p) for s, p in rows if p.buildable and p.admissible]
    bad = [(s, p) for s, p in rows if not (p.buildable and p.admissible)]
    ok.sort(key=lambda sp: sp[1].device_s)

    lines = [header, ""]
    if not ok:
        lines.append("  no buildable+admissible candidate:")
        for s, p in bad[:3]:
            lines.append(f"    {s.name}: {'; '.join(p.build_errors or p.missing)}")
        return "\n".join(lines)

    lines.append(f"  {'candidate':<40} {'device':>10} {'limiter':>9} {'TOPS':>7} {'energy':>9} {'AI':>7} {'E-bound':>9}")
    for s, p in ok:
        ebound = "movement" if p.energy_terms.get("movement", 0) > p.energy_terms.get("compute", 0) else "compute"
        lines.append(f"  {s.name:<40} {p.device_s * 1e6:>9.1f}u {p.limiter:>9} {p.useful_tops:>7.2f} "
                     f"{p.energy_j * 1e3:>8.3f}m {p.arithmetic_intensity:>7.1f} {ebound:>9}")
    if bad:
        lines.append("")
        for s, p in bad:
            why = p.build_errors[0] if p.build_errors else f"uncalibrated: {p.missing[0]}"
            lines.append(f"  REJECTED {s.name}: {why}")

    best_s, best_p = ok[0]
    lines.append("")
    lines.append(f"  best: {best_s.name}")
    for k, v in sorted(best_p.terms.items(), key=lambda kv: -kv[1]):
        mark = "  <== limiter" if k == best_p.limiter else ""
        lines.append(f"    {k:<10} {v * 1e6:9.2f} us{mark}")
    for a in best_p.advice:
        lines.append(f"    ADVICE: {a}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
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
        print(rank_and_render(prob.candidates(target), key, args.device, hdr))
        return 0

    if args.cmd == "attn":
        prob = DecodeAttnProblem(args.context, args.kv_heads, args.head_dim, args.kv_bits)
        hdr = (f"decode attention, 1 layer, ctx={args.context} kv_heads={args.kv_heads} "
               f"head_dim={args.head_dim} kvarn{args.kv_bits} on {target.key}\n"
               f"  KV bytes/layer {prob.kv_bytes() / 1024:.0f} KiB, {prob.total_macs / 1e6:.1f} M MACs")
        print(rank_and_render(prob.candidates(target), key, args.device, hdr))
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
