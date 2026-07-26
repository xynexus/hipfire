#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase 6: commit-first ordinal validation.

Per docs/npu/aie2-cost-model-plan.md §7 phase 6. There is no aie2p-style
back-test corpus for npu1 — all 95 durable CSVs are halo/aie2p — so the model
cannot be validated against history. Instead it must **commit first**: print its
predicted ranking, then run the sweep, then get scored. The model has no chance
to see the answer before it speaks.

Ordinal accuracy is the product (plan §3): every win in the aie2p corpus came
from correctly ranking two candidates, not from a precise number.

Independence note, stated plainly: the transport family here reuses C2's kernel
shape, so bw_feed was fitted on the same *kind* of schedule. Operating points are
deliberately chosen OUTSIDE the calibration grid (C2 used tiles in
{512,1024,2048} at a 4 KiB tile), but this is an interpolation test, not a fully
independent one. It is scored as such.

Usage:
    python -m aiecost.validate            # commit, run, score
    python -m aiecost.validate --dry-run  # print the committed ranking only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiecost import calib, model, refit  # noqa: E402
from aiecost.spec import ScheduleSpec  # noqa: E402
from aiecost.target import resolve_target  # noqa: E402

TILE_BYTES = 4096
ACC_BYTES = 64


@dataclass
class Candidate:
    name: str
    n_tiles: int
    columns: int

    def spec(self) -> ScheduleSpec:
        # Every column consumes the full stream (C2's broadcast shape).
        wire = TILE_BYTES * self.n_tiles * self.columns
        return ScheduleSpec(
            name=self.name,
            columns=self.columns,
            cores=self.columns,
            wire_bytes_in=wire,
            output_bytes=ACC_BYTES * self.columns,
            dma_tasks_live=self.n_tiles,
            bds_per_core=2,
            locks_per_core=2,
            fifo_depth=2,
            mmul_calls_per_core=0,  # the sink kernel does no mmul
            local_stage_bytes=TILE_BYTES * 2,
            host_pack_bytes=TILE_BYTES * self.n_tiles,
            host_deblock_bytes=ACC_BYTES * self.columns,
            n_bos=1 + self.columns,
        )


# Operating points deliberately off C2's calibration grid {512,1024,2048}x{1,2,4}.
#
# FAMILY A is BURNED as a validation set. Its first run exposed the missing
# dispatch-floor term (every candidate under-predicted by a near-constant
# ~160-260 us); the model was then fixed using those residuals. Re-running A now
# scores tau=1.000, but that number is contaminated — the fix was derived from
# this data. A is kept for regression only.
#
# FAMILY B has never been used to fit or fix anything. It is the honest
# post-fix validation set. Burn it once.
FAMILY_A = [
    Candidate("c1-t1500", 1500, 1),
    Candidate("c1-t3000", 3000, 1),
    Candidate("c2-t700", 700, 2),
    Candidate("c2-t1400", 1400, 2),
    Candidate("c4-t400", 400, 4),
    Candidate("c4-t1500", 1500, 4),
]

FAMILY_B = [
    Candidate("b-c1-t900", 900, 1),
    Candidate("b-c1-t2200", 2200, 1),
    Candidate("b-c2-t1100", 1100, 2),
    Candidate("b-c2-t250", 250, 2),
    Candidate("b-c4-t800", 800, 4),
    Candidate("b-c4-t2600", 2600, 4),
]

# FAMILY C is a different REGIME, not just different points. A and B are entirely
# t_feed-limited (transport-bound, trivial compute). C is t_core-limited: resident
# operands, zero feed, pure VMAC. LLM kernels span both — a QKV GEMM can be
# compute-bound while KV-cache attention is bandwidth-bound — so a model validated
# only on transport says nothing about half the design space.
#
# Points are OFF every calibration grid: K1 fitted f_H from iters in
# {200k, 400k, 800k, 1.6M} and C4 from {400k, 800k, 1.6M}. C uses none of them.
# The int4 rows additionally exercise the ceiling lookup that C4 added.
@dataclass
class ComputeCandidate:
    name: str
    iters: int
    shape: tuple  # (M, K, N, TypeA, TypeB)
    chains: int = 4  # matches c4_mmul.CHAINS

    def spec(self) -> ScheduleSpec:
        m, k, n, ta, tb = self.shape
        return ScheduleSpec(
            name=self.name,
            columns=1,
            cores=1,
            wire_bytes_in=0,  # operands are resident; nothing streams
            output_bytes=0,
            dma_tasks_live=1,
            bds_per_core=3,
            locks_per_core=4,
            fifo_depth=1,
            mmul_calls_per_core=self.iters * self.chains,
            mmul_shape=(m, k, n),
            dtype_a=ta,
            dtype_b=tb,
            local_stage_bytes=256,
            host_pack_bytes=0,
            host_deblock_bytes=0,
            n_bos=3,
        )


FAMILY_C = [
    ComputeCandidate("c-i300k-i8", 300_000, (4, 8, 8, "int8", "int8")),
    ComputeCandidate("c-i1200k-i8", 1_200_000, (4, 8, 8, "int8", "int8")),
    ComputeCandidate("c-i600k-i8-virt", 600_000, (8, 8, 8, "int8", "int8")),
    ComputeCandidate("c-i300k-i4", 300_000, (4, 16, 8, "int8", "int4")),
    ComputeCandidate("c-i1200k-i4", 1_200_000, (4, 16, 8, "int8", "int4")),
    ComputeCandidate("c-i600k-i4-virt", 600_000, (4, 32, 8, "int8", "int4")),
]

FAMILIES = {"a": FAMILY_A, "b": FAMILY_B, "c": FAMILY_C}


def kendall_tau(a: list[float], b: list[float], tol: float = 0.02) -> tuple[float, int]:
    """Rank correlation over VALUES, not ranks. Returns (tau, n_tied_pairs).

    Takes predicted/measured values rather than sorted ranks so that TIES are
    visible. This matters: family C's candidates are physically identical work
    (all 4.8M native VMACs via different call/shape splits), so the model
    correctly predicts equal times. Ranking them is meaningless, and measurement
    noise orders them arbitrarily — an earlier version sorted predictions into
    distinct ranks, which destroyed the tie information and scored those
    arbitrary orderings as errors (tau=0.600 for a model that was right).

    A pair is tied if the two predictions are within `tol` relative, and such
    pairs are excluded from the denominator: the model is scored only on
    distinctions it actually claims to make.
    """
    n = len(a)
    if n < 2:
        return float("nan"), 0
    conc = disc = tied = 0
    for i, j in itertools.combinations(range(n), 2):
        denom = max(abs(a[i]), abs(a[j])) or 1.0
        if abs(a[i] - a[j]) / denom <= tol:
            tied += 1
            continue
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    total = conc + disc
    return ((conc - disc) / total if total else float("nan")), tied


def measure(cand, reps: int, warmup: int, cache: Path, device: str) -> float | None:
    if isinstance(cand, ComputeCandidate):
        # Compute-bound: resident operands, no feed. c4_mmul builds the VMAC chain.
        # c4_mmul is not device-parameterised yet (npu1 only), unlike c2_feed —
        # family C is therefore AIE2-only until that is threaded through.
        from aiecost.benches import c4_mmul

        built = c4_mmul.build(cand.shape, cand.iters, cache)
        if not built:
            return None
        return c4_mmul.run(*built, cand.shape, reps)

    from aiecost.benches import c2_feed

    built = c2_feed.build(cand.n_tiles, cand.columns, cache, device)
    if not built:
        return None
    r = c2_feed.run(*built, cand.n_tiles, cand.columns, reps, warmup)
    return r["npu_med"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=["a", "b", "c"], default="b",
                   help="b = transport-bound (clean); c = compute-bound regime; a = burned, regression only")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--dry-run", action="store_true", help="print the committed ranking, do not measure")
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "validate"))
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    target = resolve_target(args.device)
    key = calib.current_key()
    print(f"phase 6: commit-first ordinal validation — target={target.cache_tag} key={key}\n")

    # ── COMMIT ──
    family = FAMILIES[args.family]
    if args.family == "a":
        print("  NOTE: family A is BURNED — the dispatch-floor fix was derived from its residuals.\n"
              "  Its tau is contaminated and counts as regression, not validation.\n")
    preds = [(c, model.predict(c.spec(), key, target.key)) for c in family]
    refused = [c.name for c, pr in preds if not (pr.buildable and pr.admissible)]
    if refused:
        print(f"model refused/rejected: {refused}")
        for c, pr in preds:
            if not (pr.buildable and pr.admissible):
                print(pr.render())
        return 1

    committed = sorted(preds, key=lambda cp: cp[1].device_s)
    print("COMMITTED PREDICTION (before any measurement), fastest first:")
    for i, (c, pr) in enumerate(committed, 1):
        print(f"  {i}. {c.name:<12} {pr.device_s * 1e6:9.2f} us   limiter={pr.limiter}")
    print()

    if args.dry_run:
        return 0

    # ── MEASURE ──
    print("measuring...")
    measured: dict[str, float] = {}
    for c, _ in preds:
        t = measure(c, args.reps, args.warmup, Path(args.cache), target.key)
        if t:
            measured[c.name] = t
            print(f"  {c.name:<12} {t * 1e6:9.2f} us")
    print()

    usable = [(c, pr) for c, pr in preds if c.name in measured]
    if len(usable) < 2:
        print("insufficient measurements")
        return 1

    # ── SCORE ──
    pred_order = {c.name: i for i, (c, _) in enumerate(sorted(usable, key=lambda cp: cp[1].device_s))}
    meas_order = {n: i for i, n in enumerate(sorted(measured, key=lambda n: measured[n]))}
    names = [c.name for c, _ in usable]
    pred_vals = {c.name: pr.device_s for c, pr in usable}
    # Score on values so predicted ties are excluded, not scored as errors.
    tau, n_tied = kendall_tau([pred_vals[n] for n in names], [measured[n] for n in names])

    print("ORDINAL RESULT:")
    print(f"  {'candidate':<12} {'pred rank':>9} {'meas rank':>9}")
    for n in sorted(names, key=lambda n: pred_order[n]):
        # Only call it misordered if the model actually claimed a distinction.
        tied_with = [m for m in names if m != n and abs(pred_vals[n] - pred_vals[m]) / max(pred_vals[n], pred_vals[m]) <= 0.02]
        if pred_order[n] == meas_order[n]:
            flag = ""
        elif tied_with:
            flag = "   (tied prediction — not scored)"
        else:
            flag = "   <- misordered"
        print(f"  {n:<16} {pred_order[n] + 1:>9} {meas_order[n] + 1:>9}{flag}")
    gate = 0.8
    n_pairs = len(names) * (len(names) - 1) // 2
    print(f"\n  Kendall tau = {tau:+.3f}   (gate: >= {gate})   {'PASS' if tau >= gate else 'FAIL'}")
    if n_tied:
        print(f"  {n_tied}/{n_pairs} pairs predicted TIED (within 2%) and excluded — the model declines to")
        print("  order them, so it is not scored on them. Their magnitudes are still checked below.")

    print()
    print(refit.report([(c.spec(), measured[c.name]) for c, _ in usable], basis="device", key=key, device=target.key))

    print("\n  scope: interpolation within C2's kernel family at off-grid operating points;")
    print("  not an independent test of a different dataflow.")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "key": key,
                    "tau": tau,
                    "committed": [{"name": c.name, "device_s": pr.device_s} for c, pr in committed],
                    "measured": measured,
                },
                indent=2,
            )
        )
    return 0 if tau >= gate else 2


if __name__ == "__main__":
    sys.exit(main())
