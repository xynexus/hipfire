#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase 5: the refit loop — residuals and what to measure next.

Per docs/npu/aie2-cost-model-plan.md §4 "the iteration loop":

    predict(spec) -> measure(spec via IRON) -> residual -> refit -> report

This is the part that makes the model a best-estimate *process* rather than a
one-shot guess. Each pass reports per-term attribution, the residual against a
real measurement, and — the actual output — which constant is least constrained
and therefore what to measure next.

A residual is not automatically the model's fault. The report distinguishes:
  - error inside the declared +/-30% gate (plan §3): fine, no action
  - error outside it, with one dominant term: that term's constant is suspect
  - error outside it, no dominant term: the composition (overlap vs sum) is suspect
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from . import calib, model
from .spec import Prediction, ScheduleSpec

GATE = 0.30  # plan §3: +/-30% on device span inside the calibrated envelope


@dataclass
class Residual:
    """One prediction scored against one measurement."""

    spec_name: str
    predicted_s: float
    measured_s: float
    basis: str  # "device" or "wrapper"

    @property
    def error(self) -> float:
        return (self.predicted_s - self.measured_s) / self.measured_s if self.measured_s else float("nan")

    @property
    def within_gate(self) -> bool:
        return abs(self.error) <= GATE

    def render(self) -> str:
        return (
            f"  {self.spec_name:<28} pred={self.predicted_s * 1e6:9.2f} us  meas={self.measured_s * 1e6:9.2f} us  "
            f"err={self.error * 100:+7.1f}%  {'ok' if self.within_gate else 'OUTSIDE GATE'}"
        )


def score(spec: ScheduleSpec, measured_s: float, basis: str = "device", key: str | None = None) -> tuple[Prediction, Residual]:
    p = model.predict(spec, key)
    pred = p.device_s if basis == "device" else p.wrapper_s
    return p, Residual(spec.name, pred, measured_s, basis)


def diagnose(p: Prediction, r: Residual) -> list[str]:
    """Turn a residual into an instruction: what to measure next, and why."""
    out: list[str] = []
    if not p.admissible:
        return [f"model refused: uncalibrated {', '.join(p.missing)} — run those benches first"]
    if r.within_gate:
        out.append(f"within the +/-{GATE * 100:.0f}% gate; no refit indicated")
        return out

    total = sum(v for k, v in p.terms.items() if k in ("t_feed", "t_core", "t_drain", "t_task", "fill_tail"))
    if total <= 0:
        return ["no device-side terms; nothing to attribute"]

    ranked = sorted(((k, v) for k, v in p.terms.items()), key=lambda kv: -kv[1])
    top, top_v = ranked[0]
    share = top_v / total if total else 0.0

    direction = "over" if r.error > 0 else "under"
    out.append(f"{direction}-predicted by {abs(r.error) * 100:.1f}% (gate {GATE * 100:.0f}%)")
    if share > 0.6:
        bench = {"t_feed": "C2", "t_core": "C4/K1", "t_drain": "C5", "t_task": "C3", "fill_tail": "C6",
                 "t_submit": "C1", "t_host": "C7"}.get(top, "?")
        out.append(f"one term dominates: {top} at {share * 100:.0f}% of device time -> suspect its constant ({bench})")
        out.append(f"next: re-run {bench} at this schedule's operating point, not the calibration point")
    else:
        out.append("no single dominant term -> suspect the composition, not a constant")
        out.append("next: check the overlap assumption — is max(feed, core, drain) actually overlapping here,")
        out.append("      or is this schedule serialising (which would make a sum closer than a max)?")
    return out


def report(rows: list[tuple[ScheduleSpec, float]], basis: str = "device", key: str | None = None) -> str:
    key = key or calib.current_key()
    lines = [f"refit residual report — key={key}  basis={basis}", ""]
    residuals = []
    for spec, meas in rows:
        p, r = score(spec, meas, basis, key)
        residuals.append((p, r))
        lines.append(r.render())
    lines.append("")

    scored = [r for _, r in residuals if r.measured_s]
    if scored:
        n_ok = sum(1 for r in scored if r.within_gate)
        lines.append(f"  {n_ok}/{len(scored)} inside the +/-{GATE * 100:.0f}% gate")

        for line in systematic(scored):
            lines.append(f"  {line}")

        worst = max(scored, key=lambda r: abs(r.error))
        lines.append("")
        lines.append(f"  worst: {worst.spec_name}")
        p = next(p for p, r in residuals if r is worst)
        for line in diagnose(p, worst):
            lines.append(f"    - {line}")
    return "\n".join(lines)


def systematic(rows: list[Residual]) -> list[str]:
    """Look across residuals for a pattern no single-point diagnosis can see.

    A uniformly-signed residual whose ABSOLUTE size is roughly constant means a
    missing fixed term; a uniform *percentage* error means a wrong rate. Judging
    each point alone cannot tell these apart — the first phase-6 run had every
    candidate under-predicted by a near-constant ~160-260 us, and the per-point
    diagnosis wrongly blamed the dominant term's constant (C2) when the real
    cause was the dispatch floor being excluded from device time.
    """
    if len(rows) < 3:
        return []
    out: list[str] = []
    signs = {r.error > 0 for r in rows}
    if len(signs) > 1:
        return []  # mixed signs: no systematic bias

    offsets = [abs(r.predicted_s - r.measured_s) for r in rows]
    errs = [abs(r.error) for r in rows]
    mean_off = sum(offsets) / len(offsets)
    mean_err = sum(errs) / len(errs)
    spread_off = (max(offsets) - min(offsets)) / mean_off if mean_off else 9e9
    spread_err = (max(errs) - min(errs)) / mean_err if mean_err else 9e9

    direction = "OVER" if next(iter(signs)) else "UNDER"
    out.append(f"SYSTEMATIC: every residual is the same sign ({direction}-predicted)")
    if spread_off < spread_err:
        out.append(f"  absolute offset is more consistent than percentage ({spread_off:.2f} vs {spread_err:.2f} spread)")
        out.append(f"  => a MISSING FIXED TERM of ~{mean_off * 1e6:.0f} us, not a wrong rate")
    else:
        out.append(f"  percentage error is more consistent than absolute ({spread_err:.2f} vs {spread_off:.2f} spread)")
        out.append(f"  => a WRONG RATE (~{mean_err * 100:.0f}% off), not a missing fixed term")
    return out


def load_rows(path: str | Path) -> list[tuple[ScheduleSpec, float]]:
    """JSON: [{"spec": {...}, "measured_s": 1.23e-4}, ...]"""
    raw = json.loads(Path(path).read_text())
    rows = []
    for entry in raw:
        s = ScheduleSpec(**entry["spec"])
        if isinstance(s.mmul_shape, list):
            s.mmul_shape = tuple(s.mmul_shape)
        rows.append((s, float(entry["measured_s"])))
    return rows
