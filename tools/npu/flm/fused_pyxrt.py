#!/usr/bin/env python3
"""Diagnostics for the way `fused.py` now drives its design.

The held `pyxrt.run` this file existed to de-risk is no longer optional: with
position a runtime `ScratchpadParameter`, the value is written through
`run.get_ctrl_scratchpad_bo()`, so `fused.Session` holds the run and there is no
other path. `PyxrtDesign` moved to `pyxrt_design.py`, which both use.

What is left here is the three checks worth being able to run on demand, each
wrapping `fused.main()` so the buffers are the real ones:

    --probe-scratchpad   is the control scratchpad BO there, is params.txt
                         there, and does ParameterScratchpad parse against it?
                         The two failures need different fixes: no BO means the
                         DESIGN declares no parameter; no params.txt means aiecc
                         was not given --get-scratchpad-parameters.
    --recheck            a held run re-reads its buffers on every start().
                         Perturb, dispatch, restore, dispatch, demand the output
                         moved and came back bit for bit -- at a fixed input a
                         dispatch that silently reused the last result is
                         otherwise indistinguishable from a correct one.
    --rebind --iters N   the per-dispatch cost of rebuilding the run and
                         re-setting 36 arguments each call, which is what the
                         iron.jit callable does: 2983 us on the fused design.
                         Timing only -- a rebind drops the scratchpad BO, so the
                         position reverts to whatever a fresh one holds.

    python3 fused_pyxrt.py --probe-scratchpad -- --layers 16 --pos 0 --no-ref

The `--iron` baseline is gone. It measured the iron.jit callable's wall time
(18.6 ms against 12.9 held, all of it host-side buffer rebinding), which is
recorded in the log; it cannot be run any more, because that path never writes
the scratchpad and so would dispatch with whatever offsets a fresh BO holds.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import fused  # noqa: E402
from pyxrt_design import PyxrtDesign  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=1,
                   help="extra dispatches after the run, for per-dispatch cost")
    p.add_argument("--rebind", action="store_true",
                   help="rebuild the pyxrt.run and re-set_arg every dispatch, "
                        "as the iron.jit callable does")
    p.add_argument("--recheck", action="store_true",
                   help="prove a held run re-reads its buffers: perturb the "
                        "input, dispatch, restore, dispatch, compare")
    p.add_argument("--probe-scratchpad", action="store_true",
                   help="report whether the run object can reach the control "
                        "scratchpad and whether params.txt exists")
    o, rest = p.parse_known_args()

    holder = {}
    inner = fused.build

    def build(*a, **kw):
        holder["d"] = PyxrtDesign(inner(*a, **kw), iters=o.iters,
                                  probe_scratchpad=o.probe_scratchpad,
                                  rebind=o.rebind, recheck=o.recheck)
        return holder["d"]

    fused.build = build
    sys.argv = [sys.argv[0]] + rest
    rc = fused.main()

    d = holder.get("d")
    if d is not None and o.probe_scratchpad:
        d._probe()
    if d is not None and o.recheck:
        d._recheck(d._bound[0])
    if d is not None and o.iters > 1:
        t = np.array([d.dispatch() for _ in range(o.iters)])
        # The FIRST dispatch is dropped from the summary and reported on its own:
        # it is an order of magnitude slower (first touch of ~500 MB of weight
        # BOs), so folding it in would report a warm-up as an overhead.
        w = t[1:] if len(t) > 1 else t
        print(f"  pyxrt: first {t[0]:.1f} us | warm n={len(w)} "
              f"min {w.min():.1f}  median {np.median(w):.1f}  "
              f"max {w.max():.1f}  spread {(w.max() / w.min() - 1) * 100:.1f}%")
    return rc


if __name__ == "__main__":
    sys.exit(main())
