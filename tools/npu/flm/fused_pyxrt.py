#!/usr/bin/env python3
"""Drive `fused.py`'s design through raw pyxrt instead of the iron.jit callable.

Why this exists
---------------
Multi-token decode needs `position` to stop being a build parameter: today it is
interpolated into the drain offsets, so every position is a separate xclbin, and
loading a new xclbin clears `g_kprev` in core .bss. The fix is a runtime
`offset_parameter=` fed by a `ScratchpadParameter`, whose VALUE is written from
the host with `ParameterScratchpad(run, "params.txt")` -- a `pyxrt.run` object
the design callable never exposes. So the question this file answers is: can the
EXISTING cached build be dispatched from a `pyxrt.run` we hold ourselves, and
does it give the same answer?

    python3 fused_pyxrt.py --layers 16 --pos 0 --seq 1 --x0 x0_bos.npy \\
                           --save x16.npy --no-ref

Everything after this file's own flags is `fused.py`'s own argv, and the buffers,
the reference chain and the save are `fused.main()`'s -- only the dispatch is
swapped, by monkeypatching `fused.build`. Nothing in `fused.py` is modified;
that is deliberate, because a copy of its 140 lines of buffer packing would be a
second place for the two paths to differ and the whole point is that they do not.

What it found
-------------
It reproduces pos 0 BIT-IDENTICALLY, and there was no restructuring to de-risk:
`iron.jit(..., full_elf=True)` ALREADY dispatches through
`pyxrt.hw_context(dev, pyxrt.elf(design.elf))` + `pyxrt.ext.kernel` +
`pyxrt.run` + `run.set_arg(i, bo)` -- see
`aie/utils/hostruntime/xrtruntime/hostruntime.py:_load_full_elf` /
`_run_full_elf`. The fused design already passes `full_elf=True`. This file
re-does by hand what the callable does internally; the only thing it adds is
HOLDING the `pyxrt.run` object across dispatches, and that is worth 5.7 ms a
token (`--iters`, and `--rebind` / `--iron` for the two baselines).

What blocks the runtime offset is NOT an aiecc flag. `--probe-scratchpad`
reports `Control scratchpad memory is not present`, and the same probe against
upstream's `scratchpad_addr_offset` ELF built WITHOUT
`--get-scratchpad-parameters` reports a scratchpad present -- so the BO comes
from the design declaring a `ScratchpadParameter`, not from the flag. The flag
only emits `params.txt` (name -> state-table slot + kind), which lands in
aiecc's `--output-dir`, i.e. the JIT cache dir, and is what `ParameterScratchpad`
reads. So the fused design needs the parameter added in `fused.py` AND
`aiecc_flags=["--get-scratchpad-parameters"]` on its `iron.jit`; both bust the
cache, so it costs one rebuild.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import pyxrt  # noqa: E402
import fused  # noqa: E402
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime  # noqa: E402


class PyxrtDesign:
    """Stands in for the `iron.jit` CallableDesign, taking the same arguments.

    The ELF, the kernel name and the argument order all come from the design
    itself, so this cannot drift from what `iron.jit` would have done: it asks
    the same `CompilableDesign` for the same cached `design.elf`.
    """

    def __init__(self, design, iters=1, probe_scratchpad=False, rebind=False,
                 recheck=False):
        self.design = design                    # kept, for the --iron baseline
        self.iters = iters
        self.probe_scratchpad = probe_scratchpad
        self.rebind = rebind
        self.recheck = recheck
        self.times_us = []
        self.params = None

        elf_path, insts = design.compile()      # cache hit -> the verified build
        assert insts is None, "not a full-ELF design; --get-full-elf is required"
        self.elf_path = Path(elf_path)
        self.kernel_name = design.compilable._full_elf_kernel_name
        self.expected_sizes = design.compilable._expected_tensor_sizes

        self.runtime = XRTHostRuntime()
        self.device = self.runtime._device
        self.elf = pyxrt.elf(str(self.elf_path))
        self.context = pyxrt.hw_context(self.device, self.elf)
        self.kernel = pyxrt.ext.kernel(self.context, self.kernel_name)
        print(f"  pyxrt: {self.elf_path}")
        print(f"  pyxrt: kernel {self.kernel_name!r}, "
              f"{len(self.expected_sizes or [])} sequence operands")

    def __call__(self, *args, **_kw):
        # Argument ORDER is the design function's parameter order, which is the
        # `Runtime(sequence, arg_types)` order, which is the
        # `aie.runtime_sequence` operand order. Cross-checked against the
        # element counts aiecc lowered, so a mis-ordered arg of a different size
        # is caught here rather than as a wrong answer.
        if self.expected_sizes is not None:
            assert len(args) == len(self.expected_sizes), (
                f"{len(args)} tensors passed, sequence takes "
                f"{len(self.expected_sizes)}")
            for i, (a, n) in enumerate(zip(args, self.expected_sizes)):
                assert a.numel() == n, f"arg {i}: {a.numel()} elements, want {n}"

        for a in args:
            a.to("npu")
        run = pyxrt.run(self.kernel)
        for i, a in enumerate(args):
            run.set_arg(i, a.buffer_object())

        if self.probe_scratchpad:
            self._probe(run)

        self.times_us = []
        for _ in range(self.iters):
            t0 = time.perf_counter_ns()
            if self.rebind:
                # What the iron.jit callable does every call: a fresh run object
                # and 36 fresh set_arg. Isolates that from the dispatch itself.
                run = pyxrt.run(self.kernel)
                for i, a in enumerate(args):
                    run.set_arg(i, a.buffer_object())
            run.start()
            r = run.wait()
            t1 = time.perf_counter_ns()
            if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"kernel returned {r}")
            self.times_us.append((t1 - t0) / 1e3)
        if self.recheck:
            self._recheck(run, args[0])
        return None

    def _recheck(self, run, xbuf):
        """The property multi-token actually needs, on a HELD run object.

        Binding once and dispatching many times is only useful if each dispatch
        reads the buffer's CURRENT contents. A `set_arg` that snapshotted, or a
        sync that silently no-op'd, would leave every token after the first
        computing the first token's answer -- and at pos 0 with the same input
        that failure is invisible, because the right answer and the stale answer
        are the same value. So: perturb, dispatch, restore, dispatch, and demand
        that the output moved and then came back bit for bit.
        """
        nlay = xbuf.numel() // fused.BLK - 1
        inp = slice(fused.K_DIM, 2 * fused.K_DIM)
        out = slice(nlay * fused.BLK + fused.K_DIM,
                    nlay * fused.BLK + 2 * fused.K_DIM)

        def dispatch():
            run.start()
            assert run.wait() == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
            return xbuf.numpy()[out].copy()

        def put(v):
            xbuf.numpy()[inp] = v
            xbuf.device = "cpu"          # mark dirty; numpy() left it "npu"
            xbuf.to("npu")

        first = xbuf.numpy()[out].copy()
        x0 = xbuf.numpy()[inp].copy()
        put(x0 * 0.5)                    # exact in bf16, and not degenerate
        moved = dispatch()
        put(x0)
        back = dispatch()
        assert not np.array_equal(moved, first), (
            "held run: perturbing the input did not change the output -- the "
            "dispatch is not re-reading the buffer")
        assert np.array_equal(back, first), (
            "held run: restoring the input did not restore the output")
        print("  recheck: held run re-reads its buffers, and repeats bit for bit")

    def _probe(self, run):
        """Ask the run object for what `ParameterScratchpad` needs, and say so.

        Two things can be missing and they need DIFFERENT fixes, so report them
        separately: the control scratchpad BO, which exists iff the design
        declares a `ScratchpadParameter` (a `fused.py` change), and
        `params.txt`, which exists iff aiecc was given
        `--get-scratchpad-parameters` (an `aiecc_flags=` change). Reporting one
        "scratchpad: no" would send the next person to the wrong file.

        On success it also constructs the `ParameterScratchpad`, because the BO
        existing and the map parsing against it are still two claims.
        """
        print("  scratchpad probe:")
        have_bo = False
        try:
            bo = run.get_ctrl_scratchpad_bo()
            have_bo = True
            print(f"    ctrl scratchpad BO: OK, {bo.size()} bytes")
        except Exception as e:            # noqa: BLE001 -- the text IS the result
            print(f"    ctrl scratchpad BO: {type(e).__name__}: {e}")
        p = self.elf_path.parent / "params.txt"
        print(f"    {p}: {'present' if p.exists() else 'ABSENT'}")
        if have_bo and p.exists():
            from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
                ParameterScratchpad)
            self.params = ParameterScratchpad(run, p)
            print("    ParameterScratchpad: constructed")


class IronDesign:
    """The `iron.jit` callable, timed the same way, as the baseline.

    `times_us` is wall time around the call, which is the number that matters
    per token. `run_iters` is also run, because it reports the runtime's OWN
    `npu_time` (the ns it measured around `run.wait()`) next to its own
    end-to-end -- and `npu_time` is what `fused.py --bench` quotes, so this is
    where the two clocks are shown to agree before any of them is compared.
    """

    def __init__(self, design, iters=1, **_kw):
        self.design = design
        self.iters = iters
        self.times_us = []

    def __call__(self, *args, **_kw):
        from aie.utils.benchmark import run_iters
        self.times_us = []
        for _ in range(self.iters):
            t0 = time.perf_counter_ns()
            self.design(*args)
            t1 = time.perf_counter_ns()
            self.times_us.append((t1 - t0) / 1e3)
        b = run_iters(self.design, *args, warmup=1, iters=max(1, self.iters - 1))
        if b.npu:
            print(f"  iron.jit run_iters npu  min {b.npu.min_us:.1f} "
                  f"avg {b.npu.avg_us:.1f} max {b.npu.max_us:.1f} us")
        print(f"  iron.jit run_iters e2e  min {b.e2e.min_us:.1f} "
              f"avg {b.e2e.avg_us:.1f} max {b.e2e.max_us:.1f} us")
        return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=1,
                   help="dispatches per run, for the per-dispatch overhead")
    p.add_argument("--iron", action="store_true",
                   help="dispatch through the iron.jit callable instead, as "
                        "the baseline to compare against")
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

    wrapper = IronDesign if o.iron else PyxrtDesign
    holder = {}
    inner = fused.build

    def build(*a, **kw):
        holder["d"] = wrapper(inner(*a, **kw), iters=o.iters,
                              probe_scratchpad=o.probe_scratchpad,
                              rebind=o.rebind, recheck=o.recheck)
        return holder["d"]

    fused.build = build
    sys.argv = [sys.argv[0]] + rest
    rc = fused.main()

    t = np.array(holder["d"].times_us)
    if len(t):
        which = "iron.jit" if o.iron else "pyxrt"
        print(f"  {which}: " + " ".join(f"{v:.1f}" for v in t))
        # The FIRST dispatch is dropped from the summary and reported on its own:
        # it is an order of magnitude slower on both paths (first touch of ~500 MB
        # of weight BOs), so folding it in would report a warm-up as an overhead.
        w = t[1:] if len(t) > 1 else t
        print(f"  {which}: first {t[0]:.1f} us | warm n={len(w)} "
              f"min {w.min():.1f}  median {np.median(w):.1f}  "
              f"max {w.max():.1f}  spread {(w.max() / w.min() - 1) * 100:.1f}%")
    return rc


if __name__ == "__main__":
    sys.exit(main())
