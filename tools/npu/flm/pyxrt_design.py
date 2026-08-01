#!/usr/bin/env python3
"""Dispatch an `iron.jit` design through a HELD `pyxrt.run`, with parameters.

Two things the `iron.jit` callable cannot do, both needed by multi-token decode:

  * **Hold the run object across dispatches.** The callable builds a fresh
    `pyxrt.run` and re-binds every buffer on each call, which costs 5714 us on
    the fused design -- more than lm_head. Held, wall time IS device time.
  * **Reach the control scratchpad.** `ParameterScratchpad` needs
    `run.get_ctrl_scratchpad_bo()`, and the callable never exposes the run.

`iron.jit(..., full_elf=True)` already dispatches through exactly this path
internally (`hostruntime.py:_load_full_elf` / `_run_full_elf`), so this is not a
different mechanism -- it is the same one, with the run kept.

The ELF, the kernel name and the argument order all come from the design's own
`CompilableDesign`, so this cannot drift from what `iron.jit` would have done.
Argument ORDER is the design function's parameter order = the
`Runtime(sequence, arg_types)` order = the `aie.runtime_sequence` operand order;
every argument's element count is asserted against what aiecc lowered, so a
mis-ordered argument of a different size fails here rather than silently.
"""

import time
from pathlib import Path

import numpy as np

import pyxrt
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime


class PyxrtDesign:
    """Stands in for the `iron.jit` CallableDesign, taking the same arguments."""

    def __init__(self, design, iters=1, probe_scratchpad=False, rebind=False,
                 recheck=False, quiet=False):
        self.design = design
        self.iters = iters
        self.probe_scratchpad = probe_scratchpad
        self.rebind = rebind
        self.recheck = recheck
        self.times_us = []
        self.params = None
        self.run = None
        self._bound = None

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
        if not quiet:
            print(f"  pyxrt: {self.elf_path}")
            print(f"  pyxrt: kernel {self.kernel_name!r}, "
                  f"{len(self.expected_sizes or [])} sequence operands")

    # ---- binding -----------------------------------------------------------
    def bind(self, *args):
        """Create the run object and bind every buffer ONCE.

        A held run re-reads its buffers' current contents on every `start()`,
        which is the property the decode loop is built on -- see `recheck`.
        """
        if self.expected_sizes is not None:
            assert len(args) == len(self.expected_sizes), (
                f"{len(args)} tensors passed, sequence takes "
                f"{len(self.expected_sizes)}")
            for i, (a, n) in enumerate(zip(args, self.expected_sizes)):
                assert a.numel() == n, f"arg {i}: {a.numel()} elements, want {n}"
        for a in args:
            a.to("npu")
        self.run = pyxrt.run(self.kernel)
        for i, a in enumerate(args):
            self.run.set_arg(i, a.buffer_object())
        self._bound = args
        # a fresh run has a fresh control scratchpad: the parameters have to be
        # written again, so drop the stale wrapper rather than keep one pointing
        # at the previous run's BO.
        self.params = None
        return self.run

    def parameters(self):
        """The `ParameterScratchpad` for this run, or None if the design has no
        `ScratchpadParameter`.

        Reports the two failure modes separately, because they need different
        fixes: no control scratchpad BO means the DESIGN declares no parameter,
        and no `params.txt` means aiecc was not given
        `--get-scratchpad-parameters`.
        """
        if self.params is not None:
            return self.params
        assert self.run is not None, "bind() first"
        try:
            self.run.get_ctrl_scratchpad_bo()
        except Exception as e:                  # noqa: BLE001 -- the text IS the result
            raise RuntimeError(
                f"no control scratchpad BO ({e}) -- the design declares no "
                f"ScratchpadParameter") from e
        p = self.elf_path.parent / "params.txt"
        if not p.exists():
            raise RuntimeError(
                f"{p} absent -- iron.jit needs "
                f"aiecc_flags=['--get-scratchpad-parameters']")
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad)
        self.params = ParameterScratchpad(self.run, p)
        return self.params

    def dispatch(self):
        """One dispatch on the held run. Returns the wall time in us."""
        t0 = time.perf_counter_ns()
        self.run.start()
        r = self.run.wait()
        us = (time.perf_counter_ns() - t0) / 1e3
        if r != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"kernel returned {r}")
        return us

    def __call__(self, *args, **_kw):
        """The `iron.jit` CallableDesign interface, for harnesses that expect it.

        `fused.Session` does not use this -- it binds once and calls `dispatch()`
        per token, because a rebind drops the control scratchpad the position
        lives in.
        """
        # Rebind when the ARGUMENTS CHANGE, not just when told to. Holding the
        # bind unconditionally silently ignored every argument after the first
        # call: `lmhead_twostage.py`'s gate passes a fresh activation tensor per
        # probe and got probe 0's logits back for all 48, which read as a coarse
        # tier with no recall (36/48 outside the top-64) rather than as a
        # dispatch that never saw the new input. Identity, not equality -- the
        # held path (`redo`, `fused.Session`) reuses the SAME tensor objects and
        # must keep its run, which is worth 5714 us on the fused design.
        # A rebind drops the control scratchpad, so anything carrying a position
        # there binds once and calls dispatch() directly, as fused.Session does.
        changed = (self._bound is not None
                   and (len(args) != len(self._bound)
                        or any(a is not b for a, b in zip(args, self._bound))))
        if self._bound is None or self.rebind or changed:
            self.bind(*args)
        self.times_us = [self.dispatch() for _ in range(self.iters)]
        return None

    def _recheck(self, xbuf):
        """The property multi-token actually needs, on a HELD run object.

        Binding once and dispatching many times is only useful if each dispatch
        reads the buffer's CURRENT contents. A `set_arg` that snapshotted, or a
        sync that silently no-op'd, would leave every token after the first
        computing the first token's answer -- and at pos 0 with the same input
        that failure is invisible, because the right answer and the stale answer
        are the same value. So: perturb, dispatch, restore, dispatch, and demand
        that the output moved and then came back bit for bit.
        """
        import fused
        nlay = xbuf.numel() // fused.BLK - 1
        inp = slice(fused.K_DIM, 2 * fused.K_DIM)
        out = slice(nlay * fused.BLK + fused.K_DIM,
                    nlay * fused.BLK + 2 * fused.K_DIM)

        def go():
            self.dispatch()
            return xbuf.numpy()[out].copy()

        def put(v):
            xbuf.numpy()[inp] = v
            xbuf.device = "cpu"          # mark dirty; numpy() left it "npu"
            xbuf.to("npu")

        first = xbuf.numpy()[out].copy()
        x0 = xbuf.numpy()[inp].copy()
        put(x0 * 0.5)                    # exact in bf16, and not degenerate
        moved = go()
        put(x0)
        back = go()
        assert not np.array_equal(moved, first), (
            "held run: perturbing the input did not change the output -- the "
            "dispatch is not re-reading the buffer")
        assert np.array_equal(back, first), (
            "held run: restoring the input did not restore the output")
        print("  recheck: held run re-reads its buffers, and repeats bit for bit")

    def _probe(self):
        print("  scratchpad probe:")
        try:
            bo = self.run.get_ctrl_scratchpad_bo()
            print(f"    ctrl scratchpad BO: OK, {bo.size()} bytes")
        except Exception as e:            # noqa: BLE001 -- the text IS the result
            print(f"    ctrl scratchpad BO: {type(e).__name__}: {e}")
        p = self.elf_path.parent / "params.txt"
        print(f"    {p}: {'present' if p.exists() else 'ABSENT'}")
        if p.exists():
            print("    " + p.read_text().strip().replace("\n", "\n    "))
        self.parameters()
        print("    ParameterScratchpad: constructed")
