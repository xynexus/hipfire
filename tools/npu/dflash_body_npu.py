#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase D (step 1) — assemble the DFlash 5-layer block body on the NPU.

Composes the already-validated per-op NPU kernels (Gate B primitives + Gate C
attention) into the full DFlash drafter block forward, UNFUSED (one dispatch per
op), and validates full-body numeric parity against the Phase-A golden.

Kernels reused (host machinery imported from the test/build modules):
  - int8 per-group G256 projection : oq_gemm_design.matmul_npu  (@iron.jit)
  - rmsnorm  [H]        : qwen35-rmsnorm-4096.xclbin      (XRTHostRuntime)
  - headnorm [nh,hd]    : qwen35-headnorm-{q,k}-{nh}h128d  (XRTHostRuntime)
  - rope (full neox)    : dflash-rope-{q,k}-{nh}h128d      (XRTHostRuntime)
  - swiglu   [I]        : qwen35-swiglu-12288.xclbin       (XRTHostRuntime)
  - attention (1 head)  : build_dflash_attention_sc.run_attn_head (@iron.jit)

Validation (bf16/int8-aware, Gate-C precedent): the kernels are bf16 (+int8
projections), so we gate on cosine similarity vs the f16 golden AND vs a
bf16/int8-precision numpy reference that mirrors each op's precision.

Modes:
  --op-by-op   layer-0 hand-off checks: feed each stage its GOLDEN input and
               check the output vs the next golden slice (isolates I/O layout).
  (default)    full unfused body: chain all ops from the real inputs, check
               final block_hidden cos vs rust_final_block_hidden.

Env (fork, nix1):
  PATH=/opt/xilinx/xrt/bin:$PATH
  LD_LIBRARY_PATH=~/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
  PEANO_INSTALL_DIR=~/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie
  PYTHONPATH=~/mlir-aie-312/install/python:/opt/xilinx/xrt/python:$PYTHONPATH
  run with ~/mlir-aie-312/venv312/bin/python

Usage:
  dflash_body_npu.py --golden-dir <OUT>/rust --weights <safetensors> --op-by-op
  dflash_body_npu.py --golden-dir <OUT>/rust --weights <safetensors>
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# oq_gemm_design does the pyxrt/LD_LIBRARY_PATH/device bootstrap AND provides the
# @iron.jit int8 matmul. Import it FIRST so the iron env is set up before the
# XRTHostRuntime imports below (which also bootstrap pyxrt, harmlessly).
import oq_gemm_design as design  # noqa: E402

import numpy as np  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402
from aie.utils.npukernel import NPUKernel  # noqa: E402

from build_dflash_attention_sc import run_attn_head, run_attn_all_kv  # noqa: E402
from dflash_ref import load_safetensors_f32  # noqa: E402

KERNEL_NAME = "MLIR_AIE"
NPU_DIR = REPO_ROOT / "target" / "npu"
# ── artifact manifest ───────────────────────────────────────────────────────
# The native (Rust) driver cannot compute the @iron.jit cache hash, so the
# Python harness emits the resolved artifact paths + per-dispatch buffer
# contract it actually used. That manifest is the contract the native driver
# consumes; see docs/npu/dflash-native-driver-plan.md.
import json  # noqa: E402


class Manifest:
    """Records, per dispatch: op name, resolved xclbin/insts, buffer order+sizes."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.kernels = {}
        self.seq = []

    def kernel(self, key, xclbin, insts, compile_args=None):
        if not self.enabled:
            return key
        if key not in self.kernels:
            xclbin, insts = Path(xclbin), Path(insts)
            self.kernels[key] = {
                "xclbin": str(xclbin),
                "insts": str(insts),
                "xclbin_bytes": xclbin.stat().st_size if xclbin.exists() else -1,
                "insts_bytes": insts.stat().st_size if insts.exists() else -1,
                "compile_args": compile_args or {},
            }
        return key

    def dispatch(self, key, args):
        if self.enabled:
            self.seq.append({"kernel": key, "args": args})

    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"kernels": self.kernels, "dispatches": self.seq}, f, indent=2)
        print(f"[manifest] {len(self.kernels)} kernels, {len(self.seq)} dispatches "
              f"-> {path}")


MANIFEST = Manifest()

# Step-2 single-op parity: capture the FIRST int8 projection's exact device
# buffers (A weight, B activation, C int32 result) so the native driver can be
# fed byte-identical inputs and its int32 output diffed BIT-FOR-BIT. The GEMM is
# integer, so any difference is a layout/arg-order bug, never precision.
DUMP_OP_DIR = None
_DUMPED_OP = False
# Where to save the int8/bf16 precision reference (the second cosine gate), so
# the native driver validates against the identical array.
DUMP_REF_PATH = None


def _tdesc(t, role):
    """Describe one dispatch argument buffer (dtype/shape/bytes) for the manifest."""
    try:
        a = t.numpy()
        return {"role": role, "dtype": str(a.dtype), "shape": list(a.shape),
                "bytes": int(a.nbytes)}
    except Exception:
        return {"role": role, "dtype": "?", "shape": [], "bytes": -1}


def _maybe_dump_op(qw, qx, C, M, K, N):
    """Dump the first int8 projection's device buffers for native bit-parity."""
    global _DUMPED_OP
    if DUMP_OP_DIR is None or _DUMPED_OP:
        return
    _DUMPED_OP = True
    d = Path(DUMP_OP_DIR)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "op_A_int8.npy", np.ascontiguousarray(qw, np.int8))
    np.save(d / "op_B_int8.npy", np.ascontiguousarray(qx, np.int8))
    np.save(d / "op_C_int32.npy", np.ascontiguousarray(C, np.int32))
    m, k, n = design._tiles_for(M, K, N, design.DEFAULT_TILE)
    with open(d / "op_meta.json", "w") as f:
        json.dump({"M": M, "K": K, "N": N, "m": m, "k": k, "n": n,
                   "dtype_in_str": "i8", "dtype_out_str": "i32",
                   "b_col_maj": 1}, f, indent=2)
    print(f"[dump-op] int_matmul M={M} K={K} N={N} -> {d}")


def dump_native_weights(W, cfg, outdir):
    """Emit everything the native (Rust) body driver needs, once.

    The int8 quantization of the projection weights is an OFFLINE step (in a
    real deployment the .hfq sidecar already stores them quantized), so the
    native driver loads them pre-quantized rather than re-deriving them — and
    loads exactly the CONCATENATED groups the body dispatches, in the same
    stacking order the harness uses.

    Raw little-endian binary + an index.json of shapes: these are ~1 GB of
    int8, and .npy framing buys nothing when both sides are ours.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    index = {"gemms": {}, "gammas": {}}

    def put_gemm(key, names):
        qw, sw, sizes = W.qproj_row_concat(names)
        (outdir / f"w_{key}.i8").write_bytes(
            np.ascontiguousarray(qw, np.int8).tobytes())
        (outdir / f"w_{key}.scale").write_bytes(
            np.ascontiguousarray(sw, np.float32).tobytes())
        index["gemms"][key] = {"M": int(qw.shape[0]), "K": int(qw.shape[1]),
                               "sizes": [int(s) for s in sizes]}

    def put_gamma(key, name):
        g = np.ascontiguousarray(W.raw(name), np.float32)
        (outdir / f"g_{key}.f32").write_bytes(g.tobytes())
        index["gammas"][key] = {"n": int(g.size)}

    put_gemm("fc", ["fc.weight"])
    put_gamma("hidden_norm", "hidden_norm.weight")
    put_gamma("final_norm", "norm.weight")
    for li in range(cfg.NL):
        p = f"layers.{li}."
        put_gemm(f"l{li}_qkv", [p + "self_attn.q_proj.weight",
                                p + "self_attn.k_proj.weight",
                                p + "self_attn.v_proj.weight"])
        put_gemm(f"l{li}_kv", [p + "self_attn.k_proj.weight",
                               p + "self_attn.v_proj.weight"])
        put_gemm(f"l{li}_o", [p + "self_attn.o_proj.weight"])
        put_gemm(f"l{li}_gateup", [p + "mlp.gate_proj.weight",
                                   p + "mlp.up_proj.weight"])
        put_gemm(f"l{li}_down", [p + "mlp.down_proj.weight"])
        put_gamma(f"l{li}_input", p + "input_layernorm.weight")
        put_gamma(f"l{li}_post", p + "post_attention_layernorm.weight")
        put_gamma(f"l{li}_qnorm", p + "self_attn.q_norm.weight")
        put_gamma(f"l{li}_knorm", p + "self_attn.k_norm.weight")

    index["cfg"] = {"H": cfg.H, "I": cfg.I, "NH": cfg.NH, "NKV": cfg.NKV,
                    "HD": cfg.HD, "NL": cfg.NL, "B": cfg.B, "L": cfg.L,
                    "NE": cfg.NE, "tot": cfg.tot, "groups": cfg.groups,
                    "EPS": cfg.EPS, "KERNEL_EPS": KERNEL_EPS,
                    "THETA": cfg.THETA}
    with open(outdir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    nb = sum(v["M"] * v["K"] for v in index["gemms"].values())
    print(f"[dump-weights] {len(index['gemms'])} GEMM weights ({nb/1e6:.0f}M int8), "
          f"{len(index['gammas'])} gammas -> {outdir}")


def _jit_artifacts(design_obj, **compile_kwargs):
    """Resolve an @iron.jit design's cached (xclbin, insts) WITHOUT recompiling.

    CompilableDesign.specialize(**kw).compile() is a pure cache lookup on a hit
    (~/.npu/cache/<hash>/{final.xclbin,insts.bin}); it is the only way to learn
    the hash-keyed path from outside, since the hash covers the generator
    bytecode + compile kwargs + tool mtimes."""
    return design_obj.specialize(**compile_kwargs).compile()
GROUP = 256
HEAD_DIM = 128
# The rmsnorm / headnorm xclbins bake eps = 1e-5 (see rms_norm_weighted_bf16.cc /
# rms_norm_head_bf16.cc). We mirror THAT in the precision reference so the bf16
# reference tracks the device; the f16 golden used the model eps (~1e-6) — the
# ~1e-5 gap is far below the cosine gate floor.
KERNEL_EPS = 1e-5


# ── bf16 helpers ────────────────────────────────────────────────────────────
def bf16(x):
    return np.asarray(x).astype(bfloat16).astype(np.float32)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


# ── int8 per-group symmetric quant (matches test_dflash_projection) ─────────
def quantize_group_symmetric(x_f32, bits=8):
    """Per-256-group symmetric int8. x:[rows,K] (K%256==0) -> (q int8, scale[rows,ng])."""
    qmax = (1 << (bits - 1)) - 1
    rows, K = x_f32.shape
    ng = K // GROUP
    xg = x_f32.reshape(rows, ng, GROUP)
    absmax = np.abs(xg).max(axis=2)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    q = np.round(xg / scale[:, :, None]).clip(-qmax, qmax).astype(np.int8)
    return q.reshape(rows, K), scale


def quantize_row_symmetric(x_f32, bits=8):
    """Per-ROW symmetric int8 over the WHOLE K (single scale per row).
    x:[rows,K] -> (q int8 [rows,K], scale[rows])."""
    qmax = (1 << (bits - 1)) - 1
    absmax = np.abs(x_f32).max(axis=1)
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    q = np.round(x_f32 / scale[:, None]).clip(-qmax, qmax).astype(np.int8)
    return q, scale


# ─────────────────────────────────────────────────────────────────────────────
# NPU op wrappers.  Each takes np.float32 in, returns np.float32 out.
# XRTHostRuntime handles are cached per-xclbin (one hw_context each).
# ─────────────────────────────────────────────────────────────────────────────
class NpuOps:
    def __init__(self, count=True, batch_ops=False, proj_mode="group",
                 resident_weights=False, concat_proj=False, attn_all_kv=False,
                 fuse_hnrope=False, fold_norm=False, stack_ctx=False,
                 attn_cores=1):
        # batch_ops   : Lever 1 — run norm/headnorm/rope/swiglu batched (all rows
        #               in ONE dispatch) via the *-b<rows> xclbins instead of the
        #               per-row loop.
        # proj_mode   : Lever 2 — "group" = per-256-group int8 (one dispatch/group,
        #               baseline); "row" = full-K per-row/per-channel int8 (one
        #               dispatch/projection, Lever 2a).
        self.batch_ops = batch_ops
        self.proj_mode = proj_mode
        # concat_proj : Lever 3 — projections sharing an input (q/k/v on x_norm,
        #               gate/up on x2, k_ctx/v_ctx on thp) stacked into ONE GEMM.
        #               Exact; row-proj mode only. Saves 4 dispatches/layer.
        self.concat_proj = concat_proj
        # attn_all_kv : Lever 3 — whole layer's attention in ONE dispatch
        #               (core loops the kv-heads, streaming via ObjectFifo).
        self.attn_all_kv = attn_all_kv
        # attn_cores : Phase 0 item 2 — spread the kv-heads of the whole-layer
        #              attention across this many columns (1 = serial single-core
        #              dflash_attn_all). Same dispatch ABI and buffer layout, so
        #              it is a drop-in xclbin swap and still ONE hw context.
        self.attn_cores = attn_cores
        # fuse_hnrope : Phase D step 1 — TRUE multi-op fusion. headnorm+rope
        #               collapse into one kernel (dflash-hnrope-*), for both q
        #               and k. Saves 2 dispatches/layer (13.6 -> 11.6).
        self.fuse_hnrope = fuse_hnrope
        # fold_norm : Phase D step 2 — the rmsnorm feeding a projection is
        #             ABSORBED into that projection (gamma folded into the
        #             weight, 1/rms folded into the per-row activation scale).
        #             Saves 2 dispatches/layer + the one-time hidden_norm.
        self.fold_norm = fold_norm
        # stack_ctx : Phase D step 3a — merge the q/k/v and k_ctx/v_ctx GEMMs
        #             into ONE by stacking their activation ROWS (exact; needs
        #             fold_norm, since the two halves carry different norms).
        self.stack_ctx = stack_ctx
        # Lever 4 (speed, not dispatch count): keep int8 projection weights
        # resident on the NPU instead of re-uploading ~1 GB every pass.
        self.resident_weights = resident_weights
        # Use the SHARED default cached runtime — the SAME CachedXRTRuntime that
        # @iron.jit's projection/attention go through. npu1 (Phoenix) can only
        # keep a handful of hw_contexts resident; a separate XRTHostRuntime added
        # contexts outside the cache's accounting and CREATE_HWCTX failed (err=-22)
        # once the total exceeded the hardware budget. Routing the primitive
        # xclbins through the shared cache gives one unified LRU (size 6 on npu1)
        # with the Phoenix full-drain + retry-on-exhaustion workaround.
        import aie.utils as _aieutils
        self._rt = _aieutils._get_default_npu_runtime()
        self._npukernels = {}  # stem -> NPUKernel (cheap, reused across loads)
        self.npu_ns = 0  # accumulated on-device (NPU-busy) time, ns
        self.dispatches = 0  # raw NPU kernel invocations
        self.op_dispatches = 0  # logical op-level dispatches
        self._count = count

    def _npukernel(self, stem):
        k = self._npukernels.get(stem)
        if k is None:
            xcl = NPU_DIR / f"{stem}.xclbin"
            instr = NPU_DIR / f"{stem}-instr.bin"
            if not xcl.exists() or not instr.exists():
                raise FileNotFoundError(f"missing xclbin/instr for {stem}: {xcl}")
            k = NPUKernel(xclbin_path=xcl, insts_path=instr, kernel_name=KERNEL_NAME)
            self._npukernels[stem] = k
        return k

    def _load(self, stem):
        # Cached load: returns the resident handle if the context is cached, else
        # creates one (evicting LRU / draining on Phoenix as needed).
        self._cur_stem = stem
        MANIFEST.kernel(stem, NPU_DIR / f"{stem}.xclbin",
                        NPU_DIR / f"{stem}-instr.bin")
        return self._rt.load(self._npukernel(stem))

    def release(self):
        """No-op: the shared cached runtime manages context lifetime/eviction."""
        return

    def _run(self, handle, tensors, roles=None):
        if MANIFEST.enabled:
            r = roles or ["?"] * len(tensors)
            MANIFEST.dispatch(getattr(self, "_cur_stem", "?"),
                              [_tdesc(t, r[i]) for i, t in enumerate(tensors)])
        res = self._rt.run(handle, tensors)
        if not res.is_success():
            raise RuntimeError(f"NPU kernel failed: {res.ret}")
        self.dispatches += 1
        self.npu_ns += int(getattr(res, "npu_time", 0) or 0)
        return res

    # ── rmsnorm  [rows, H] weighted, over last dim ──────────────────────────
    def rmsnorm(self, x, weight):
        x = np.ascontiguousarray(x, np.float32)
        rows, H = x.shape
        if self.batch_ops:
            # Lever 1: one dispatch for all rows. The -b<rows> xclbin bakes
            # N_div_n=rows; weight is a tiled input so replicate gamma per row.
            h = self._load(f"qwen35-rmsnorm-{H}-b{rows}")
            xin = x.reshape(-1).astype(bfloat16)
            wrep = np.tile(bf16(weight).astype(bfloat16), rows)
            t_in = XRTTensor(xin, dtype=bfloat16, device="cpu")
            t_w = XRTTensor(wrep, dtype=bfloat16, device="cpu")
            t_out = XRTTensor((rows * H,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_w, t_out], ["in", "weight", "out"])
            t_out.to("cpu")
            self.op_dispatches += 1
            return t_out.numpy().astype(bfloat16).astype(np.float32).reshape(rows, H)
        h = self._load(f"qwen35-rmsnorm-{H}")
        w = XRTTensor(bf16(weight).astype(bfloat16), dtype=bfloat16, device="cpu")
        out = np.empty((rows, H), np.float32)
        for r in range(rows):
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((H,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, w, t_out], ["in", "weight", "out"])
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── headnorm  [rows, nh*hd] per-head rmsnorm ────────────────────────────
    def headnorm(self, x, weight, which, nh):
        x = np.ascontiguousarray(x, np.float32)
        rows, ND = x.shape
        assert ND == nh * HEAD_DIM
        w = XRTTensor(bf16(weight).astype(bfloat16), dtype=bfloat16, device="cpu")
        if self.batch_ops:
            # Lever 1: one dispatch, N_div_n=rows*nh head-tiles; weight is a
            # once-acquired param (shared across all head-tiles) — unchanged.
            h = self._load(f"qwen35-headnorm-{which}-{nh}h{HEAD_DIM}d-b{rows}")
            t_in = XRTTensor(x.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((rows * ND,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_out, w], ["in", "out", "weight"])
            t_out.to("cpu")
            self.op_dispatches += 1
            return t_out.numpy().astype(bfloat16).astype(np.float32).reshape(rows, ND)
        h = self._load(f"qwen35-headnorm-{which}-{nh}h{HEAD_DIM}d")
        out = np.empty((rows, ND), np.float32)
        for r in range(rows):
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((ND,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_out, w], ["in", "out", "weight"])
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── rope (full neox)  [rows, nh*hd] with per-row position ───────────────
    def rope(self, x, positions, which, nh, theta):
        x = np.ascontiguousarray(x, np.float32)
        rows, ND = x.shape
        assert ND == nh * HEAD_DIM
        if self.batch_ops:
            # Lever 1: one dispatch for all rows via dflash_rope_batched_bf16
            # (cs is a 2nd TILED input, so each head-tile gets its row's cs).
            # Pack cs[rows*nh*head_dim]: per row repeat that position's cs over
            # all nh heads; the input tile is head_dim == cs tile size.
            h = self._load(f"dflash-rope-{which}-{nh}h{HEAD_DIM}d-b{rows}")
            cs_rows = np.empty((rows, ND), np.float32)
            for r in range(rows):
                cs_r = np.asarray(_make_cs_buf(HEAD_DIM, int(positions[r]), theta), np.float32)
                cs_rows[r] = np.tile(cs_r, nh)
            t_in = XRTTensor(x.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
            t_cs = XRTTensor(cs_rows.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((rows * ND,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_cs, t_out], ["in", "cs", "out"])
            t_out.to("cpu")
            self.op_dispatches += 1
            return t_out.numpy().astype(bfloat16).astype(np.float32).reshape(rows, ND)
        h = self._load(f"dflash-rope-{which}-{nh}h{HEAD_DIM}d")
        out = np.empty((rows, ND), np.float32)
        for r in range(rows):
            cs = _make_cs_buf(HEAD_DIM, int(positions[r]), theta)  # bf16 [head_dim]
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((ND,), dtype=bfloat16, device="cpu")
            t_cs = XRTTensor(cs, dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_out, t_cs], ["in", "out", "cs"])
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── FUSED headnorm + rope (full neox)  [rows, nh*hd] ────────────────────
    def headnorm_rope(self, x, weight, positions, which, nh, theta):
        """Phase D step 1: headnorm and rope in ONE dispatch (was two).

        Falls back to the separate ops when --fuse-hnrope is off, so the fused
        and unfused paths stay comparable in one binary.

        Tiling: 2*HEAD_DIM per tile == one PAIR of heads, because an AIE2 core
        tile has only 2 inbound DMA channels — tiled-qk + tiled-cs + a weight
        param would need 3 and will not place. The widened tile lets gamma and
        the row's cs share the second stream as [gamma | cs]. Both heads of a
        pair are in the same row (nh even), so they share the position/cs.
        """
        if not self.fuse_hnrope:
            x = self.headnorm(x, weight, which, nh)
            return self.rope(x, positions, which, nh, theta)
        x = np.ascontiguousarray(x, np.float32)
        rows, ND = x.shape
        assert ND == nh * HEAD_DIM
        assert nh % 2 == 0, f"fused headnorm+rope needs even n_heads, got {nh}"
        h = self._load(f"dflash-hnrope-{which}-{nh}h{HEAD_DIM}d-b{rows}")
        # coeff[r, pair, 0:HD] = gamma ; coeff[r, pair, HD:2*HD] = cs(position[r])
        g = bf16(weight).astype(np.float32)
        coeff = np.empty((rows, nh // 2, 2 * HEAD_DIM), np.float32)
        coeff[:, :, :HEAD_DIM] = g
        for r in range(rows):
            cs_r = np.asarray(_make_cs_buf(HEAD_DIM, int(positions[r]), theta), np.float32)
            coeff[r, :, HEAD_DIM:] = cs_r
        t_in = XRTTensor(x.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
        t_co = XRTTensor(coeff.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
        t_out = XRTTensor((rows * ND,), dtype=bfloat16, device="cpu")
        self._run(h, [t_in, t_co, t_out], ["in", "coeff", "out"])
        t_out.to("cpu")
        self.op_dispatches += 1
        return t_out.numpy().astype(bfloat16).astype(np.float32).reshape(rows, ND)

    # ── swiglu  silu(gate)*up  [rows, I] ────────────────────────────────────
    def swiglu(self, gate, up):
        gate = np.ascontiguousarray(gate, np.float32)
        up = np.ascontiguousarray(up, np.float32)
        rows, I = gate.shape
        if self.batch_ops:
            # Lever 1: one dispatch over rows*I (parallel silu(gate)*up).
            h = self._load(f"qwen35-swiglu-{I}-b{rows}")
            t_g = XRTTensor(gate.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
            t_u = XRTTensor(up.reshape(-1).astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((rows * I,), dtype=bfloat16, device="cpu")
            self._run(h, [t_g, t_u, t_out], ["gate", "up", "out"])
            t_out.to("cpu")
            self.op_dispatches += 1
            return t_out.numpy().astype(bfloat16).astype(np.float32).reshape(rows, I)
        h = self._load(f"qwen35-swiglu-{I}")
        out = np.empty((rows, I), np.float32)
        for r in range(rows):
            t_g = XRTTensor(gate[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_u = XRTTensor(up[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((I,), dtype=bfloat16, device="cpu")
            self._run(h, [t_g, t_u, t_out], ["gate", "up", "out"])
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── int8 projection  Y[rows,N] = X[rows,K] @ W[N,K]^T ───────────────────
    def project(self, x, W, name):
        """Dispatch to the active projection mode (Lever 2)."""
        if self.proj_mode == "row":
            qw, sw = W.qproj_row(name)
            # Weight resident on the NPU (uploaded once) when enabled.
            dev = W.device_weight(name) if self.resident_weights else None
            return self._proj_row(x, qw, sw, dev)
        return self._proj_group(x, *W.qproj(name))

    def project_concat(self, x, W, names):
        """Lever 3 (concat-GEMM): ONE dispatch for projections sharing an input.

        q/k/v all consume x_norm; gate/up both consume x2; k_ctx/v_ctx both consume
        thp — stacking their weights along the output dim turns N GEMMs into 1.
        EXACT (per-row int8 scales are per-output-channel, so stacking changes
        nothing). Falls back to sequential `project` in per-group mode.
        Returns the outputs split back out, in `names` order.
        """
        if self.proj_mode != "row" or not self.concat_proj:
            return [self.project(x, W, n) for n in names]
        qw, sw, sizes = W.qproj_row_concat(names)
        dev = W.device_weight_concat(names) if self.resident_weights else None
        y = self._proj_row(x, qw, sw, dev)          # [rows, sum(N)]
        out, off = [], 0
        for n in sizes:
            out.append(y[:, off:off + n])
            off += n
        return out

    def project_concat_prenorm(self, x, W, names, gamma_name, eps=KERNEL_EPS):
        """Phase D step 2: the preceding rmsnorm ABSORBED into the projection.

        rmsnorm(x)[r,k] = x[r,k] * gamma[k] / rms(x[r]) is the product of a
        per-INPUT-CHANNEL scale (gamma) and a per-ROW scale (1/rms), and the
        int8 projection can absorb both without a single extra dispatch:

          * gamma multiplies the activation elementwise, on the host, alongside
            the per-row int8 quant that already happens there — free.
          * 1/rms is a per-ROW scale, and per-row int8 quantization is invariant
            to a per-row rescale, so it never has to touch the activation at
            all: quantize (x*gamma) and divide the resulting per-row activation
            SCALE by rms. Identical int8 codes, exact.

        So the rmsnorm dispatch disappears with no custom GEMM kernel and, just
        as importantly, no change to the weight or its quantization.

        Saves 2 dispatches/layer (both rmsnorms) plus the one-time hidden_norm.

        Why not fold gamma into the WEIGHT instead (W'[n,k] = W[n,k]*gamma[k]),
        which is the more obvious "fuse into the GEMM" move: it is algebraically
        just as exact, but the weight then gets int8-quantized AFTER folding, and
        a per-INPUT-channel scale is exactly what per-OUTPUT-channel quantization
        cannot absorb. Measured on the 9B drafter it cost ~0.0035 cosine on the
        full body (0.998132 -> 0.994632 vs golden), with the precision reference
        degrading identically (0.998592 -> 0.994792) — i.e. inherent to the
        quantization, not a kernel artifact. The activation-side fold below has
        no such cost.

        Numerically this is not bit-identical to running the two ops separately;
        it moves in the ACCURATE direction on both counts that change (rms is
        computed in f32 on the host rather than bf16 on device, and the normed
        activation is never rounded to bf16).
        """
        if not self.fold_norm:
            xn = self.rmsnorm(x, W.raw(gamma_name))
            return self.project_concat(xn, W, names)
        x = np.ascontiguousarray(x, np.float32)
        rms = np.sqrt(np.mean(x ** 2, axis=1) + eps).astype(np.float32)  # [rows]
        xg = x * W.raw(gamma_name).astype(np.float32)[None, :]
        qw, sw, sizes = W.qproj_row_concat(names)
        dev = W.device_weight_concat(names) if self.resident_weights else None
        y = self._proj_row(xg, qw, sw, dev, act_div=rms)   # [rows, sum(N)]
        out, off = [], 0
        for n in sizes:
            out.append(y[:, off:off + n])
            off += n
        return out

    def project_qkv_stacked(self, hidden, thp, W, p, cfg):
        """Phase D step 3a: the q/k/v projection and the k_ctx/v_ctx projection
        merged into ONE GEMM by stacking their ROWS.

        Both consume the SAME per-layer [q|k|v] weight and differ only in their
        activation (block hidden, 16 rows, vs the context projection thp, 32
        rows). Per-row int8 quantization is independent per row, so vstacking
        the two activations and running a single GEMM is EXACT — bit-identical
        to the two separate GEMMs for every output that is kept.

        This is only expressible because step 2 moved gamma to the activation
        side: the two halves carry DIFFERENT norms (input_layernorm vs
        hidden_norm), which a shared weight could not have absorbed, but which
        per-row activation scaling handles for free.

        Rows are stacked [thp ; hidden] so k/v come out already in the
        ctx-then-noise order the attention expects. The q outputs of the 32 thp
        rows are computed and discarded — ~2.1x the MACs in this GEMM, but the
        25 MB of int8 weight streams ONCE instead of twice, which is the actual
        bound. Saves 1 dispatch/layer.
        """
        names = [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                 p + "self_attn.v_proj.weight"]
        value, rms_thp, gamma_thp = thp
        hidden = np.ascontiguousarray(hidden, np.float32)
        rms_hid = np.sqrt(np.mean(hidden ** 2, axis=1) + cfg.EPS).astype(np.float32)
        act_hid = hidden * W.raw(p + "input_layernorm.weight").astype(np.float32)[None, :]
        act_thp = value.astype(np.float32) * W.raw(gamma_thp).astype(np.float32)[None, :]
        stacked = np.vstack([act_thp, act_hid])                 # [tot, H]
        act_div = np.concatenate([rms_thp, rms_hid])            # [tot]
        qw, sw, sizes = W.qproj_row_concat(names)
        dev = W.device_weight_concat(names) if self.resident_weights else None
        y = self._proj_row(stacked, qw, sw, dev, act_div=act_div)
        nq, nk, _nv = sizes
        q = y[cfg.L:, :nq]                    # block rows only
        k = y[:, nq:nq + nk]                  # [tot] ctx rows then noise rows
        v = y[:, nq + nk:]
        return q, k, v

    def project_concat_thp(self, thp, W, names):
        """k_ctx/v_ctx from the context projection `thp`.

        `thp` is a bundle (value, rms_or_None, gamma_name_or_None) from
        compute_thp. With fold_norm on, the value is the RAW fc output and the
        hidden_norm is absorbed here exactly as in project_concat_prenorm —
        which is what lets compute_thp skip its rmsnorm dispatch entirely.
        """
        value, rms, gamma_name = thp
        if not self.fold_norm:
            return self.project_concat(value, W, names)
        vg = value.astype(np.float32) * W.raw(gamma_name).astype(np.float32)[None, :]
        qw, sw, sizes = W.qproj_row_concat(names)
        dev = W.device_weight_concat(names) if self.resident_weights else None
        y = self._proj_row(vg, qw, sw, dev, act_div=rms)
        out, off = [], 0
        for n in sizes:
            out.append(y[:, off:off + n])
            off += n
        return out

    def _manifest_gemm(self, M, K, N):
        """Record one int8 `int_matmul` dispatch (C[M,N] = A[M,K]·B[N,K]^T).

        The tile split mirrors matmul_npu_resident so the recorded CompileTime
        args are exactly the ones that keyed the JIT cache entry."""
        if not MANIFEST.enabled:
            return
        m, k, n = design._tiles_for(M, K, N, design.DEFAULT_TILE)
        ca = {"M": M, "K": K, "N": N, "m": m, "k": k, "n": n,
              "dtype_in_str": "i8", "dtype_out_str": "i32", "b_col_maj": 1}
        key = f"int_matmul:M{M}_K{K}_N{N}_m{m}_k{k}_n{n}"
        if key not in MANIFEST.kernels:
            xcl, ins = _jit_artifacts(design.int_matmul, **ca)
            MANIFEST.kernel(key, xcl, ins, ca)
        MANIFEST.dispatch(key, [
            {"role": "A_weight_int8", "dtype": "int8", "shape": [M, K], "bytes": M * K},
            {"role": "B_act_int8", "dtype": "int8", "shape": [N, K], "bytes": N * K},
            {"role": "C_out_int32", "dtype": "int32", "shape": [M, N], "bytes": M * N * 4},
        ])

    def _manifest_attn_all(self, q_len, kv_len, n_iters, n_cores=1):
        """Record the one-dispatch whole-layer attention.

        n_cores=1 -> dflash_attn_all (serial); n_cores>1 -> dflash_attn_mc, whose
        kv-heads are split across columns. Both keys start with "dflash_attn_all"
        so dflash_body_native.rs finds either by its existing prefix match, and
        both take the same (Q, KV, O) buffers at the same sizes.
        """
        if not MANIFEST.enabled:
            return
        ca = {"q_len": q_len, "kv_len": kv_len, "n_iters": n_iters}
        if n_cores > 1:
            from build_dflash_attention_sc import dflash_attn_mc as _attn_fn
            ca["n_cores"] = n_cores
            key = f"dflash_attn_all_mc{n_cores}:q{q_len}_kv{kv_len}_it{n_iters}"
        else:
            from build_dflash_attention_sc import dflash_attn_all as _attn_fn
            key = f"dflash_attn_all:q{q_len}_kv{kv_len}_it{n_iters}"
        if key not in MANIFEST.kernels:
            xcl, ins = _jit_artifacts(_attn_fn, **ca)
            MANIFEST.kernel(key, xcl, ins, ca)
        qn = n_iters * q_len * HEAD_DIM
        kvn = n_iters * 2 * kv_len * HEAD_DIM
        MANIFEST.dispatch(key, [
            {"role": "Q_bf16", "dtype": "bfloat16", "shape": [qn], "bytes": qn * 2},
            {"role": "KV_bf16", "dtype": "bfloat16", "shape": [kvn], "bytes": kvn * 2},
            {"role": "O_bf16", "dtype": "bfloat16", "shape": [qn], "bytes": qn * 2},
        ])

    def _proj_group(self, x, qw, sw):
        """Per-256-group int8 (baseline): one dispatch per group.
        qw int8 [N,K], sw f32 [N,ng] pre-quantized weight."""
        self.release()  # free XRT columns for the @iron.jit matmul
        x = np.ascontiguousarray(x, np.float32)
        rows, K = x.shape
        N = qw.shape[0]
        ng = K // GROUP
        qx, sx = quantize_group_symmetric(x, 8)  # [rows,K], [rows,ng]
        Y = np.zeros((rows, N), np.float32)
        for g in range(ng):
            Wg = qw[:, g * GROUP:(g + 1) * GROUP]  # [N,256]
            Xg = qx[:, g * GROUP:(g + 1) * GROUP]  # [rows,256]
            C, _tile = design.matmul_npu(Wg, Xg)   # [N, rows] int32
            self.dispatches += 1
            Y += (sx[:, g][:, None] * sw[:, g][None, :]) * C.T.astype(np.float32)
        self.op_dispatches += 1
        return Y

    def _proj_row(self, x, qw, sw, dev=None, act_div=None):
        """Lever 2a: full-K per-ROW/per-CHANNEL int8 — ONE dispatch/projection.
        Single int8 quant over the whole K (one scale per activation row and per
        output channel); rank-1 rescale C·(a_scale[m]⊗w_scale[n]).
        qw int8 [N,K] per-channel, sw f32 [N] per-channel scale.
        `dev` = (A_t, M, K) NPU-resident weight from design.upload_int8().
        `act_div` = optional per-row divisor folded into the activation scale
        (Phase D step 2: the rmsnorm 1/rms, see project_concat_prenorm)."""
        self.release()  # free XRT columns for the @iron.jit matmul
        x = np.ascontiguousarray(x, np.float32)
        rows, K = x.shape
        N = qw.shape[0]
        qx, sx = quantize_row_symmetric(x, 8)      # [rows,K], [rows]
        if act_div is not None:
            sx = sx / np.asarray(act_div, np.float32)
        if dev is not None:
            A_t, M_w, K_w = dev
            self._manifest_gemm(M_w, K_w, rows)
            C, _tile = design.matmul_npu_resident(A_t, M_w, K_w, qx)  # C[N,rows] i32
            self.npu_ns += int(design.LAST_NPU_US * 1000.0)
            _maybe_dump_op(qw, qx, C, M_w, K_w, rows)
        else:
            self._manifest_gemm(N, K, rows)
            C, _tile = design.matmul_npu(qw, qx)    # A=W[N,K], B=X[rows,K] -> C[N,rows] i32
        self.dispatches += 1
        Y = (sw[:, None] * sx[None, :]) * C.astype(np.float32)  # [N,rows]
        self.op_dispatches += 1
        return Y.T.copy()                            # [rows,N]

    # legacy alias (op-by-op paths that pre-fetch the group weight)
    def proj(self, x, qw, sw):
        return self._proj_group(x, qw, sw)

    # ── attention  q[B,NH,HD] k/v[tot,NKV,HD] -> ctx[B, NH*HD] ──────────────
    def attention(self, q, k, v, groups):
        """GQA head-batched: one dispatch per KV-head, not per q-head.

        Attention is per-query independent given K/V, and the `groups` q-heads
        that share a kv-head attend to the SAME K/V — so their queries can be
        STACKED into a single q_len = groups*B call. Mathematically exact (verified
        cos=1.00000 vs the bf16 reference on all 8 kv-heads,
        test_dflash_attention_batched_npu.py) and needs NO kernel change: the
        kernel already takes q_len as a compile-time constant.
        Dispatches/layer: NH (32) -> NKV (8). Tile budget at groups=4, kv_len=48:
        Q 16 KB + KV 24 KB + O 16 KB = 56 KB of 64 KB.
        """
        self.release()  # free XRT columns for the @iron.jit attention
        B, NH, HD = q.shape
        tot = k.shape[0]
        NKV = NH // groups
        q_len = groups * B
        if self.attn_all_kv:
            # ONE dispatch for the whole layer: the core loops the NKV kv-heads,
            # streaming each group's Q/KV/O through the ObjectFifos (tile holds one
            # iteration, 56 KB). Verified cos=1.00000 vs bf16 ref on all 32 q-heads.
            self._manifest_attn_all(q_len, tot, NKV, self.attn_cores)
            ctx = run_attn_all_kv(q, k, v, groups, self.attn_cores)
            self.dispatches += 1
            self.op_dispatches += 1
            return ctx
        ctx = np.empty((B, NH * HD), np.float32)
        for kvh in range(NKV):
            heads = range(kvh * groups, (kvh + 1) * groups)
            qs = np.concatenate([q[:, h, :] for h in heads], axis=0)  # [groups*B, HD]
            o = run_attn_head(qs, k[:, kvh, :], v[:, kvh, :], q_len, tot)  # [groups*B, HD]
            self.dispatches += 1
            for i, h in enumerate(heads):
                ctx[:, h * HD:(h + 1) * HD] = o[i * B:(i + 1) * B, :]
        self.op_dispatches += 1
        return ctx


def _make_cs_buf(n_rot, pos, theta):
    """[cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}] at position pos, bf16."""
    n_rot2 = n_rot // 2
    i = np.arange(n_rot2, dtype=np.float64)
    freq = 1.0 / (theta ** (2.0 * i / n_rot))
    ang = pos * freq
    return np.concatenate([np.cos(ang), np.sin(ang)]).astype(bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# bf16/int8-precision numpy reference (mirrors each op's on-device precision).
# ─────────────────────────────────────────────────────────────────────────────
def ref_rmsnorm(x, weight, eps=KERNEL_EPS):
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return bf16((x / rms) * weight.astype(np.float32))


def ref_headnorm(x, weight, nh, eps=KERNEL_EPS):
    rows = x.shape[0]
    x32 = x.astype(np.float32).reshape(rows, nh, HEAD_DIM)
    w = weight.astype(np.float32)
    rms = np.sqrt(np.mean(x32**2, axis=2, keepdims=True) + eps)
    return bf16(((x32 / rms) * w[None, None, :]).reshape(rows, nh * HEAD_DIM))


def ref_rope(x, positions, nh, theta):
    rows = x.shape[0]
    n_rot2 = HEAD_DIM // 2
    out = x.astype(np.float32).reshape(rows, nh, HEAD_DIM).copy()
    for r in range(rows):
        cs = _make_cs_buf(HEAD_DIM, int(positions[r]), theta).astype(np.float32)
        c, s = cs[:n_rot2], cs[n_rot2:]
        xi = out[r, :, :n_rot2].copy()
        yi = out[r, :, n_rot2:].copy()
        out[r, :, :n_rot2] = xi * c[None, :] - yi * s[None, :]
        out[r, :, n_rot2:] = yi * c[None, :] + xi * s[None, :]
    return bf16(out.reshape(rows, nh * HEAD_DIM))


def ref_swiglu(gate, up):
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    silu = g * (1.0 / (1.0 + np.exp(-g)))
    return bf16(silu * u)


def ref_proj_int8(x, qw, sw):
    rows, K = x.shape
    N = qw.shape[0]
    ng = K // GROUP
    qx, sx = quantize_group_symmetric(x.astype(np.float32), 8)
    qwg = qw.astype(np.int64).reshape(N, ng, GROUP)
    qxg = qx.astype(np.int64).reshape(rows, ng, GROUP)
    Y = np.zeros((rows, N), np.float32)
    for g in range(ng):
        C = (qxg[:, g, :] @ qwg[:, g, :].T).astype(np.float32)  # [rows,N]
        Y += (sx[:, g][:, None] * sw[:, g][None, :]) * C
    return Y


def ref_proj_int8_row(x, qw, sw):
    """bf16/int8 mirror of NpuOps._proj_row (full-K per-row/per-channel int8)."""
    qx, sx = quantize_row_symmetric(x.astype(np.float32), 8)
    C = (qx.astype(np.int64) @ qw.astype(np.int64).T).astype(np.float32)  # [rows,N]
    return (sx[:, None] * sw[None, :]) * C


def ref_project(W, name, x, proj_mode):
    if proj_mode == "row":
        return ref_proj_int8_row(x, *W.qproj_row(name))
    return ref_proj_int8(x, *W.qproj(name))


def ref_proj_int8_row_prenorm(xg, qw, sw, rms):
    """Mirror of NpuOps._proj_row with act_div (Phase D step 2 norm folding).
    xg is the gamma-scaled activation; rms is the per-row rmsnorm divisor,
    folded into the per-row activation scale rather than into the values."""
    qx, sx = quantize_row_symmetric(xg.astype(np.float32), 8)
    sx = sx / np.asarray(rms, np.float32)
    C = (qx.astype(np.int64) @ qw.astype(np.int64).T).astype(np.float32)
    return (sx[:, None] * sw[None, :]) * C


def _split(y, sizes):
    out, off = [], 0
    for n in sizes:
        out.append(y[:, off:off + n])
        off += n
    return out


def ref_project_concat_prenorm(W, names, gamma_name, x, eps, fold_norm):
    """Reference twin of NpuOps.project_concat_prenorm (row proj only)."""
    if not fold_norm:
        xn = ref_rmsnorm(x, W.raw(gamma_name), eps)
        return [ref_proj_int8_row(xn, *W.qproj_row(n)) for n in names]
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x ** 2, axis=1) + eps).astype(np.float32)
    xg = x * W.raw(gamma_name).astype(np.float32)[None, :]
    qw, sw, sizes = W.qproj_row_concat(names)
    return _split(ref_proj_int8_row_prenorm(xg, qw, sw, rms), sizes)


def ref_project_qkv_stacked(W, p, cfg, hidden, thp):
    """Reference twin of NpuOps.project_qkv_stacked."""
    names = [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
             p + "self_attn.v_proj.weight"]
    value, rms_thp, gamma_thp = thp
    hidden = hidden.astype(np.float32)
    rms_hid = np.sqrt(np.mean(hidden ** 2, axis=1) + cfg.EPS).astype(np.float32)
    act_hid = hidden * W.raw(p + "input_layernorm.weight").astype(np.float32)[None, :]
    act_thp = value.astype(np.float32) * W.raw(gamma_thp).astype(np.float32)[None, :]
    stacked = np.vstack([act_thp, act_hid])
    act_div = np.concatenate([rms_thp, rms_hid])
    qw, sw, sizes = W.qproj_row_concat(names)
    y = ref_proj_int8_row_prenorm(stacked, qw, sw, act_div)
    nq, nk, _nv = sizes
    return y[cfg.L:, :nq], y[:, nq:nq + nk], y[:, nq + nk:]


def ref_project_thp(W, names, thp, fold_norm):
    """Reference twin of NpuOps.project_concat_thp."""
    value, rms, gamma_name = thp
    if not fold_norm:
        return [ref_proj_int8_row(value, *W.qproj_row(n)) for n in names]
    vg = value.astype(np.float32) * W.raw(gamma_name).astype(np.float32)[None, :]
    qw, sw, sizes = W.qproj_row_concat(names)
    return _split(ref_proj_int8_row_prenorm(vg, qw, sw, rms), sizes)


def ref_attention(q, k, v, groups):
    B, NH, HD = q.shape
    tot = k.shape[0]
    scale = HD ** -0.5
    ctx = np.empty((B, NH * HD), np.float32)
    for h in range(NH):
        kvh = h // groups
        qh = bf16(q[:, h, :]); kh = bf16(k[:, kvh, :]); vh = bf16(v[:, kvh, :])
        scores = (qh @ kh.T) * scale
        w = np.exp(scores - scores.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        ctx[:, h * HD:(h + 1) * HD] = bf16(w @ vh)
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Weight bundle: load safetensors, lazily int8-quantize projection weights.
# ─────────────────────────────────────────────────────────────────────────────
class Weights:
    def __init__(self, path):
        self.W = load_safetensors_f32(path)
        self._q = {}  # name -> (qw int8, sw f32)
        self._dev = {}  # name -> (iron.tensor A_t, M, K) resident int8 weight

    def raw(self, name):
        return self.W[name]

    def qproj(self, name):
        q = self._q.get(name)
        if q is None:
            qw, sw = quantize_group_symmetric(self.W[name].astype(np.float32), 8)
            q = (qw, sw)
            self._q[name] = q
        return q

    def device_weight(self, name):
        """NPU-resident int8 weight (uploaded once) as (A_t, M, K)."""
        d = self._dev.get(name)
        if d is None:
            qw, _sw = self.qproj_row(name)
            d = design.upload_int8(qw)
            self._dev[name] = d
        return d

    def qproj_row(self, name):
        """Per-output-channel (per-row of W) int8 over full K — Lever 2a."""
        key = ("row", name)
        q = self._q.get(key)
        if q is None:
            qw, sw = quantize_row_symmetric(self.W[name].astype(np.float32), 8)
            q = (qw, sw)
            self._q[key] = q
        return q

    def qproj_row_concat(self, names):
        """Lever 3 (concat-GEMM): projections that share an INPUT are stacked along
        the output dim into ONE weight, so they run as a single GEMM.

        EXACT: per-row (per-output-channel) int8 quantization is independent per
        row, so vstacking pre-quantized weights and concatenating their scales is
        bit-identical to projecting separately. Returns (qw_cat[sum(N),K],
        sw_cat[sum(N)], sizes[list of N]). Cached per name-tuple.
        """
        key = ("rowcat", tuple(names))
        q = self._q.get(key)
        if q is None:
            parts = [self.qproj_row(n) for n in names]
            qw = np.concatenate([p[0] for p in parts], axis=0)
            sw = np.concatenate([p[1] for p in parts], axis=0)
            sizes = [p[0].shape[0] for p in parts]
            q = (qw, sw, sizes)
            self._q[key] = q
        return q

    def device_weight_concat(self, names):
        """NPU-resident upload of the concatenated weight (uploaded once)."""
        key = ("rowcat", tuple(names))
        d = self._dev.get(key)
        if d is None:
            qw, _sw, _sizes = self.qproj_row_concat(names)
            d = design.upload_int8(qw)
            self._dev[key] = d
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class Cfg:
    def __init__(self, meta):
        self.H = meta["hidden"]
        self.NH = meta["n_heads"]
        self.NKV = meta["n_kv_heads"]
        self.HD = meta["head_dim"]
        self.I = meta["intermediate"]
        self.NL = meta["n_layers"]
        self.EPS = meta["norm_eps"]
        self.THETA = meta["rope_theta"]
        self.B = meta["block_size"]
        self.L = meta["ctx_len"]
        self.NE = meta["num_extract"]
        self.groups = self.NH // self.NKV
        self.tot = self.L + self.B


def compute_thp(ops, W, cfg, target_hidden, use_ref=False, proj_mode="group",
                fold_norm=False):
    """One-time context projection: thp = hidden_norm(fc(target_hidden)).

    Returns the BUNDLE (value, rms_or_None, gamma_name_or_None) consumed by
    NpuOps.project_concat_thp / ref_project_thp. With fold_norm the hidden_norm
    is not applied here at all — the value stays raw and the norm is absorbed
    into the per-layer k_ctx/v_ctx projection, saving this rmsnorm dispatch.
    """
    th_flat = target_hidden.reshape(cfg.L, cfg.NE * cfg.H)
    gamma_name = "hidden_norm.weight"
    if use_ref:
        thp = ref_project(W, "fc.weight", th_flat, proj_mode)
        if fold_norm:
            rms = np.sqrt(np.mean(thp.astype(np.float32) ** 2, axis=1) + cfg.EPS)
            return (thp, rms.astype(np.float32), gamma_name)
        return (ref_rmsnorm(thp, W.raw(gamma_name), cfg.EPS), None, None)
    thp = ops.project(th_flat, W, "fc.weight")
    if fold_norm:
        rms = np.sqrt(np.mean(thp.astype(np.float32) ** 2, axis=1) + cfg.EPS)
        return (thp, rms.astype(np.float32), gamma_name)
    return (ops.rmsnorm(thp, W.raw(gamma_name)), None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Full unfused body
# ─────────────────────────────────────────────────────────────────────────────
def run_body(ops, W, cfg, noise, thp, per_layer_out=None):
    hidden = noise.astype(np.float32).copy()
    for li in range(cfg.NL):
        p = f"layers.{li}."
        residual = hidden
        if ops.stack_ctx:
            # Phase D step 3a: q/k/v and k_ctx/v_ctx merged into ONE GEMM by
            # stacking their rows; k/v come out already ctx-then-noise ordered.
            q, k, v = ops.project_qkv_stacked(hidden, thp, W, p, cfg)
        else:
            # Lever 3 concat-GEMM: q/k/v share x_norm -> 1 dispatch; k_ctx/v_ctx share thp -> 1.
            # Phase D step 2: the input_layernorm is absorbed into that GEMM.
            q, k_noise, v_noise = ops.project_concat_prenorm(
                hidden, W, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                            p + "self_attn.v_proj.weight"],
                p + "input_layernorm.weight", cfg.EPS)
            k_ctx, v_ctx = ops.project_concat_thp(
                thp, W, [p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"])
            k = np.concatenate([k_ctx, k_noise], axis=0)  # [tot, NKV*HD]
            v = np.concatenate([v_ctx, v_noise], axis=0)

        # Phase D step 1: headnorm+rope fused into one dispatch each (q, k).
        q = ops.headnorm_rope(q, W.raw(p + "self_attn.q_norm.weight"),
                              np.arange(cfg.L, cfg.L + cfg.B), "q", cfg.NH, cfg.THETA)
        k = ops.headnorm_rope(k, W.raw(p + "self_attn.k_norm.weight"),
                              np.arange(0, cfg.tot), "k", cfg.NKV, cfg.THETA)

        qh = q.reshape(cfg.B, cfg.NH, cfg.HD)
        kh = k.reshape(cfg.tot, cfg.NKV, cfg.HD)
        vh = v.reshape(cfg.tot, cfg.NKV, cfg.HD)
        ctx = ops.attention(qh, kh, vh, cfg.groups)

        attn_proj = ops.project(ctx, W, p + "self_attn.o_proj.weight")
        hidden = residual + attn_proj

        residual = hidden
        # Lever 3 concat-GEMM: gate/up share x2 -> 1 dispatch.
        # Phase D step 2: the post_attention_layernorm is absorbed into it.
        gate, up = ops.project_concat_prenorm(
            hidden, W, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"],
            p + "post_attention_layernorm.weight", cfg.EPS)
        s = ops.swiglu(gate, up)
        d = ops.project(s, W, p + "mlp.down_proj.weight")
        hidden = residual + d
        if per_layer_out is not None:
            per_layer_out.append(hidden.copy())
    final = ops.rmsnorm(hidden, W.raw("norm.weight"))
    return final


def materialize_thp(thp, W):
    """The normed thp value, for comparison against the golden checkpoint.
    With fold_norm the bundle holds the raw fc output plus its rms, so rebuild
    normed = value * gamma / rms; otherwise the value is already normed."""
    value, rms, gamma_name = thp
    if rms is None:
        return value
    return bf16((value.astype(np.float32) / rms[:, None]) * W.raw(gamma_name).astype(np.float32))


def proj_weight_names(cfg):
    """Every projection weight the body touches (one-time int8 quant targets)."""
    names = ["fc.weight"]
    for li in range(cfg.NL):
        p = f"layers.{li}."
        names += [p + n for n in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
    return names


def prequantize(W, cfg, proj_mode, resident=False, fold_norm=False):
    """Lever-4 speed win: int8-quantize every projection weight ONCE, up front,
    so the timed block forward does not pay for it. In a real deployment this is
    an offline step (the .hfq sidecar already stores quantized weights)."""
    t0 = time.perf_counter()
    n_elem = 0
    for name in proj_weight_names(cfg):
        if proj_mode == "row":
            qw, _ = W.qproj_row(name)
        else:
            qw, _ = W.qproj(name)
        n_elem += qw.size
    dt = time.perf_counter() - t0
    print(f"  [setup] pre-quantized {len(proj_weight_names(cfg))} projection weights "
          f"({n_elem/1e6:.0f}M elems) in {dt:.1f} s")
    if resident:
        t1 = time.perf_counter()
        for name in proj_weight_names(cfg):
            W.device_weight(name)
        du = time.perf_counter() - t1
        print(f"  [setup] uploaded {n_elem/1e6:.0f}M int8 weight elems to the NPU "
              f"once in {du:.1f} s")
        dt += du
    return dt


def ref_body(W, cfg, noise, thp_ref, per_layer_out=None, proj_mode="group",
             fold_norm=False, stack_ctx=False):
    """bf16/int8-precision numpy mirror of run_body (no NPU)."""
    hidden = noise.astype(np.float32).copy()
    for li in range(cfg.NL):
        p = f"layers.{li}."
        residual = hidden
        if proj_mode == "row" and stack_ctx:
            q, k, v = ref_project_qkv_stacked(W, p, cfg, hidden, thp_ref)
            k_ctx = k_noise = v_ctx = v_noise = None
        elif proj_mode == "row":
            q, k_noise, v_noise = ref_project_concat_prenorm(
                W, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                    p + "self_attn.v_proj.weight"],
                p + "input_layernorm.weight", hidden, cfg.EPS, fold_norm)
            k_ctx, v_ctx = ref_project_thp(
                W, [p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"],
                thp_ref, fold_norm)
        else:
            xn = ref_rmsnorm(hidden, W.raw(p + "input_layernorm.weight"), cfg.EPS)
            q = ref_project(W, p + "self_attn.q_proj.weight", xn, proj_mode)
            k_noise = ref_project(W, p + "self_attn.k_proj.weight", xn, proj_mode)
            v_noise = ref_project(W, p + "self_attn.v_proj.weight", xn, proj_mode)
            k_ctx = ref_project(W, p + "self_attn.k_proj.weight", thp_ref[0], proj_mode)
            v_ctx = ref_project(W, p + "self_attn.v_proj.weight", thp_ref[0], proj_mode)

        q = ref_headnorm(q, W.raw(p + "self_attn.q_norm.weight"), cfg.NH, cfg.EPS)
        if k_ctx is not None:
            k = np.concatenate([k_ctx, k_noise], axis=0)
            v = np.concatenate([v_ctx, v_noise], axis=0)
        k = ref_headnorm(k, W.raw(p + "self_attn.k_norm.weight"), cfg.NKV, cfg.EPS)

        q = ref_rope(q, np.arange(cfg.L, cfg.L + cfg.B), cfg.NH, cfg.THETA)
        k = ref_rope(k, np.arange(0, cfg.tot), cfg.NKV, cfg.THETA)

        qh = q.reshape(cfg.B, cfg.NH, cfg.HD)
        kh = k.reshape(cfg.tot, cfg.NKV, cfg.HD)
        vh = v.reshape(cfg.tot, cfg.NKV, cfg.HD)
        ctx = ref_attention(qh, kh, vh, cfg.groups)

        attn_proj = ref_project(W, p + "self_attn.o_proj.weight", ctx, proj_mode)
        hidden = residual + attn_proj

        residual = hidden
        if proj_mode == "row":
            gate, up = ref_project_concat_prenorm(
                W, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"],
                p + "post_attention_layernorm.weight", hidden, cfg.EPS, fold_norm)
        else:
            xn2 = ref_rmsnorm(hidden, W.raw(p + "post_attention_layernorm.weight"), cfg.EPS)
            gate = ref_project(W, p + "mlp.gate_proj.weight", xn2, proj_mode)
            up = ref_project(W, p + "mlp.up_proj.weight", xn2, proj_mode)
        s = ref_swiglu(gate, up)
        d = ref_project(W, p + "mlp.down_proj.weight", s, proj_mode)
        hidden = residual + d
        if per_layer_out is not None:
            per_layer_out.append(hidden.copy())
    return ref_rmsnorm(hidden, W.raw("norm.weight"), cfg.EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Golden loader
# ─────────────────────────────────────────────────────────────────────────────
class Golden:
    def __init__(self, gdir):
        self.gdir = Path(gdir)
        self.root = self.gdir.parent  # inputs live one level up (<OUT>/)

    def rust(self, name):
        return np.load(self.gdir / f"rust_{name}.npy").astype(np.float32)

    def inp(self, name):
        return np.load(self.root / f"{name}.npy")


# ─────────────────────────────────────────────────────────────────────────────
# Op-by-op layer-0 validation
# ─────────────────────────────────────────────────────────────────────────────
def op_by_op(ops, W, cfg, G, cos_gate=0.99):
    print("=== op-by-op layer 0 (feed golden input to each stage) ===")
    noise = G.inp("noise_embedding").astype(np.float32)   # [B,H] = layer0 hidden
    target_hidden = G.inp("target_hidden").astype(np.float32)
    p = "layers.0."
    results = []

    def check(tag, out, golden):
        c = cos(out, golden)
        d = float(np.abs(out - golden).max())
        ok = c > cos_gate
        results.append((tag, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:26s} cos={c:.6f}  max_abs={d:.3e}")
        return out

    # thp (one-time) vs golden target_hidden_proj. With fold_norm the bundle
    # carries the RAW fc output; materialize the normed value to compare.
    thp = compute_thp(ops, W, cfg, target_hidden, proj_mode=ops.proj_mode,
                      fold_norm=ops.fold_norm)
    check("thp (fc+hidden_norm)", materialize_thp(thp, W), G.rust("target_hidden_proj"))

    # 1. input_layernorm : noise -> l0_input_norm. Absorbed into the projection
    #    when fold_norm is on, so there is nothing standalone to check then.
    if not ops.fold_norm:
        xn = ops.rmsnorm(noise, W.raw(p + "input_layernorm.weight"))
        xn = check("input_norm", xn, G.rust("l0_input_norm"))

    # 2. projections from GOLDEN input_norm + golden thp, headnorm, rope
    #    checkpoints available: q_roped, k_roped, v.
    #    With fold_norm the norm+projection are one op, so it is fed the RAW
    #    layer-0 hidden (== noise) instead of the golden normed activation —
    #    q_roped/k_roped then validate norm-fold and projection together.
    xn_g = G.rust("l0_input_norm")          # feed golden to isolate downstream
    thp_g = G.rust("target_hidden_proj")
    if ops.stack_ctx:
        q, k_all, v_all = ops.project_qkv_stacked(noise, thp, W, p, cfg)
        k_ctx, k_noise = k_all[:cfg.L], k_all[cfg.L:]
        v_ctx, v_noise = v_all[:cfg.L], v_all[cfg.L:]
    elif ops.fold_norm:
        q, k_noise, v_noise = ops.project_concat_prenorm(
            noise, W, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                       p + "self_attn.v_proj.weight"],
            p + "input_layernorm.weight", cfg.EPS)
        k_ctx, v_ctx = ops.project_concat_thp(
            thp, W, [p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"])
    else:
        q = ops.project(xn_g, W, p + "self_attn.q_proj.weight")
        k_noise = ops.project(xn_g, W, p + "self_attn.k_proj.weight")
        v_noise = ops.project(xn_g, W, p + "self_attn.v_proj.weight")
        k_ctx = ops.project(thp_g, W, p + "self_attn.k_proj.weight")
        v_ctx = ops.project(thp_g, W, p + "self_attn.v_proj.weight")
    k = np.concatenate([k_ctx, k_noise], axis=0)
    v = np.concatenate([v_ctx, v_noise], axis=0)
    q = ops.headnorm_rope(q, W.raw(p + "self_attn.q_norm.weight"),
                          np.arange(cfg.L, cfg.L + cfg.B), "q", cfg.NH, cfg.THETA)
    k = ops.headnorm_rope(k, W.raw(p + "self_attn.k_norm.weight"),
                          np.arange(0, cfg.tot), "k", cfg.NKV, cfg.THETA)
    check("q_roped (proj+hn+rope)", q, G.rust("l0_q_roped"))
    check("k_roped (proj+hn+rope)", k, G.rust("l0_k_roped"))
    check("v (proj)", v, G.rust("l0_v").reshape(cfg.tot, cfg.NKV * cfg.HD))

    # 3. attention from GOLDEN q/k/v -> l0_attn_out
    qh = G.rust("l0_q_roped").reshape(cfg.B, cfg.NH, cfg.HD)
    kh = G.rust("l0_k_roped").reshape(cfg.tot, cfg.NKV, cfg.HD)
    vh = G.rust("l0_v").reshape(cfg.tot, cfg.NKV, cfg.HD)
    ctx = ops.attention(qh, kh, vh, cfg.groups)
    check("attn_out", ctx, G.rust("l0_attn_out"))

    # 4. o_proj + residual from GOLDEN attn_out -> l0_post_attn_residual
    ctx_g = G.rust("l0_attn_out")
    attn_proj = ops.project(ctx_g, W, p + "self_attn.o_proj.weight")
    post = noise + attn_proj
    check("post_attn_residual", post, G.rust("l0_post_attn_residual"))

    # 5. MLP from GOLDEN post_attn_residual -> l0_out
    post_g = G.rust("l0_post_attn_residual")
    if ops.fold_norm:
        gate, up = ops.project_concat_prenorm(
            post_g, W, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"],
            p + "post_attention_layernorm.weight", cfg.EPS)
    else:
        xn2 = ops.rmsnorm(post_g, W.raw(p + "post_attention_layernorm.weight"))
        gate = ops.project(xn2, W, p + "mlp.gate_proj.weight")
        up = ops.project(xn2, W, p + "mlp.up_proj.weight")
    s = ops.swiglu(gate, up)
    d = ops.project(s, W, p + "mlp.down_proj.weight")
    l0_out = post_g + d
    check("l0_out (MLP)", l0_out, G.rust("l0_out"))

    n_ok = sum(1 for _, ok in results if ok)
    print(f"--- op-by-op: {n_ok}/{len(results)} PASS  "
          f"(raw dispatches={ops.dispatches}) ---")
    return all(ok for _, ok in results)


# ─────────────────────────────────────────────────────────────────────────────
# Full unfused body validation (Gate D step 1)
# ─────────────────────────────────────────────────────────────────────────────
def full_body(ops, W, cfg, G, cos_gate=0.99, warm_runs=0):
    print("=== full unfused body ===")
    noise = G.inp("noise_embedding").astype(np.float32)
    target_hidden = G.inp("target_hidden").astype(np.float32)

    # One-time setup (weight int8 quant) is NOT part of the block forward budget.
    prequantize(W, cfg, ops.proj_mode, resident=ops.resident_weights,
                fold_norm=ops.fold_norm)

    t0 = time.perf_counter()
    thp = compute_thp(ops, W, cfg, target_hidden, proj_mode=ops.proj_mode,
                      fold_norm=ops.fold_norm)
    per_layer = []
    final = run_body(ops, W, cfg, noise, thp, per_layer_out=per_layer)
    wall = time.perf_counter() - t0

    # Steady-state (warm) block wall: repeat the body with everything resident
    # (JIT cache warm, contexts loaded, weights already quantized). This is the
    # number the <57 ms Gate-D budget refers to; the first pass still pays
    # xclbin/context load + first-touch costs.
    warm_wall = None
    if warm_runs > 0:
        d0, o0 = ops.dispatches, ops.op_dispatches
        wt = []
        warm_npu = []
        for _ in range(warm_runs):
            n0 = ops.npu_ns
            t1 = time.perf_counter()
            thp_w = compute_thp(ops, W, cfg, target_hidden, proj_mode=ops.proj_mode,
                                fold_norm=ops.fold_norm)
            run_body(ops, W, cfg, noise, thp_w)
            wt.append(time.perf_counter() - t1)
            warm_npu.append(ops.npu_ns - n0)
        warm_wall = min(wt)
        warm_npu_s = warm_npu[wt.index(warm_wall)] / 1e9
        # warm passes are timing-only: don't let them inflate the reported counts
        ops.dispatches, ops.op_dispatches = d0, o0

    # bf16/int8-precision numpy reference (mirrors the active device precision)
    thp_ref = compute_thp(None, W, cfg, target_hidden, use_ref=True,
                          proj_mode=ops.proj_mode, fold_norm=ops.fold_norm)
    per_layer_ref = []
    final_ref = ref_body(W, cfg, noise, thp_ref, per_layer_out=per_layer_ref,
                         proj_mode=ops.proj_mode, fold_norm=ops.fold_norm,
                         stack_ctx=ops.stack_ctx)

    if DUMP_REF_PATH is not None:
        # The native driver gates on the SAME two cosines as Python, and the
        # int8/bf16 precision reference is pure numpy — dump it so the native
        # side compares against an identical array rather than recomputing it.
        np.save(DUMP_REF_PATH, np.ascontiguousarray(final_ref, np.float32))
        print(f"[dump-ref] precision reference -> {DUMP_REF_PATH}")

    gold_final = G.rust("final_block_hidden")
    ok = True
    print("  per-layer (NPU vs golden l{li}_out / vs precision-ref):")
    for li in range(cfg.NL):
        gl = G.rust(f"l{li}_out")
        c_g = cos(per_layer[li], gl)
        c_r = cos(per_layer[li], per_layer_ref[li])
        print(f"    l{li}_out  cos_golden={c_g:.6f}  cos_ref={c_r:.6f}")

    c_golden = cos(final, gold_final)
    c_ref = cos(final, final_ref)
    c_refgold = cos(final_ref, gold_final)
    print(f"  final block_hidden:")
    print(f"    cos vs golden          = {c_golden:.6f}")
    print(f"    cos vs int8/bf16 ref   = {c_ref:.6f}")
    print(f"    (ref  vs golden        = {c_refgold:.6f})")
    nl = cfg.NL
    print(f"  raw NPU dispatches = {ops.dispatches}  |  logical op dispatches = {ops.op_dispatches}")
    print(f"  per-layer: raw≈{ops.dispatches/nl:.1f}  logical≈{ops.op_dispatches/nl:.1f} "
          f"(config: batch_ops={ops.batch_ops} proj_mode={ops.proj_mode})")
    print(f"  wall (cold) = {wall:.2f} s  ({wall*1e3/nl:.0f} ms/layer)")
    if warm_wall is not None:
        print(f"  wall (warm) = {warm_wall*1e3:.0f} ms  ({warm_wall*1e3/nl:.0f} ms/layer)"
              f"   [Gate D budget: <57 ms/block]")
        host = warm_wall - warm_npu_s
        print(f"    NPU-busy  = {warm_npu_s*1e3:.0f} ms  ({warm_npu_s/warm_wall*100:.0f}% of warm wall)")
        print(f"    host/XRT  = {host*1e3:.0f} ms  ({host*1e3/max(ops.dispatches,1):.2f} ms per dispatch"
              f" over {ops.dispatches} dispatches)")

    gate = (c_golden > cos_gate) and (c_ref > cos_gate)
    print(f"=== GATE D step 1: {'MET' if gate else 'NOT MET'} "
          f"(need cos_golden>{cos_gate} AND cos_ref>{cos_gate}) ===")
    return gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden-dir", type=Path, required=True, help="<OUT>/rust dir")
    ap.add_argument("--weights", type=str, required=True, help="safetensors file or HF dir")
    ap.add_argument("--op-by-op", action="store_true", help="layer-0 hand-off checks only")
    ap.add_argument("--cos-gate", type=float, default=0.99)
    ap.add_argument("--batch-ops", action="store_true",
                    help="Lever 1: batch norm/headnorm/rope/swiglu (one dispatch each)")
    ap.add_argument("--resident-weights", action="store_true",
                    help="upload int8 projection weights to the NPU once "
                         "(row proj only); avoids ~1 GB of re-upload per pass")
    ap.add_argument("--warm-runs", type=int, default=0,
                    help="extra timing-only body passes; reports the best as the "
                         "steady-state (warm) block wall")
    ap.add_argument("--proj", choices=["group", "row"], default="group",
                    help="Lever 2: 'group'=per-256-group int8 (baseline); "
                         "'row'=full-K per-row/per-channel int8 (2a, one dispatch/proj)")
    ap.add_argument("--concat-proj", action="store_true",
                    help="Lever 3: stack projections sharing an input (q/k/v, "
                         "gate/up, k_ctx/v_ctx) into ONE GEMM each — exact, "
                         "saves 4 dispatches/layer (row proj only)")
    ap.add_argument("--attn-all-kv", action="store_true",
                    help="Lever 3: whole layer attention in ONE dispatch "
                         "(core loops kv-heads); exact, saves 7 dispatches/layer")
    ap.add_argument("--attn-cores", type=int, default=1,
                    help="Phase 0 item 2: spread the whole-layer attention's "
                         "kv-heads across N columns (1 = serial single core). "
                         "Exact; same dispatch ABI and still one hw context. "
                         "N must divide n_kv and is capped at 4 by the "
                         "per-column shim MM2S budget (2 per column: Q + KV)")
    ap.add_argument("--fold-norm", action="store_true",
                    help="Phase D step 2: absorb each rmsnorm that feeds a "
                         "projection INTO that projection (gamma folded into the "
                         "weight, 1/rms into the per-row activation scale); "
                         "saves 2 dispatches/layer (row proj only)")
    ap.add_argument("--stack-ctx", action="store_true",
                    help="Phase D step 3a: merge the q/k/v and k_ctx/v_ctx GEMMs "
                         "into ONE by stacking activation rows (exact; implies "
                         "--fold-norm); saves 1 dispatch/layer")
    ap.add_argument("--dump-ref", type=Path, default=None,
                    help="save the int8/bf16 precision reference final "
                         "block_hidden (the native driver's second cosine gate)")
    ap.add_argument("--dump-weights", type=Path, default=None,
                    help="emit pre-quantized int8 GEMM weights + gammas + shape "
                         "index for the native Rust body driver, then exit")
    ap.add_argument("--dump-op", type=Path, default=None,
                    help="dump the FIRST int8 projection's exact A/B/C device "
                         "buffers for native bit-for-bit parity (step 2)")
    ap.add_argument("--dump-manifest", type=Path, default=None,
                    help="write the per-dispatch artifact manifest (op name, "
                         "resolved xclbin/insts paths, buffer order+sizes, "
                         "CompileTime args) consumed by the native Rust driver")
    ap.add_argument("--fuse-hnrope", action="store_true",
                    help="Phase D step 1: fuse headnorm+rope into ONE kernel for "
                         "q and k (dflash-hnrope-*); saves 2 dispatches/layer")
    args = ap.parse_args()

    G = Golden(args.golden_dir)
    if args.dump_manifest:
        MANIFEST.enabled = True
    if args.dump_op:
        global DUMP_OP_DIR
        DUMP_OP_DIR = args.dump_op
    if args.dump_ref:
        global DUMP_REF_PATH
        DUMP_REF_PATH = args.dump_ref
    meta = json.load(open(G.root / "ref_meta.json"))
    cfg = Cfg(meta)
    print(f"[dflash_body_npu] B={cfg.B} L={cfg.L} tot={cfg.tot} H={cfg.H} I={cfg.I} "
          f"NH={cfg.NH} NKV={cfg.NKV} NL={cfg.NL} theta={cfg.THETA:.0e}")
    print(f"[dflash_body_npu] loading weights ...")
    W = Weights(args.weights)
    ops = NpuOps(batch_ops=args.batch_ops, proj_mode=args.proj,
                 resident_weights=args.resident_weights,
                 concat_proj=args.concat_proj, attn_all_kv=args.attn_all_kv,
                 attn_cores=args.attn_cores,
                 fuse_hnrope=args.fuse_hnrope,
                 fold_norm=args.fold_norm or args.stack_ctx,
                 stack_ctx=args.stack_ctx)
    print(f"[dflash_body_npu] config: batch_ops={ops.batch_ops} "
          f"proj_mode={ops.proj_mode} resident_weights={ops.resident_weights} "
          f"concat_proj={ops.concat_proj} attn_all_kv={ops.attn_all_kv} "
          f"attn_cores={ops.attn_cores} "
          f"fuse_hnrope={ops.fuse_hnrope} fold_norm={ops.fold_norm} "
          f"stack_ctx={ops.stack_ctx}")

    if args.dump_weights:
        dump_native_weights(W, cfg, args.dump_weights)
        return 0

    if args.op_by_op:
        ok = op_by_op(ops, W, cfg, G, args.cos_gate)
    else:
        ok = full_body(ops, W, cfg, G, args.cos_gate, warm_runs=args.warm_runs)
    if args.dump_manifest:
        MANIFEST.dump(args.dump_manifest)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
