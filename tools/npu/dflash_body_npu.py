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

from build_dflash_attention_sc import run_attn_head  # noqa: E402
from dflash_ref import load_safetensors_f32  # noqa: E402

KERNEL_NAME = "MLIR_AIE"
NPU_DIR = REPO_ROOT / "target" / "npu"
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


# ─────────────────────────────────────────────────────────────────────────────
# NPU op wrappers.  Each takes np.float32 in, returns np.float32 out.
# XRTHostRuntime handles are cached per-xclbin (one hw_context each).
# ─────────────────────────────────────────────────────────────────────────────
class NpuOps:
    def __init__(self, count=True):
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
        return self._rt.load(self._npukernel(stem))

    def release(self):
        """No-op: the shared cached runtime manages context lifetime/eviction."""
        return

    def _run(self, handle, tensors):
        res = self._rt.run(handle, tensors)
        if not res.is_success():
            raise RuntimeError(f"NPU kernel {stem} failed: {res.ret}")
        self.dispatches += 1
        return res

    # ── rmsnorm  [rows, H] weighted, over last dim ──────────────────────────
    def rmsnorm(self, x, weight):
        x = np.ascontiguousarray(x, np.float32)
        rows, H = x.shape
        h = self._load(f"qwen35-rmsnorm-{H}")
        w = XRTTensor(bf16(weight).astype(bfloat16), dtype=bfloat16, device="cpu")
        out = np.empty((rows, H), np.float32)
        for r in range(rows):
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((H,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, w, t_out])  # order: in, weight, out
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── headnorm  [rows, nh*hd] per-head rmsnorm ────────────────────────────
    def headnorm(self, x, weight, which, nh):
        x = np.ascontiguousarray(x, np.float32)
        rows, ND = x.shape
        assert ND == nh * HEAD_DIM
        h = self._load(f"qwen35-headnorm-{which}-{nh}h{HEAD_DIM}d")
        w = XRTTensor(bf16(weight).astype(bfloat16), dtype=bfloat16, device="cpu")
        out = np.empty((rows, ND), np.float32)
        for r in range(rows):
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((ND,), dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_out, w])  # order: in, out, weight
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── rope (full neox)  [rows, nh*hd] with per-row position ───────────────
    def rope(self, x, positions, which, nh, theta):
        x = np.ascontiguousarray(x, np.float32)
        rows, ND = x.shape
        assert ND == nh * HEAD_DIM
        h = self._load(f"dflash-rope-{which}-{nh}h{HEAD_DIM}d")
        out = np.empty((rows, ND), np.float32)
        for r in range(rows):
            cs = _make_cs_buf(HEAD_DIM, int(positions[r]), theta)  # bf16 [head_dim]
            t_in = XRTTensor(x[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((ND,), dtype=bfloat16, device="cpu")
            t_cs = XRTTensor(cs, dtype=bfloat16, device="cpu")
            self._run(h, [t_in, t_out, t_cs])  # order: in, out, cs
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── swiglu  silu(gate)*up  [rows, I] ────────────────────────────────────
    def swiglu(self, gate, up):
        gate = np.ascontiguousarray(gate, np.float32)
        up = np.ascontiguousarray(up, np.float32)
        rows, I = gate.shape
        h = self._load(f"qwen35-swiglu-{I}")
        out = np.empty((rows, I), np.float32)
        for r in range(rows):
            t_g = XRTTensor(gate[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_u = XRTTensor(up[r].astype(bfloat16), dtype=bfloat16, device="cpu")
            t_out = XRTTensor((I,), dtype=bfloat16, device="cpu")
            self._run(h, [t_g, t_u, t_out])  # order: gate, up, out
            t_out.to("cpu")
            out[r] = t_out.numpy().astype(bfloat16).astype(np.float32)
        self.op_dispatches += 1
        return out

    # ── int8 per-group projection  Y[rows,N] = X[rows,K] @ W[N,K]^T ─────────
    def proj(self, x, qw, sw):
        """qw int8 [N,K], sw f32 [N,ng] are the pre-quantized weight."""
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

    # ── attention  q[B,NH,HD] k/v[tot,NKV,HD] -> ctx[B, NH*HD] ──────────────
    def attention(self, q, k, v, groups):
        self.release()  # free XRT columns for the @iron.jit attention
        B, NH, HD = q.shape
        tot = k.shape[0]
        ctx = np.empty((B, NH * HD), np.float32)
        for h in range(NH):
            kvh = h // groups
            o = run_attn_head(q[:, h, :], k[:, kvh, :], v[:, kvh, :], B, tot)  # [B,HD]
            self.dispatches += 1
            ctx[:, h * HD:(h + 1) * HD] = o
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

    def raw(self, name):
        return self.W[name]

    def qproj(self, name):
        q = self._q.get(name)
        if q is None:
            qw, sw = quantize_group_symmetric(self.W[name].astype(np.float32), 8)
            q = (qw, sw)
            self._q[name] = q
        return q


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


def compute_thp(ops, W, cfg, target_hidden, use_ref=False):
    """One-time context projection: thp = hidden_norm(fc(target_hidden))."""
    th_flat = target_hidden.reshape(cfg.L, cfg.NE * cfg.H)
    qw, sw = W.qproj("fc.weight")
    if use_ref:
        thp = ref_proj_int8(th_flat, qw, sw)
        thp = ref_rmsnorm(thp, W.raw("hidden_norm.weight"), cfg.EPS)
    else:
        thp = ops.proj(th_flat, qw, sw)
        thp = ops.rmsnorm(thp, W.raw("hidden_norm.weight"))
    return thp


# ─────────────────────────────────────────────────────────────────────────────
# Full unfused body
# ─────────────────────────────────────────────────────────────────────────────
def run_body(ops, W, cfg, noise, thp, per_layer_out=None):
    hidden = noise.astype(np.float32).copy()
    for li in range(cfg.NL):
        p = f"layers.{li}."
        residual = hidden
        xn = ops.rmsnorm(hidden, W.raw(p + "input_layernorm.weight"))
        q = ops.proj(xn, *W.qproj(p + "self_attn.q_proj.weight"))
        k_noise = ops.proj(xn, *W.qproj(p + "self_attn.k_proj.weight"))
        v_noise = ops.proj(xn, *W.qproj(p + "self_attn.v_proj.weight"))
        k_ctx = ops.proj(thp, *W.qproj(p + "self_attn.k_proj.weight"))
        v_ctx = ops.proj(thp, *W.qproj(p + "self_attn.v_proj.weight"))

        q = ops.headnorm(q, W.raw(p + "self_attn.q_norm.weight"), "q", cfg.NH)
        k = np.concatenate([k_ctx, k_noise], axis=0)  # [tot, NKV*HD]
        v = np.concatenate([v_ctx, v_noise], axis=0)
        k = ops.headnorm(k, W.raw(p + "self_attn.k_norm.weight"), "k", cfg.NKV)

        q = ops.rope(q, np.arange(cfg.L, cfg.L + cfg.B), "q", cfg.NH, cfg.THETA)
        k = ops.rope(k, np.arange(0, cfg.tot), "k", cfg.NKV, cfg.THETA)

        qh = q.reshape(cfg.B, cfg.NH, cfg.HD)
        kh = k.reshape(cfg.tot, cfg.NKV, cfg.HD)
        vh = v.reshape(cfg.tot, cfg.NKV, cfg.HD)
        ctx = ops.attention(qh, kh, vh, cfg.groups)

        attn_proj = ops.proj(ctx, *W.qproj(p + "self_attn.o_proj.weight"))
        hidden = residual + attn_proj

        residual = hidden
        xn2 = ops.rmsnorm(hidden, W.raw(p + "post_attention_layernorm.weight"))
        gate = ops.proj(xn2, *W.qproj(p + "mlp.gate_proj.weight"))
        up = ops.proj(xn2, *W.qproj(p + "mlp.up_proj.weight"))
        s = ops.swiglu(gate, up)
        d = ops.proj(s, *W.qproj(p + "mlp.down_proj.weight"))
        hidden = residual + d
        if per_layer_out is not None:
            per_layer_out.append(hidden.copy())
    final = ops.rmsnorm(hidden, W.raw("norm.weight"))
    return final


def ref_body(W, cfg, noise, thp_ref, per_layer_out=None):
    """bf16/int8-precision numpy mirror of run_body (no NPU)."""
    hidden = noise.astype(np.float32).copy()
    for li in range(cfg.NL):
        p = f"layers.{li}."
        residual = hidden
        xn = ref_rmsnorm(hidden, W.raw(p + "input_layernorm.weight"), cfg.EPS)
        q = ref_proj_int8(xn, *W.qproj(p + "self_attn.q_proj.weight"))
        k_noise = ref_proj_int8(xn, *W.qproj(p + "self_attn.k_proj.weight"))
        v_noise = ref_proj_int8(xn, *W.qproj(p + "self_attn.v_proj.weight"))
        k_ctx = ref_proj_int8(thp_ref, *W.qproj(p + "self_attn.k_proj.weight"))
        v_ctx = ref_proj_int8(thp_ref, *W.qproj(p + "self_attn.v_proj.weight"))

        q = ref_headnorm(q, W.raw(p + "self_attn.q_norm.weight"), cfg.NH, cfg.EPS)
        k = np.concatenate([k_ctx, k_noise], axis=0)
        v = np.concatenate([v_ctx, v_noise], axis=0)
        k = ref_headnorm(k, W.raw(p + "self_attn.k_norm.weight"), cfg.NKV, cfg.EPS)

        q = ref_rope(q, np.arange(cfg.L, cfg.L + cfg.B), cfg.NH, cfg.THETA)
        k = ref_rope(k, np.arange(0, cfg.tot), cfg.NKV, cfg.THETA)

        qh = q.reshape(cfg.B, cfg.NH, cfg.HD)
        kh = k.reshape(cfg.tot, cfg.NKV, cfg.HD)
        vh = v.reshape(cfg.tot, cfg.NKV, cfg.HD)
        ctx = ref_attention(qh, kh, vh, cfg.groups)

        attn_proj = ref_proj_int8(ctx, *W.qproj(p + "self_attn.o_proj.weight"))
        hidden = residual + attn_proj

        residual = hidden
        xn2 = ref_rmsnorm(hidden, W.raw(p + "post_attention_layernorm.weight"), cfg.EPS)
        gate = ref_proj_int8(xn2, *W.qproj(p + "mlp.gate_proj.weight"))
        up = ref_proj_int8(xn2, *W.qproj(p + "mlp.up_proj.weight"))
        s = ref_swiglu(gate, up)
        d = ref_proj_int8(s, *W.qproj(p + "mlp.down_proj.weight"))
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

    # thp (one-time) vs golden target_hidden_proj
    thp = compute_thp(ops, W, cfg, target_hidden)
    check("thp (fc+hidden_norm)", thp, G.rust("target_hidden_proj"))

    # 1. input_layernorm : noise -> l0_input_norm
    xn = ops.rmsnorm(noise, W.raw(p + "input_layernorm.weight"))
    xn = check("input_norm", xn, G.rust("l0_input_norm"))

    # 2. projections from GOLDEN input_norm + golden thp, headnorm, rope
    #    checkpoints available: q_roped, k_roped, v.
    xn_g = G.rust("l0_input_norm")          # feed golden to isolate downstream
    thp_g = G.rust("target_hidden_proj")
    q = ops.proj(xn_g, *W.qproj(p + "self_attn.q_proj.weight"))
    k_noise = ops.proj(xn_g, *W.qproj(p + "self_attn.k_proj.weight"))
    v_noise = ops.proj(xn_g, *W.qproj(p + "self_attn.v_proj.weight"))
    k_ctx = ops.proj(thp_g, *W.qproj(p + "self_attn.k_proj.weight"))
    v_ctx = ops.proj(thp_g, *W.qproj(p + "self_attn.v_proj.weight"))
    q = ops.headnorm(q, W.raw(p + "self_attn.q_norm.weight"), "q", cfg.NH)
    k = np.concatenate([k_ctx, k_noise], axis=0)
    v = np.concatenate([v_ctx, v_noise], axis=0)
    k = ops.headnorm(k, W.raw(p + "self_attn.k_norm.weight"), "k", cfg.NKV)
    q = ops.rope(q, np.arange(cfg.L, cfg.L + cfg.B), "q", cfg.NH, cfg.THETA)
    k = ops.rope(k, np.arange(0, cfg.tot), "k", cfg.NKV, cfg.THETA)
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
    attn_proj = ops.proj(ctx_g, *W.qproj(p + "self_attn.o_proj.weight"))
    post = noise + attn_proj
    check("post_attn_residual", post, G.rust("l0_post_attn_residual"))

    # 5. MLP from GOLDEN post_attn_residual -> l0_out
    post_g = G.rust("l0_post_attn_residual")
    xn2 = ops.rmsnorm(post_g, W.raw(p + "post_attention_layernorm.weight"))
    gate = ops.proj(xn2, *W.qproj(p + "mlp.gate_proj.weight"))
    up = ops.proj(xn2, *W.qproj(p + "mlp.up_proj.weight"))
    s = ops.swiglu(gate, up)
    d = ops.proj(s, *W.qproj(p + "mlp.down_proj.weight"))
    l0_out = post_g + d
    check("l0_out (MLP)", l0_out, G.rust("l0_out"))

    n_ok = sum(1 for _, ok in results if ok)
    print(f"--- op-by-op: {n_ok}/{len(results)} PASS  "
          f"(raw dispatches={ops.dispatches}) ---")
    return all(ok for _, ok in results)


# ─────────────────────────────────────────────────────────────────────────────
# Full unfused body validation (Gate D step 1)
# ─────────────────────────────────────────────────────────────────────────────
def full_body(ops, W, cfg, G, cos_gate=0.99):
    print("=== full unfused body ===")
    noise = G.inp("noise_embedding").astype(np.float32)
    target_hidden = G.inp("target_hidden").astype(np.float32)

    t0 = time.perf_counter()
    thp = compute_thp(ops, W, cfg, target_hidden)
    per_layer = []
    final = run_body(ops, W, cfg, noise, thp, per_layer_out=per_layer)
    wall = time.perf_counter() - t0

    # bf16/int8-precision numpy reference
    thp_ref = compute_thp(None, W, cfg, target_hidden, use_ref=True)
    per_layer_ref = []
    final_ref = ref_body(W, cfg, noise, thp_ref, per_layer_out=per_layer_ref)

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
    print(f"  raw NPU dispatches = {ops.dispatches}  |  logical op dispatches = {ops.op_dispatches}")
    print(f"  wall = {wall:.1f} s")

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
    args = ap.parse_args()

    G = Golden(args.golden_dir)
    import json
    meta = json.load(open(G.root / "ref_meta.json"))
    cfg = Cfg(meta)
    print(f"[dflash_body_npu] B={cfg.B} L={cfg.L} tot={cfg.tot} H={cfg.H} I={cfg.I} "
          f"NH={cfg.NH} NKV={cfg.NKV} NL={cfg.NL} theta={cfg.THETA:.0e}")
    print(f"[dflash_body_npu] loading weights ...")
    W = Weights(args.weights)
    ops = NpuOps()

    if args.op_by_op:
        ok = op_by_op(ops, W, cfg, G, args.cos_gate)
    else:
        ok = full_body(ops, W, cfg, G, args.cos_gate)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
