// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dispatch wrappers for the ZAYA1 CCA + EDA/MoD custom kernels
//! (`kernels/src/zaya_cca.hip`). All are register-only (no LDS). Faithful to
//! `crates/hipfire-arch-zaya/src/cpu.rs`.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const BLOCK: u32 = 256;

impl Gpu {
    fn zaya_ensure(&mut self, func: &str) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("zaya_cca", kernels::ZAYA_CCA_SRC, func)
    }

    fn zaya_launch(
        &self,
        func: &str,
        threads: usize,
        params: &mut Vec<*mut c_void>,
    ) -> HipResult<()> {
        let grid = (threads as u32).div_ceil(BLOCK);
        let f = &self.functions[func];
        unsafe {
            self.hip
                .launch_kernel(f, [grid, 1, 1], [BLOCK, 1, 1], 0, self.stream_ref(), params)
        }
    }

    /// Embedding gather: `out[t,i] = embed[ids[t]*hidden + i]`. `ids` is an i32
    /// device buffer of length `s`; `n = s*hidden`.
    pub fn zaya_embed_gather_f32(
        &mut self,
        out: &GpuTensor,
        embed: &GpuTensor,
        ids: &GpuTensor,
        hidden: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_embed_gather_f32")?;
        let (op, ep, ip) = (out.buf.as_ptr(), embed.buf.as_ptr(), ids.buf.as_ptr());
        let (hi, ni) = (hidden as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_embed_gather_f32", n, &mut p)
    }

    /// Broadcast bias add: `x[i] += bias[i % d]`, `n = s*d`.
    pub fn zaya_bias_add_f32(
        &mut self,
        x: &GpuTensor,
        bias: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_bias_add_f32")?;
        let (xp, bp) = (x.buf.as_ptr(), bias.buf.as_ptr());
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_bias_add_f32", n, &mut p)
    }

    /// Global input residual affine on the embeddings:
    /// `out[i] = (x[i]+bias[c])*scale[c]`, `c = i % d`, `n = seq*d`.
    pub fn zaya_affine_input_f32(
        &mut self,
        out: &GpuTensor,
        x: &GpuTensor,
        scale: &GpuTensor,
        bias: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_affine_input_f32")?;
        let (op, xp, sp, bp) = (
            out.buf.as_ptr(),
            x.buf.as_ptr(),
            scale.buf.as_ptr(),
            bias.buf.as_ptr(),
        );
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_affine_input_f32", n, &mut p)
    }

    /// Learned residual merge:
    /// `out[i] = (h[i]+hb[c])*hs[c] + (res[i]+rb[c])*rs[c]`, `c = i % d`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_affine_residual_f32(
        &mut self,
        out: &GpuTensor,
        h: &GpuTensor,
        res: &GpuTensor,
        hs: &GpuTensor,
        hb: &GpuTensor,
        rs: &GpuTensor,
        rb: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_affine_residual_f32")?;
        let (op, hp, rp) = (out.buf.as_ptr(), h.buf.as_ptr(), res.buf.as_ptr());
        let (hsp, hbp, rsp, rbp) = (
            hs.buf.as_ptr(),
            hb.buf.as_ptr(),
            rs.buf.as_ptr(),
            rb.buf.as_ptr(),
        );
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hsp as *const _ as *mut c_void,
            &hbp as *const _ as *mut c_void,
            &rsp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_affine_residual_f32", n, &mut p)
    }

    /// Scatter q `[s,q_dim]` and k `[s,k_dim]` into a left-padded channel-major
    /// stream `[q_dim+k_dim, pad+s]` (pad region pre-zeroed by the caller).
    pub fn zaya_qk_stream_f32(
        &mut self,
        stream: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        s: usize,
        q_dim: usize,
        k_dim: usize,
        pad: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_stream_f32")?;
        let (stp, qp, kp) = (stream.buf.as_ptr(), q.buf.as_ptr(), k.buf.as_ptr());
        let (si, qd, kd, pd) = (s as i32, q_dim as i32, k_dim as i32, pad as i32);
        let mut p: Vec<*mut c_void> = vec![
            &stp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &pd as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_qk_stream_f32", (q_dim + k_dim) * s, &mut p)
    }

    /// VALID causal grouped conv1d over channel-major input `[channels, in_len]`
    /// → `[channels, out_len]`, `out_len = in_len - kernel + 1`. weight layout
    /// `[channels, channels/groups, kernel]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_conv1d_valid_f32(
        &mut self,
        out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        channels: usize,
        groups: usize,
        kernel: usize,
        in_len: usize,
        out_len: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_conv1d_valid_f32")?;
        let (op, ip, wp, bp) = (
            out.buf.as_ptr(),
            input.buf.as_ptr(),
            weight.buf.as_ptr(),
            bias.buf.as_ptr(),
        );
        let (ch, gr, kn, il, ol) = (
            channels as i32,
            groups as i32,
            kernel as i32,
            in_len as i32,
            out_len as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ch as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &kn as *const _ as *mut c_void,
            &il as *const _ as *mut c_void,
            &ol as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_conv1d_valid_f32", channels * out_len, &mut p)
    }

    /// q/k residual paths. `mode = 0` writes `query_res [s,nq,hd]`; `mode = 1`
    /// writes `key_res [s,nkv,hd]` (mean over groups of query_res).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_qk_residual_f32(
        &mut self,
        query_res: &GpuTensor,
        key_res: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        mode: i32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_residual_f32")?;
        let (qrp, krp, qp, kp) = (
            query_res.buf.as_ptr(),
            key_res.buf.as_ptr(),
            q.buf.as_ptr(),
            k.buf.as_ptr(),
        );
        let (si, nqi, nkvi, hdi) = (s as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &mode as *const _ as *mut c_void,
        ];
        let threads = if mode == 0 { s * nq * hd } else { s * nkv * hd };
        self.zaya_launch("zaya_qk_residual_f32", threads, &mut p)
    }

    /// Add conv output (channel-major `[conv_ch, s]`) to a residual path, emit
    /// head-major `[s, heads, hd]`. `mode = 0` query (conv ch = head*hd+d),
    /// `mode = 1` key (conv ch = q_dim + head*hd+d).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_add_conv_residual_f32(
        &mut self,
        out: &GpuTensor,
        conv: &GpuTensor,
        res: &GpuTensor,
        s: usize,
        heads: usize,
        hd: usize,
        q_dim: usize,
        mode: i32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_add_conv_residual_f32")?;
        let (op, cp, rp) = (out.buf.as_ptr(), conv.buf.as_ptr(), res.buf.as_ptr());
        let (si, hi, hdi, qd) = (s as i32, heads as i32, hd as i32, q_dim as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &mode as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_add_conv_residual_f32", s * heads * hd, &mut p)
    }

    /// Compositional value: head 0 = current `v_cur[s,hd]`, head 1 = previous
    /// token's `v_del` (0 at t==0). Writes `value [s, nkv, hd]`.
    pub fn zaya_value_compose_f32(
        &mut self,
        value: &GpuTensor,
        v_cur: &GpuTensor,
        v_del: &GpuTensor,
        s: usize,
        nkv: usize,
        hd: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_value_compose_f32")?;
        let (vp, cp, dp) = (value.buf.as_ptr(), v_cur.buf.as_ptr(), v_del.buf.as_ptr());
        let (si, nkvi, hdi) = (s as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &vp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_value_compose_f32", s * hd, &mut p)
    }

    /// Per `(token, head)` L2-normalize rows to `scale` (= sqrt(head_dim)); if
    /// `temp` is `Some`, multiply each head by `temp[head]` (key path).
    pub fn zaya_qk_l2norm_temp_f32(
        &mut self,
        x: &GpuTensor,
        temp: Option<&GpuTensor>,
        s: usize,
        heads: usize,
        hd: usize,
        scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_l2norm_temp_f32")?;
        let xp = x.buf.as_ptr();
        // Dummy pointer (x) when no temp; `has_temp` gates the read.
        let tp = temp.map_or(xp, |t| t.buf.as_ptr());
        let has_temp: i32 = temp.is_some() as i32;
        let (si, hi, hdi) = (s as i32, heads as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &has_temp as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_qk_l2norm_temp_f32", s * heads, &mut p)
    }

    /// Exact erf-GELU in place over `n` elements.
    pub fn zaya_gelu_exact_f32(&mut self, x: &GpuTensor, n: usize) -> HipResult<()> {
        self.zaya_ensure("zaya_gelu_exact_f32")?;
        let xp = x.buf.as_ptr();
        let ni = n as i32;
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gelu_exact_f32", n, &mut p)
    }

    /// Partial rotary (half-split) at position = `pos_base + token index`, over
    /// `x [s,heads,hd]`. `pos_base=0` for prefill; current pos for decode.
    pub fn zaya_rope_partial_f32(
        &mut self,
        x: &GpuTensor,
        s: usize,
        heads: usize,
        hd: usize,
        n_rot: usize,
        theta: f32,
        pos_base: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_rope_partial_f32")?;
        let xp = x.buf.as_ptr();
        let (si, hi, hdi, nr, pb) = (
            s as i32,
            heads as i32,
            hd as i32,
            n_rot as i32,
            pos_base as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &theta as *const _ as *mut c_void,
            &pb as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_rope_partial_f32", s * heads, &mut p)
    }

    // ── Fused decode-glue kernels (one launch replaces a pair/triple) ──

    /// Fused router MLP (single workgroup): down_proj(Oq8) → prep → rmsnorm →
    /// FWHT→fc1(Oq8)→gelu → FWHT→fc2(Oq8)→gelu → FWHT→out(Oq8) → select. One launch
    /// replaces ~9. All weight tensors are planar Oq8 `[int8 M*K | f32 scales]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_router_mlp_fused(
        &mut self,
        pa_xrot: &GpuTensor,
        dp_w: &GpuTensor,
        dp_b: &GpuTensor,
        rstate: &GpuTensor,
        escale: Option<&GpuTensor>,
        rnorm_w: &GpuTensor,
        fc1_w: &GpuTensor,
        fc1_b: &GpuTensor,
        fc2_w: &GpuTensor,
        fc2_b: &GpuTensor,
        out_w: &GpuTensor,
        bbias: &GpuTensor,
        sel_idx: &GpuTensor,
        sel_gate: &GpuTensor,
        h: usize,
        rh: usize,
        n_route: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_router_mlp_fused")?;
        self.ensure_mq_signs()?;
        let s1 = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let has_eda: i32 = escale.is_some() as i32;
        let esc = escale.map_or(dp_b.buf.as_ptr(), |e| e.buf.as_ptr());
        let (pxr, dpw, dpb) = (pa_xrot.buf.as_ptr(), dp_w.buf.as_ptr(), dp_b.buf.as_ptr());
        let (rst, rnw) = (rstate.buf.as_ptr(), rnorm_w.buf.as_ptr());
        let (f1w, f1b, f2w, f2b, ow) = (
            fc1_w.buf.as_ptr(),
            fc1_b.buf.as_ptr(),
            fc2_w.buf.as_ptr(),
            fc2_b.buf.as_ptr(),
            out_w.buf.as_ptr(),
        );
        let (bb, si, sg) = (
            bbias.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            sel_gate.buf.as_ptr(),
        );
        let (hi, rhi, nri) = (h as i32, rh as i32, n_route as i32);
        let mut p: Vec<*mut c_void> = vec![
            &pxr as *const _ as *mut c_void,
            &dpw as *const _ as *mut c_void,
            &dpb as *const _ as *mut c_void,
            &rst as *const _ as *mut c_void,
            &esc as *const _ as *mut c_void,
            &rnw as *const _ as *mut c_void,
            &f1w as *const _ as *mut c_void,
            &f1b as *const _ as *mut c_void,
            &f2w as *const _ as *mut c_void,
            &f2b as *const _ as *mut c_void,
            &ow as *const _ as *mut c_void,
            &bb as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &sg as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &rhi as *const _ as *mut c_void,
            &nri as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &has_eda as *const _ as *mut c_void,
        ];
        let f = &self.functions["zaya_router_mlp_fused"];
        unsafe {
            self.hip
                .launch_kernel(f, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut p)
        }
    }

    /// Fused pre-conv prep (single workgroup): qk_residual (both modes) + qk-stream
    /// column in one launch. Replaces 2× qk_residual + qk_stream.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_qk_prep_decode_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        query_res: &GpuTensor,
        key_res: &GpuTensor,
        cur_qk: &GpuTensor,
        nq: usize,
        nkv: usize,
        hd: usize,
        q_dim: usize,
        k_dim: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_prep_decode_f32")?;
        let (qp, kp, qrp, krp, cp) = (
            q.buf.as_ptr(),
            k.buf.as_ptr(),
            query_res.buf.as_ptr(),
            key_res.buf.as_ptr(),
            cur_qk.buf.as_ptr(),
        );
        let (nqi, nkvi, hdi, qdi, kdi) =
            (nq as i32, nkv as i32, hd as i32, q_dim as i32, k_dim as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &qdi as *const _ as *mut c_void,
            &kdi as *const _ as *mut c_void,
        ];
        // Single workgroup (needs __syncthreads between query_res and key_res).
        let f = &self.functions["zaya_qk_prep_decode_f32"];
        unsafe {
            self.hip
                .launch_kernel(f, [1, 1, 1], [BLOCK, 1, 1], 0, self.stream_ref(), &mut p)
        }
    }

    /// Fused broadcast bias-add + exact GELU in place: `x[i] = gelu(x[i]+bias[i%d])`.
    pub fn zaya_bias_gelu_f32(
        &mut self,
        x: &GpuTensor,
        bias: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_bias_gelu_f32")?;
        let (xp, bp) = (x.buf.as_ptr(), bias.buf.as_ptr());
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_bias_gelu_f32", n, &mut p)
    }

    /// Fused partial-RoPE over query `[s,nq,hd]` AND key `[s,nkv,hd]` in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_rope_partial_qk_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        n_rot: usize,
        theta: f32,
        pos_base: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_rope_partial_qk_f32")?;
        let (qp, kp) = (q.buf.as_ptr(), k.buf.as_ptr());
        let (si, nqi, nkvi, hdi, nr, pb) = (
            s as i32,
            nq as i32,
            nkv as i32,
            hd as i32,
            n_rot as i32,
            pos_base as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &theta as *const _ as *mut c_void,
            &pb as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_rope_partial_qk_f32", s * (nq + nkv), &mut p)
    }

    /// Device-position variant of [`zaya_rope_partial_qk_f32`]: base position is
    /// read from `pos_buf[0]` (device i32), so the launch is capture-safe.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_rope_partial_qk_posbuf_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &hip_bridge::DeviceBuffer,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        n_rot: usize,
        theta: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_rope_partial_qk_posbuf_f32")?;
        let (qp, kp, pp) = (q.buf.as_ptr(), k.buf.as_ptr(), pos_buf.as_ptr());
        let (si, nqi, nkvi, hdi, nr) = (s as i32, nq as i32, nkv as i32, hd as i32, n_rot as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &theta as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_rope_partial_qk_posbuf_f32", s * (nq + nkv), &mut p)
    }

    /// Fused L2-norm+scale over query (no temp) AND key (per-head `temp`) in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_qk_l2norm_qk_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        temp: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_l2norm_qk_f32")?;
        let (qp, kp, tp) = (q.buf.as_ptr(), k.buf.as_ptr(), temp.buf.as_ptr());
        let (si, nqi, nkvi, hdi) = (s as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_qk_l2norm_qk_f32", s * (nq + nkv), &mut p)
    }

    /// Gather whole rows into a contiguous run: `out[r,c] = src[idx[r],c]`.
    ///
    /// `idx` is an i32 device buffer of `rows` entries; `n = rows * width`.
    /// Indices are rows within a batch, not model-scale ids, so the int32
    /// arithmetic in the kernel has no realistic overflow bound.
    pub fn zaya_gather_rows_f32(
        &mut self,
        out: &GpuTensor,
        src: &GpuTensor,
        idx: &hip_bridge::DeviceBuffer,
        width: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_gather_rows_f32")?;
        let (op, sp, ip) = (out.buf.as_ptr(), src.buf.as_ptr(), idx.as_ptr());
        let (wi, ni) = (width as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gather_rows_f32", n, &mut p)
    }

    /// Batched SwiGLU over an interleaved `[rows, 2*inter]` gate_up result:
    /// `out[r,c] = silu(gate_up[r,c]) * gate_up[r,inter+c]`, `n = rows * inter`.
    ///
    /// The flat `silu_mul_f32` cannot express this because the gate and up
    /// halves are interleaved per row; this takes the row stride explicitly and
    /// uses the same `expf` formula, so one call is bit-identical to `rows`
    /// per-row calls.
    pub fn zaya_silu_mul_gate_up_f32(
        &mut self,
        out: &GpuTensor,
        gate_up: &GpuTensor,
        inter: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_silu_mul_gate_up_f32")?;
        let (op, gp) = (out.buf.as_ptr(), gate_up.buf.as_ptr());
        let (ii, ni) = (inter as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &ii as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_silu_mul_gate_up_f32", n, &mut p)
    }

    /// Scatter-accumulate whole rows with a per-row scale:
    /// `dst[idx[r],c] += scale[r] * src[r,c]`, `n = rows * width`.
    ///
    /// Callers must guarantee each destination row is named by at most one
    /// source row (true under top-1 routing) — there are no atomics.
    pub fn zaya_scatter_scaled_add_f32(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        idx: &hip_bridge::DeviceBuffer,
        scale: &GpuTensor,
        width: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_scatter_scaled_add_f32")?;
        let (dp, sp, ip, cp) = (
            dst.buf.as_ptr(),
            src.buf.as_ptr(),
            idx.as_ptr(),
            scale.buf.as_ptr(),
        );
        let (wi, ni) = (width as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &wi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_scatter_scaled_add_f32", n, &mut p)
    }

    /// Fused add-conv-residual over query AND key in one launch (both modes).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_add_conv_residual_qk_f32(
        &mut self,
        query: &GpuTensor,
        key: &GpuTensor,
        conv: &GpuTensor,
        qres: &GpuTensor,
        kres: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        q_dim: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_add_conv_residual_qk_f32")?;
        let (qp, kp, cp) = (query.buf.as_ptr(), key.buf.as_ptr(), conv.buf.as_ptr());
        let (qrp, krp) = (qres.buf.as_ptr(), kres.buf.as_ptr());
        let (si, nqi, nkvi, hdi, qdi) = (s as i32, nq as i32, nkv as i32, hd as i32, q_dim as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &qdi as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_add_conv_residual_qk_f32", s * (nq + nkv) * hd, &mut p)
    }

    /// Fused router-input prep: `rhid += bias`; if `scale` is `Some`, `rhid +=
    /// prev*scale` (EDA); then `prev := rhid`. Replaces bias_add + eda_add + copy.
    pub fn zaya_router_prep_f32(
        &mut self,
        rhid: &GpuTensor,
        bias: &GpuTensor,
        prev: &GpuTensor,
        scale: Option<&GpuTensor>,
        rh: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_router_prep_f32")?;
        let (rp, bp, pp) = (rhid.buf.as_ptr(), bias.buf.as_ptr(), prev.buf.as_ptr());
        // Dummy pointer (bias) when no EDA; `has_eda` gates the read.
        let sp = scale.map_or(bp, |s| s.buf.as_ptr());
        let has_eda: i32 = scale.is_some() as i32;
        let (rhi, ni) = (rh as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &rhi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
            &has_eda as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_router_prep_f32", n, &mut p)
    }

    /// Fused single-token value assembly + delayed-value advance (decode).
    pub fn zaya_value_assemble_decode_f32(
        &mut self,
        value: &GpuTensor,
        v_cur: &GpuTensor,
        delayed_v: &GpuTensor,
        v_del: &GpuTensor,
        vh: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_value_assemble_decode_f32")?;
        let (vp, cp, dp, xp) = (
            value.buf.as_ptr(),
            v_cur.buf.as_ptr(),
            delayed_v.buf.as_ptr(),
            v_del.buf.as_ptr(),
        );
        let vhi = vh as i32;
        let mut p: Vec<*mut c_void> = vec![
            &vp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &vhi as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_value_assemble_decode_f32", vh, &mut p)
    }

    /// Strided row copy: `dst[r*dst_stride + j] = src[r*src_stride + src_off + j]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_strided_copy_f32(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        rows: usize,
        dst_stride: usize,
        src_stride: usize,
        src_off: usize,
        len: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_strided_copy_f32")?;
        let (dp, sp) = (dst.buf.as_ptr(), src.buf.as_ptr());
        let (r, ds, ss, so, l) = (
            rows as i32,
            dst_stride as i32,
            src_stride as i32,
            src_off as i32,
            len as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &r as *const _ as *mut c_void,
            &ds as *const _ as *mut c_void,
            &ss as *const _ as *mut c_void,
            &so as *const _ as *mut c_void,
            &l as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_strided_copy_f32", rows * len, &mut p)
    }

    /// Decode conv window build + ring advance: `window [conv_ch, pad+1]` = `[ring | cur]`.
    pub fn zaya_conv_window_f32(
        &mut self,
        window: &GpuTensor,
        ring: &GpuTensor,
        cur: &GpuTensor,
        conv_ch: usize,
        pad: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_conv_window_f32")?;
        let (wp, rp, cp) = (window.buf.as_ptr(), ring.buf.as_ptr(), cur.buf.as_ptr());
        let (cc, pd) = (conv_ch as i32, pad as i32);
        let mut p: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &cc as *const _ as *mut c_void,
            &pd as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_conv_window_f32", conv_ch, &mut p)
    }

    /// Copy `src[0..n]` into `dst[offset..offset+n]` (KV-cache append).
    pub fn zaya_write_at_f32(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        offset: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_write_at_f32")?;
        let (dp, sp) = (dst.buf.as_ptr(), src.buf.as_ptr());
        let (off, ni) = (offset as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &off as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_write_at_f32", n, &mut p)
    }

    /// Single-token GQA decode attention over the KV cache (positions 0..=pos).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_gqa_decode_f32(
        &mut self,
        out: &GpuTensor,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        pos: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        scaling: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_gqa_decode_f32")?;
        let (op, qp, kp, vp) = (
            out.buf.as_ptr(),
            q.buf.as_ptr(),
            k_cache.buf.as_ptr(),
            v_cache.buf.as_ptr(),
        );
        let (posi, nqi, nkvi, hdi) = (pos as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &posi as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scaling as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gqa_decode_f32", nq, &mut p)
    }

    /// Simple GQA causal attention (bring-up). q `[s,nq,hd]`, k/v `[s,nkv,hd]`
    /// head-major; out `[s,nq,hd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_gqa_attn_f32(
        &mut self,
        out: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        scaling: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_gqa_attn_f32")?;
        let (op, qp, kp, vp) = (
            out.buf.as_ptr(),
            q.buf.as_ptr(),
            k.buf.as_ptr(),
            v.buf.as_ptr(),
        );
        let (si, nqi, nkvi, hdi) = (s as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scaling as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gqa_attn_f32", s * nq, &mut p)
    }

    /// EDA cross-layer state add: `router_hidden[i] += prev[i] * scale[i%rh]`.
    pub fn zaya_eda_add_f32(
        &mut self,
        router_hidden: &GpuTensor,
        prev: &GpuTensor,
        scale: &GpuTensor,
        rh: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_eda_add_f32")?;
        let (rp, pp, sp) = (
            router_hidden.buf.as_ptr(),
            prev.buf.as_ptr(),
            scale.buf.as_ptr(),
        );
        let (rhi, ni) = (rh as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &rhi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_eda_add_f32", n, &mut p)
    }

    /// On-device top-1 router select: softmax over `n_route` logits, argmax over
    /// (prob + balancing bias), writing the winning expert id to `sel_idx` (i32)
    /// and its unbiased softmax gate to `sel_gate` (0 for the null slot). Removes
    /// the per-block `download_f32` host readback from `gpu_decode`. Single-CTA.
    pub fn zaya_router_select_f32(
        &mut self,
        logits: &GpuTensor,
        bias: &GpuTensor,
        sel_idx: &GpuTensor,
        sel_gate: &GpuTensor,
        n_route: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_router_select_f32")?;
        let (lp, bp, sip, sgp) = (
            logits.buf.as_ptr(),
            bias.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            sel_gate.buf.as_ptr(),
        );
        let nr = n_route as i32;
        let mut p: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &sgp as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
        ];
        // threads=1 → grid 1, block 256, only tid 0 executes (n_route is tiny).
        self.zaya_launch("zaya_router_select_f32", 1, &mut p)
    }

    /// Launch a per-output-row wave32 kernel: grid = `rows`, block = 32 lanes.
    fn zaya_launch_rows(
        &self,
        func: &str,
        rows: usize,
        params: &mut Vec<*mut c_void>,
    ) -> HipResult<()> {
        let f = &self.functions[func];
        unsafe {
            self.hip.launch_kernel(
                f,
                [rows as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                params,
            )
        }
    }

    /// Indexed OQ8 top-1 fused gate_up GEMV over zaya's planar `oq8_combined`
    /// expert buffers. `y_gate`/`y_up` are `[M/2]` each; `M = 2*moe_int`, `K = hidden`.
    pub fn zaya_moe_gate_up_oq8_planar_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        sel_idx: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_moe_gate_up_oq8_planar_indexed")?;
        let (pp, sip, xp, ygp, yup) = (
            expert_ptrs.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            x.buf.as_ptr(),
            y_gate.buf.as_ptr(),
            y_up.buf.as_ptr(),
        );
        let (mi, ki) = (m as i32, k as i32);
        let mut p: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
        ];
        // LDS activation-staged variant (HIPFIRE_ZAYA_MOE_LDS): a block of 8 waves stages
        // x[K] in LDS once, removing the re-read activation loads from the memory unit so
        // the latency-bound weight fetches get its full request budget. See zaya_cca.hip.
        if std::env::var("HIPFIRE_ZAYA_MOE_LDS").is_ok() {
            self.zaya_ensure("zaya_moe_gate_up_oq8_planar_indexed_lds")?;
            let grid = m.div_ceil(8) as u32;
            let lds = (k * 4) as u32;
            let f = &self.functions["zaya_moe_gate_up_oq8_planar_indexed_lds"];
            return unsafe {
                self.hip
                    .launch_kernel(f, [grid, 1, 1], [256, 1, 1], lds, self.stream_ref(), &mut p)
            };
        }
        // Multi-row register-blocked variant (HIPFIRE_ZAYA_MOE_MROW): 4 rows/wave
        // for ~4× memory-level parallelism (the one-row kernel is latency-bound).
        if std::env::var("HIPFIRE_ZAYA_MOE_MROW").is_ok() {
            self.zaya_ensure("zaya_moe_gate_up_oq8_planar_indexed_mrow")?;
            let grid = m.div_ceil(4) as u32;
            let f = &self.functions["zaya_moe_gate_up_oq8_planar_indexed_mrow"];
            return unsafe {
                self.hip
                    .launch_kernel(f, [grid, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut p)
            };
        }
        self.zaya_launch_rows("zaya_moe_gate_up_oq8_planar_indexed", m, &mut p)
    }

    /// Indexed OQ8 top-1 down GEMV over zaya's planar expert buffers. `y` is
    /// `[M = hidden]`, `K = moe_int`.
    pub fn zaya_moe_down_oq8_planar_indexed(
        &mut self,
        expert_ptrs: &GpuTensor,
        sel_idx: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_moe_down_oq8_planar_indexed")?;
        let (pp, sip, xp, yp) = (
            expert_ptrs.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            x.buf.as_ptr(),
            y.buf.as_ptr(),
        );
        let (mi, ki) = (m as i32, k as i32);
        let mut p: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
        ];
        if std::env::var("HIPFIRE_ZAYA_MOE_LDS").is_ok() {
            self.zaya_ensure("zaya_moe_down_oq8_planar_indexed_lds")?;
            let grid = m.div_ceil(8) as u32;
            let lds = (k * 4) as u32;
            let f = &self.functions["zaya_moe_down_oq8_planar_indexed_lds"];
            return unsafe {
                self.hip
                    .launch_kernel(f, [grid, 1, 1], [256, 1, 1], lds, self.stream_ref(), &mut p)
            };
        }
        if std::env::var("HIPFIRE_ZAYA_MOE_MROW").is_ok() {
            self.zaya_ensure("zaya_moe_down_oq8_planar_indexed_mrow")?;
            let grid = m.div_ceil(4) as u32;
            let f = &self.functions["zaya_moe_down_oq8_planar_indexed_mrow"];
            return unsafe {
                self.hip
                    .launch_kernel(f, [grid, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut p)
            };
        }
        self.zaya_launch_rows("zaya_moe_down_oq8_planar_indexed", m, &mut p)
    }

    /// W8A8 int8-activation indexed gate_up: reads an int8-quantized activation `xq`
    /// (+ per-256-group scales `xs`) instead of f32 — 4× less activation load traffic,
    /// signed V_DOT4 int8 dot. See zaya_cca.hip.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_moe_gate_up_oq8_planar_indexed_w8a8(
        &mut self,
        expert_ptrs: &GpuTensor,
        sel_idx: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_moe_gate_up_oq8_planar_indexed_w8a8")?;
        let (pp, sip, xqp, xsp, ygp, yup) = (
            expert_ptrs.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            xq.buf.as_ptr(),
            xs.buf.as_ptr(),
            y_gate.buf.as_ptr(),
            y_up.buf.as_ptr(),
        );
        let (mi, ki) = (m as i32, k as i32);
        let mut p: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
        ];
        self.zaya_launch_rows("zaya_moe_gate_up_oq8_planar_indexed_w8a8", m, &mut p)
    }

    /// W8A8 int8-activation indexed down projection (see the gate_up variant).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_moe_down_oq8_planar_indexed_w8a8(
        &mut self,
        expert_ptrs: &GpuTensor,
        sel_idx: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_moe_down_oq8_planar_indexed_w8a8")?;
        let (pp, sip, xqp, xsp, yp) = (
            expert_ptrs.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            xq.buf.as_ptr(),
            xs.buf.as_ptr(),
            y.buf.as_ptr(),
        );
        let (mi, ki) = (m as i32, k as i32);
        let mut p: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mi as *const _ as *mut c_void,
            &ki as *const _ as *mut c_void,
        ];
        self.zaya_launch_rows("zaya_moe_down_oq8_planar_indexed_w8a8", m, &mut p)
    }

    /// Micro-benchmark the cost of one `grid.sync()` cooperative barrier at the
    /// megakernel launch shape (`grid_blocks` × 256). Returns µs/sync. Diagnostic
    /// for whether grid.sync overhead dominates the megakernel (EXP-19).
    pub fn zaya_bench_grid_sync(&mut self, grid_blocks: u32, n_syncs: i32) -> HipResult<f64> {
        self.bind_thread()?;
        self.ensure_kernel(
            "zaya_megakernel",
            kernels::ZAYA_MEGAKERNEL_SRC,
            "zaya_grid_sync_bench",
        )?;
        let scratch = self.hip.malloc(4)?;
        let sp = scratch.as_ptr();
        let ns = n_syncs;
        let launch = |g: &Self| -> HipResult<()> {
            let mut p: Vec<*mut c_void> = vec![
                &sp as *const _ as *mut c_void,
                &ns as *const _ as *mut c_void,
            ];
            let f = &g.functions["zaya_grid_sync_bench"];
            unsafe {
                g.hip.launch_cooperative_kernel(
                    f,
                    [grid_blocks, 1, 1],
                    [256, 1, 1],
                    0,
                    g.stream_ref(),
                    &mut p,
                )
            }
        };
        // Warmup, then time `iters` launches.
        launch(self)?;
        self.hip.device_synchronize()?;
        let iters = 5;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(self)?;
        }
        self.hip.device_synchronize()?;
        let us = t.elapsed().as_micros() as f64;
        let _ = self.hip.free(scratch);
        Ok(us / (iters as f64 * n_syncs as f64))
    }

    /// ZAYA decode cooperative megakernel — Phase 0 (megakernel-B). Fuses the
    /// MLP half of one decode block (post-attn rmsnorm+rotate → router MLP+select
    /// → MoE gate_up → silu_mul+rotate → MoE down + affine residual) into ONE
    /// cooperative launch, grid-strided over the device's resident workgroups
    /// with `grid.sync()` between phases. Faithful fusion of the reference op
    /// chain; env-gated (`HIPFIRE_ZAYA_MEGAKERNEL`) and diffed against it.
    ///
    /// `pm_rs` is the 4-element affine-residual scale set `[hs, hb, rs, rb]`.
    /// `escale` is the EDA cross-layer scale (`None` on layer 0 → `has_eda=0`).
    /// `mk_norm` is `[h]` scratch (rmsnormed pre-FWHT); `pa_xrot` `[h]`,
    /// `gate_up` `[2*moe_int]`, `xr_act` `[moe_int]` are the shared scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_decode_megakernel_b(
        &mut self,
        g_res2: &GpuTensor,
        hidden: &GpuTensor,
        // Phase 2 (fold_oproj): o_proj + attn affine folded into B's head. When
        // fold_oproj is false these are unused (pass any valid tensors).
        ctx: &GpuTensor,
        o_proj_w: &GpuTensor,
        ctx_rot: &GpuTensor,
        pa_rs: &[GpuTensor; 4],
        q_dim: usize,
        fold_oproj: bool,
        // Phase 3 (fold_attn): stage 9 KV write + flash attention folded into B.
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        pos_buf: &hip_bridge::DeviceBuffer,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
        attn_scale: f32,
        fold_attn: bool,
        post_attn_ln: &GpuTensor,
        pa_xrot: &GpuTensor,
        mk_norm: &GpuTensor,
        dp_w: &GpuTensor,
        dp_b: &GpuTensor,
        rstate: &GpuTensor,
        escale: Option<&GpuTensor>,
        rnorm_w: &GpuTensor,
        fc1_w: &GpuTensor,
        fc1_b: &GpuTensor,
        fc2_w: &GpuTensor,
        fc2_b: &GpuTensor,
        out_w: &GpuTensor,
        bbias: &GpuTensor,
        sel_idx: &GpuTensor,
        sel_gate: &GpuTensor,
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        gate_up: &GpuTensor,
        xr_act: &GpuTensor,
        pm_rs: &[GpuTensor; 4],
        h: usize,
        rh: usize,
        n_route: usize,
        moe_int: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "zaya_megakernel",
            kernels::ZAYA_MEGAKERNEL_SRC,
            "zaya_decode_megakernel_b",
        )?;
        self.ensure_mq_signs()?;
        // Size the cooperative grid once to the device residency limit
        // (occupancy/MP × MP count). A cooperative launch that exceeds it fails
        // loudly with hipErrorCooperativeLaunchTooLarge.
        let grid_blocks = match self.zaya_megakernel_grid {
            Some(g) => g,
            None => {
                let f = &self.functions["zaya_decode_megakernel_b"];
                let per_mp = self.hip.occupancy_max_active_blocks_per_mp(f, 256, 0)?;
                let mp = self.hip.multiprocessor_count(self.device_id)?;
                let g = (per_mp.max(1) as u32) * (mp.max(1) as u32);
                self.zaya_megakernel_grid = Some(g);
                g
            }
        };

        let s1 = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let has_eda: i32 = escale.is_some() as i32;
        let esc = escale.map_or(dp_b.buf.as_ptr(), |e| e.buf.as_ptr());
        let (gr, hd) = (g_res2.buf.as_ptr(), hidden.buf.as_ptr());
        let (ctxp, opw, ctxr) = (
            ctx.buf.as_ptr(),
            o_proj_w.buf.as_ptr(),
            ctx_rot.buf.as_ptr(),
        );
        let (pahs, pahb, pars2, parb) = (
            pa_rs[0].buf.as_ptr(),
            pa_rs[1].buf.as_ptr(),
            pa_rs[2].buf.as_ptr(),
            pa_rs[3].buf.as_ptr(),
        );
        let qdi = q_dim as i32;
        let foldi = fold_oproj as i32;
        let (qy, ky, vy) = (query.buf.as_ptr(), key.buf.as_ptr(), value.buf.as_ptr());
        let (kc, vc, posp) = (k_cache.buf.as_ptr(), v_cache.buf.as_ptr(), pos_buf.as_ptr());
        let (nkvi, hdi, msi) = (nkv as i32, head_dim as i32, max_seq as i32);
        let ascale = attn_scale;
        let fai = fold_attn as i32;
        let pal = post_attn_ln.buf.as_ptr();
        let (pxr, mkn) = (pa_xrot.buf.as_ptr(), mk_norm.buf.as_ptr());
        let (dpw, dpb, rst, rnw) = (
            dp_w.buf.as_ptr(),
            dp_b.buf.as_ptr(),
            rstate.buf.as_ptr(),
            rnorm_w.buf.as_ptr(),
        );
        let (f1w, f1b, f2w, f2b, ow) = (
            fc1_w.buf.as_ptr(),
            fc1_b.buf.as_ptr(),
            fc2_w.buf.as_ptr(),
            fc2_b.buf.as_ptr(),
            out_w.buf.as_ptr(),
        );
        let (bb, si, sg) = (
            bbias.buf.as_ptr(),
            sel_idx.buf.as_ptr(),
            sel_gate.buf.as_ptr(),
        );
        let (gup_p, dwn_p, gup, xra) = (
            gate_up_ptrs.buf.as_ptr(),
            down_ptrs.buf.as_ptr(),
            gate_up.buf.as_ptr(),
            xr_act.buf.as_ptr(),
        );
        let (phs, phb, prs, prb) = (
            pm_rs[0].buf.as_ptr(),
            pm_rs[1].buf.as_ptr(),
            pm_rs[2].buf.as_ptr(),
            pm_rs[3].buf.as_ptr(),
        );
        let (hi, rhi, nri, mii) = (h as i32, rh as i32, n_route as i32, moe_int as i32);
        let mut p: Vec<*mut c_void> = vec![
            &gr as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ctxp as *const _ as *mut c_void,
            &opw as *const _ as *mut c_void,
            &ctxr as *const _ as *mut c_void,
            &pahs as *const _ as *mut c_void,
            &pahb as *const _ as *mut c_void,
            &pars2 as *const _ as *mut c_void,
            &parb as *const _ as *mut c_void,
            &qdi as *const _ as *mut c_void,
            &foldi as *const _ as *mut c_void,
            &qy as *const _ as *mut c_void,
            &ky as *const _ as *mut c_void,
            &vy as *const _ as *mut c_void,
            &kc as *const _ as *mut c_void,
            &vc as *const _ as *mut c_void,
            &posp as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &msi as *const _ as *mut c_void,
            &ascale as *const _ as *mut c_void,
            &fai as *const _ as *mut c_void,
            &pal as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &pxr as *const _ as *mut c_void,
            &mkn as *const _ as *mut c_void,
            &dpw as *const _ as *mut c_void,
            &dpb as *const _ as *mut c_void,
            &rst as *const _ as *mut c_void,
            &esc as *const _ as *mut c_void,
            &rnw as *const _ as *mut c_void,
            &f1w as *const _ as *mut c_void,
            &f1b as *const _ as *mut c_void,
            &f2w as *const _ as *mut c_void,
            &f2b as *const _ as *mut c_void,
            &ow as *const _ as *mut c_void,
            &bb as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &sg as *const _ as *mut c_void,
            &gup_p as *const _ as *mut c_void,
            &dwn_p as *const _ as *mut c_void,
            &gup as *const _ as *mut c_void,
            &xra as *const _ as *mut c_void,
            &phs as *const _ as *mut c_void,
            &phb as *const _ as *mut c_void,
            &prs as *const _ as *mut c_void,
            &prb as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &rhi as *const _ as *mut c_void,
            &nri as *const _ as *mut c_void,
            &mii as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &has_eda as *const _ as *mut c_void,
        ];
        let f = &self.functions["zaya_decode_megakernel_b"];
        unsafe {
            self.hip.launch_cooperative_kernel(
                f,
                [grid_blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut p,
            )
        }
    }

    /// ZAYA decode cooperative megakernel — Phase 1 (megakernel-A, the front
    /// half). Fuses stages 1–8 (input rmsnorm+rotate → fused qkv Oq8 gemv →
    /// qk-prep → conv window+ring → depthwise/grouped conv1d → add-conv-residual
    /// + value assemble → q/k L2-norm+scale → partial RoPE) into ONE cooperative
    /// launch. `w_*` are the four planar Oq8 in-projection weight buffers (K=h).
    /// Attention + o_proj stay separate launches after this. Env-gated + diffed.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_decode_megakernel_a(
        &mut self,
        hidden: &GpuTensor,
        input_ln: &GpuTensor,
        qkv_xrot: &GpuTensor,
        w_q: &GpuTensor,
        w_k: &GpuTensor,
        w_vc: &GpuTensor,
        w_vd: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        vcur: &GpuTensor,
        vdel: &GpuTensor,
        qres: &GpuTensor,
        kres: &GpuTensor,
        cur_qk: &GpuTensor,
        conv_dw_w: &GpuTensor,
        conv_dw_b: &GpuTensor,
        conv_gr_w: &GpuTensor,
        conv_gr_b: &GpuTensor,
        conv_ring: &GpuTensor,
        window: &GpuTensor,
        dw: &GpuTensor,
        gw: &GpuTensor,
        delayed_v: &GpuTensor,
        query: &GpuTensor,
        key: &GpuTensor,
        value: &GpuTensor,
        qk_temp: &GpuTensor,
        pos_buf: &hip_bridge::DeviceBuffer,
        h: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        q_dim: usize,
        k_dim: usize,
        v_half: usize,
        conv_ch: usize,
        pad: usize,
        dwk: usize,
        grk: usize,
        dw_len: usize,
        n_rot: usize,
        rope_theta: f32,
        l2_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "zaya_megakernel",
            kernels::ZAYA_MEGAKERNEL_SRC,
            "zaya_decode_megakernel_a",
        )?;
        self.ensure_mq_signs()?;
        let grid_blocks = match self.zaya_megakernel_a_grid {
            Some(g) => g,
            None => {
                let f = &self.functions["zaya_decode_megakernel_a"];
                let per_mp = self.hip.occupancy_max_active_blocks_per_mp(f, 256, 0)?;
                let mp = self.hip.multiprocessor_count(self.device_id)?;
                let g = (per_mp.max(1) as u32) * (mp.max(1) as u32);
                self.zaya_megakernel_a_grid = Some(g);
                g
            }
        };
        let s1 = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let hidp = hidden.buf.as_ptr();
        let iln = input_ln.buf.as_ptr();
        let xrot = qkv_xrot.buf.as_ptr();
        let (wq, wk, wvc, wvd) = (
            w_q.buf.as_ptr(),
            w_k.buf.as_ptr(),
            w_vc.buf.as_ptr(),
            w_vd.buf.as_ptr(),
        );
        let (qp, kp, vcp, vdp) = (
            q.buf.as_ptr(),
            k.buf.as_ptr(),
            vcur.buf.as_ptr(),
            vdel.buf.as_ptr(),
        );
        let (qrp, krp, cqk) = (qres.buf.as_ptr(), kres.buf.as_ptr(), cur_qk.buf.as_ptr());
        let (cdw, cdb, cgw, cgb) = (
            conv_dw_w.buf.as_ptr(),
            conv_dw_b.buf.as_ptr(),
            conv_gr_w.buf.as_ptr(),
            conv_gr_b.buf.as_ptr(),
        );
        let (ring, win, dwp, gwp) = (
            conv_ring.buf.as_ptr(),
            window.buf.as_ptr(),
            dw.buf.as_ptr(),
            gw.buf.as_ptr(),
        );
        let (delv, qy, ky, vy) = (
            delayed_v.buf.as_ptr(),
            query.buf.as_ptr(),
            key.buf.as_ptr(),
            value.buf.as_ptr(),
        );
        let (qkt, posp) = (qk_temp.buf.as_ptr(), pos_buf.as_ptr());
        let (hi, nqi, nkvi, hdi) = (h as i32, nq as i32, nkv as i32, hd as i32);
        let (qdi, kdi, vhi) = (q_dim as i32, k_dim as i32, v_half as i32);
        let (cci, padi, dwki, grki, dwli, nri) = (
            conv_ch as i32,
            pad as i32,
            dwk as i32,
            grk as i32,
            dw_len as i32,
            n_rot as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &hidp as *const _ as *mut c_void,
            &iln as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &xrot as *const _ as *mut c_void,
            &wq as *const _ as *mut c_void,
            &wk as *const _ as *mut c_void,
            &wvc as *const _ as *mut c_void,
            &wvd as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vcp as *const _ as *mut c_void,
            &vdp as *const _ as *mut c_void,
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &cqk as *const _ as *mut c_void,
            &cdw as *const _ as *mut c_void,
            &cdb as *const _ as *mut c_void,
            &cgw as *const _ as *mut c_void,
            &cgb as *const _ as *mut c_void,
            &ring as *const _ as *mut c_void,
            &win as *const _ as *mut c_void,
            &dwp as *const _ as *mut c_void,
            &gwp as *const _ as *mut c_void,
            &delv as *const _ as *mut c_void,
            &qy as *const _ as *mut c_void,
            &ky as *const _ as *mut c_void,
            &vy as *const _ as *mut c_void,
            &qkt as *const _ as *mut c_void,
            &posp as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &qdi as *const _ as *mut c_void,
            &kdi as *const _ as *mut c_void,
            &vhi as *const _ as *mut c_void,
            &cci as *const _ as *mut c_void,
            &padi as *const _ as *mut c_void,
            &dwki as *const _ as *mut c_void,
            &grki as *const _ as *mut c_void,
            &dwli as *const _ as *mut c_void,
            &nri as *const _ as *mut c_void,
            &rope_theta as *const _ as *mut c_void,
            &l2_scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let f = &self.functions["zaya_decode_megakernel_a"];
        unsafe {
            self.hip.launch_cooperative_kernel(
                f,
                [grid_blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut p,
            )
        }
    }
}
