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

    fn zaya_launch(&self, func: &str, threads: usize, params: &mut Vec<*mut c_void>) -> HipResult<()> {
        let grid = (threads as u32).div_ceil(BLOCK);
        let f = &self.functions[func];
        unsafe {
            self.hip
                .launch_kernel(f, [grid, 1, 1], [BLOCK, 1, 1], 0, self.stream_ref(), params)
        }
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
        let (op, xp, sp, bp) = (out.buf.as_ptr(), x.buf.as_ptr(), scale.buf.as_ptr(), bias.buf.as_ptr());
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
        let (hsp, hbp, rsp, rbp) = (hs.buf.as_ptr(), hb.buf.as_ptr(), rs.buf.as_ptr(), rb.buf.as_ptr());
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
        let (op, ip, wp, bp) = (out.buf.as_ptr(), input.buf.as_ptr(), weight.buf.as_ptr(), bias.buf.as_ptr());
        let (ch, gr, kn, il, ol) = (channels as i32, groups as i32, kernel as i32, in_len as i32, out_len as i32);
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
        let (qrp, krp, qp, kp) = (query_res.buf.as_ptr(), key_res.buf.as_ptr(), q.buf.as_ptr(), k.buf.as_ptr());
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
        let (rp, pp, sp) = (router_hidden.buf.as_ptr(), prev.buf.as_ptr(), scale.buf.as_ptr());
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
}
