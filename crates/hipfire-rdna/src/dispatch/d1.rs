//! D1: a Qwen3.6-35B-A3B DeltaNetMoe layer as one counter-scheduled persistent
//! kernel (`kernels/src/d1_dn_moe_layer.hip`,
//! docs/plans/2026-10-09-d1-persistent-layer-kernel.md). It covers the
//! attention half -- norm, the four in-projections, gates, conv, q/k norm,
//! delta-net recurrence, gated norm, wo + residual -- bit-identical to that op
//! chain.

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::collections::HashMap;
use std::ffi::c_void;

// Mirrors `D1Proj` / `D1Params` in the kernel.
#[repr(C)]
#[derive(Clone, Copy)]
struct D1Proj {
    w: *mut c_void,
    awq: *mut c_void,
    y: *mut c_void,
    xq: *mut c_void,
    xs: *mut c_void,
    resid: *mut c_void,
    m: i32,
    k: i32,
    block_stride: i32,
    pad: i32,
}

#[repr(C)]
struct D1Params {
    x: *mut c_void,
    attn_norm: *mut c_void,
    xn: *mut c_void,
    signs1: *mut c_void,
    signs2: *mut c_void,
    dt_bias: *mut c_void,
    a_log: *mut c_void,
    counters: *mut c_void,
    q_raw: *mut c_void,
    k_raw: *mut c_void,
    v: *mut c_void,
    q: *mut c_void,
    k: *mut c_void,
    conv_weight: *mut c_void,
    conv_ptrs: *mut c_void,
    s_ptrs: *mut c_void,
    row_session: *mut c_void,
    attn_out: *mut c_void,
    normed: *mut c_void,
    norm_weight: *mut c_void,
    proj: [D1Proj; 5],
    n: i32,
    dim: i32,
    n_v_heads: i32,
    n_k_heads: i32,
    k_dim: i32,
    v_dim: i32,
    sessions: i32,
    ptr_stride: i32,
    layer: i32,
    stochastic_round: i32,
    eps: f32,
    q_scale: f32,
}

/// The kernel's own int8 activations (one per projection, so the quantizes
/// can run at once), its counters and the per-row-count grids.
pub struct D1Scratch {
    counters: GpuTensor,
    xq: GpuTensor,
    xs: GpuTensor,
    ks: [usize; 5],
    grid: HashMap<usize, u32>,
}

/// One projection: OqCompact weight blocks, its AWQ scale, the output, M, K and
/// the block stride. wo's output is added into `D1AttnHalf::x` instead.
pub type D1ProjArgs<'a> = (
    &'a GpuTensor,
    &'a GpuTensor,
    &'a GpuTensor,
    usize,
    usize,
    usize,
);

/// What one DeltaNetMoe attention half reads and writes, named as the chain
/// names them (`PrefillBatchScratch`).
pub struct D1AttnHalf<'a> {
    /// The residual stream: the norm reads it, wo adds into it.
    pub x: &'a GpuTensor,
    pub attn_norm: &'a GpuTensor,
    pub xn: &'a GpuTensor,
    /// wqkv, wz, w_beta, w_alpha, wo.
    pub proj: [D1ProjArgs<'a>; 5],
    pub dt_bias: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub q_raw: &'a GpuTensor,
    pub k_raw: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    /// u64 per (session, layer) state pointers and the row -> session map.
    pub conv_ptrs: &'a GpuTensor,
    pub s_ptrs: &'a GpuTensor,
    pub row_session: &'a GpuTensor,
    pub n: usize,
    pub n_v_heads: usize,
    pub n_k_heads: usize,
    pub sessions: usize,
    pub ptr_stride: usize,
    pub layer: usize,
    pub eps: f32,
}

const MAX_ROWS: usize = 32;

impl Gpu {
    /// Whether the op chain runs a projection of this K the way D1 copies it:
    /// gfx1151's rmsnorm, and the WIDE multicol at this row count.
    pub fn d1_attn_in_ok(&self, n: usize, k: usize) -> bool {
        (1..=MAX_ROWS).contains(&n)
            && k % 1024 == 0
            && self.arch_caps.is_gfx1151()
            && self.oq_compact_multicol_takes(n, k)
            && (self.flags.oq_compact_multicol_wide || self.oq_batch_serving)
    }

    /// The attention half in one cooperative launch.
    pub fn d1_dn_moe_attn_half(&mut self, a: &D1AttnHalf<'_>) -> HipResult<()> {
        self.bind_thread()?;
        let (n, dim) = (a.n, a.proj[0].4);
        let ks = a.proj.map(|p| p.4);
        assert!(
            ks.iter().all(|&k| self.d1_attn_in_ok(n, k)),
            "d1: n={n} K={ks:?} not covered"
        );
        assert!(a.proj[2].3 == a.n_v_heads && a.proj[3].3 == a.n_v_heads);
        self.ensure_mq_signs()?;
        let func = format!("d1_dn_moe_layer_n{n}");
        if !self.functions.contains_key(&func) {
            let src = format!(
                "#define D1_N {n}\n{}",
                kernels::D1_DN_MOE_LAYER_SRC
                    .replace("void d1_dn_moe_layer(", &format!("void {func}("))
            );
            self.ensure_kernel(&func, &src, &func)?;
        }
        if self.d1.as_ref().is_none_or(|s| s.ks != ks) {
            if let Some(old) = self.d1.take() {
                self.free_tensor(old.counters)?;
                self.free_tensor(old.xq)?;
                self.free_tensor(old.xs)?;
            }
            let total_k: usize = ks.iter().sum();
            let counters = self.zeros(&[32], DType::F32)?;
            let xq = self.alloc_tensor(&[MAX_ROWS * total_k], DType::Raw)?;
            let xs = self.alloc_tensor(&[MAX_ROWS * total_k / 256], DType::F32)?;
            self.d1 = Some(D1Scratch {
                counters,
                xq,
                xs,
                ks,
                grid: HashMap::new(),
            });
        }
        // Full residency: the spin-waits are deadlock-free only if every
        // workgroup is resident, which the cooperative launch enforces.
        let grid = match self.d1.as_ref().unwrap().grid.get(&n) {
            Some(&g) => g,
            None => {
                let f = &self.functions[&func];
                let per_mp = self.hip.occupancy_max_active_blocks_per_mp(f, 256, 0)?;
                let mp = self.hip.multiprocessor_count(self.device_id)?;
                let g = (per_mp.max(1) as u32) * (mp.max(1) as u32);
                self.d1.as_mut().unwrap().grid.insert(n, g);
                g
            }
        };
        let s = self.d1.as_ref().unwrap();
        let (xq0, xs0) = (s.xq.buf.as_ptr() as *mut u8, s.xs.buf.as_ptr() as *mut f32);
        let mut off = 0;
        let proj = std::array::from_fn(|i| {
            let (w, awq, y, m, k, bs) = a.proj[i];
            let p = D1Proj {
                w: w.buf.as_ptr(),
                awq: awq.buf.as_ptr(),
                y: y.buf.as_ptr(),
                xq: unsafe { xq0.add(off * MAX_ROWS) } as *mut c_void,
                xs: unsafe { xs0.add(off * MAX_ROWS / 256) } as *mut c_void,
                resid: if i == 4 {
                    a.x.buf.as_ptr()
                } else {
                    std::ptr::null_mut()
                },
                m: m as i32,
                k: k as i32,
                block_stride: bs as i32,
                pad: 0,
            };
            off += k;
            p
        });
        let hd = 128;
        let params = D1Params {
            x: a.x.buf.as_ptr(),
            attn_norm: a.attn_norm.buf.as_ptr(),
            xn: a.xn.buf.as_ptr(),
            signs1: self.mq_signs1.as_ref().unwrap().buf.as_ptr(),
            signs2: self.mq_signs2.as_ref().unwrap().buf.as_ptr(),
            dt_bias: a.dt_bias.buf.as_ptr(),
            a_log: a.a_log.buf.as_ptr(),
            counters: s.counters.buf.as_ptr(),
            q_raw: a.q_raw.buf.as_ptr(),
            k_raw: a.k_raw.buf.as_ptr(),
            v: a.v.buf.as_ptr(),
            q: a.q.buf.as_ptr(),
            k: a.k.buf.as_ptr(),
            conv_weight: a.conv_weight.buf.as_ptr(),
            conv_ptrs: a.conv_ptrs.buf.as_ptr(),
            s_ptrs: a.s_ptrs.buf.as_ptr(),
            row_session: a.row_session.buf.as_ptr(),
            attn_out: a.attn_out.buf.as_ptr(),
            normed: a.normed.buf.as_ptr(),
            norm_weight: a.norm_weight.buf.as_ptr(),
            proj,
            n: n as i32,
            dim: dim as i32,
            n_v_heads: a.n_v_heads as i32,
            n_k_heads: a.n_k_heads as i32,
            k_dim: (a.n_k_heads * hd) as i32,
            v_dim: (a.n_v_heads * hd) as i32,
            sessions: a.sessions as i32,
            ptr_stride: a.ptr_stride as i32,
            layer: a.layer as i32,
            stochastic_round: super::gated::fp16_state_dither() as i32,
            eps: a.eps,
            // The chain's `1.0 / (hd as f32).sqrt()`, to the bit.
            q_scale: 1.0 / (hd as f32).sqrt(),
        };
        let mut p = [&params as *const D1Params as *mut c_void];
        let f = &self.functions[&func];
        unsafe {
            self.hip.launch_cooperative_kernel(
                f,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut p,
            )?;
        }
        for t in [
            a.x, a.xn, a.q_raw, a.k_raw, a.v, a.q, a.k, a.attn_out, a.normed,
        ]
        .into_iter()
        .chain(a.proj.iter().map(|p| p.2))
        {
            self.invalidate_x_caches_for(t.buf.as_ptr());
        }
        Ok(())
    }
}
