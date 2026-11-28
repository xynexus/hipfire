// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA slot gather from a KVarN cache. See `kernels/src/qsa_gather_kvarn.hip`.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Dequantise the `n_slots` positions in `slots` (i32-as-f32 bits) from a
    /// KVarN K cache (`recs` full blocks + `window` open block) and a Q8_0 V
    /// cache into `out_k` / `out_v`, both `[n_kv, n_slots, head_dim]` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_gather_kvarn(
        &mut self,
        recs: &GpuTensor,
        window: &GpuTensor,
        vq8: &GpuTensor,
        slots: &GpuTensor,
        out_k: &GpuTensor,
        out_v: &GpuTensor,
        n_kv: usize,
        n_slots: usize,
        head_dim: usize,
        group: usize,
        record_bytes: usize,
        bits: usize,
        n_full_blocks: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            matches!(bits, 2 | 4 | 8),
            "qsa_gather_kvarn: bits must be 2, 4 or 8"
        );
        assert_eq!(
            (n_kv * head_dim) % 32,
            0,
            "qsa_gather_kvarn: kv_dim % 32 != 0 (Q8_0 V)"
        );
        if n_slots == 0 {
            return Ok(());
        }
        self.ensure_kernel(
            "qsa_gather_kvarn",
            kernels::QSA_GATHER_KVARN_SRC,
            "qsa_gather_kvarn",
        )?;
        let func = &self.functions["qsa_gather_kvarn"];
        let (rp, wp, vp, sp) = (
            recs.buf.as_ptr(),
            window.buf.as_ptr(),
            vq8.buf.as_ptr(),
            slots.buf.as_ptr(),
        );
        let (kp, op) = (out_k.buf.as_ptr(), out_v.buf.as_ptr());
        let (mut nk, mut ns, mut hd, mut g) =
            (n_kv as i32, n_slots as i32, head_dim as i32, group as i32);
        let (mut rb, mut b, mut nf) = (record_bytes as i32, bits as i32, n_full_blocks as i32);
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut g as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut nf as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_slots as u32, n_kv as u32, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
