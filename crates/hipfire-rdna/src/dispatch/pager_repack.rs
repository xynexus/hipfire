// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-side repack of routed-expert modules at page-in. See
//! `kernels/src/pager_repack.hip`.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Canonical `Oq4G256` bytes (`src`, `m * k/256 * 130` B) into the arch
    /// combined layout (`dst`, `oq4_arch_combined_len(m, k)` B). Byte-identical
    /// to `hipfire_runtime::oq4_arch::oq4_pack_arch_combined`.
    pub fn pager_oq4_combined(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 256, 0, "pager_oq4_combined: K % 256 != 0");
        self.ensure_kernel(
            "pager_repack",
            kernels::PAGER_REPACK_SRC,
            "pager_oq4_combined",
        )?;
        let func = &self.functions["pager_oq4_combined"];
        let (sp, dp) = (src.buf.as_ptr(), dst.buf.as_ptr());
        let (mut mi, mut ki) = (m as i32, k as i32);
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let blocks = (m * (k / 256)) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [128, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Interleaved `OqPlusCompact` blocks (`n_groups` of `block_bytes`, group
    /// 256 or 128) into the device split planes, overlays normalised.
    /// Byte-identical to `normalize_compact_overlays` + `split_compact_planes`.
    pub fn pager_compact_split(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n_groups: usize,
        block_bytes: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "pager_repack",
            kernels::PAGER_REPACK_SRC,
            "pager_compact_split",
        )?;
        let func = &self.functions["pager_compact_split"];
        let (sp, dp) = (src.buf.as_ptr(), dst.buf.as_ptr());
        let (mut n, mut bb, mut g) = (n_groups as i32, block_bytes as i32, group as i32);
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut bb as *mut _ as *mut c_void,
            &mut g as *mut _ as *mut c_void,
        ];
        let blocks = n_groups.div_ceil(256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
