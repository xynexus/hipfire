// SPDX-License-Identifier: Apache-2.0
//! Persistent-handle AMDGPU MMIO register sampler over a dlopen'd libdrm_amdgpu.
//!
//! Extracted from libdrm-amdgpu-sys-rs (Umio-Yasuno, MIT — same author as
//! amdgpu_top) so hipfire carries only the few pieces the terminal monitor needs
//! to sample the GRBM/GRBM2 "block busy" registers: device init,
//! `amdgpu_read_mm_registers`, and the family→chip-class mapping. It is dlopen'd
//! (not linked) exactly like this crate's other libdrm probe, so there is no
//! `libdrm-dev` / `-ldrm` build dependency — only the runtime `.so.1`. The
//! chip-class table mirrors mesa's `amd_family.h` / `amdgpu_asic_addr.h`.

use std::ffi::{c_void, CString};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::sync::Arc;

/// MMIO dword offsets for the graphics-block busy registers (mmGRBM_STATUS /
/// mmGRBM_STATUS2); identical across every supported GFX generation.
pub const GRBM_OFFSET: u32 = 0x2004;
pub const GRBM2_OFFSET: u32 = 0x2002;

// AMDGPU family ids (drm/amdgpu_drm.h). Only GFX9+ families the fleet spans.
const FAMILY_AI: u32 = 141; // Vega / MI (GFX9)
const FAMILY_RV: u32 = 142; // Raven / Renoir (GFX9)
const FAMILY_NV: u32 = 143; // Navi1x (GFX10) + Navi2x (GFX10_3)
const FAMILY_VGH: u32 = 144; // Van Gogh (GFX10_3)
const FAMILY_GC_11_0_0: u32 = 145; // Navi3x (GFX11)
const FAMILY_YC: u32 = 146; // Rembrandt (GFX10_3)
const FAMILY_GC_11_0_1: u32 = 148; // Phoenix / Hawk Point (GFX11)
const FAMILY_GC_10_3_6: u32 = 149; // Raphael/Mendocino (GFX10_3)
const FAMILY_GC_11_5_0: u32 = 150; // Strix (GFX11_5)
const FAMILY_GC_10_3_7: u32 = 151; // (GFX10_3)
const FAMILY_GC_12_0_0: u32 = 152; // RDNA4 (GFX12)

/// AMDGPU chip class (GFX generation), coarse enough to pick a register layout.
/// Ordering is significant: the GRBM2 bit tables key off `>=` comparisons.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChipClass {
    Unknown,
    Gfx9,
    Gfx10,
    Gfx10_3,
    Gfx11,
    Gfx11_5,
    Gfx12,
}

impl ChipClass {
    /// Derive from `amdgpu_gpu_info.family_id` + `chip_external_rev`, mirroring
    /// mesa's amd_family.h family→ASIC→class mapping. The NV family spans RDNA1
    /// (Navi10-14, GFX10) and RDNA2 (Navi21-24, GFX10_3); `external_rev` splits
    /// them per amdgpu_asic_addr.h.
    fn from_family(family_id: u32, external_rev: u32) -> Self {
        match family_id {
            FAMILY_AI | FAMILY_RV => Self::Gfx9,
            // 0x28..0x50 = Navi21-24 (GFX10_3); everything else in NV (Navi10-14,
            // GFX1013 Cyan Skillfish) is GFX10.
            FAMILY_NV => {
                if (0x28..0x50).contains(&external_rev) {
                    Self::Gfx10_3
                } else {
                    Self::Gfx10
                }
            }
            FAMILY_VGH | FAMILY_YC | FAMILY_GC_10_3_6 | FAMILY_GC_10_3_7 => Self::Gfx10_3,
            FAMILY_GC_11_0_0 | FAMILY_GC_11_0_1 => Self::Gfx11,
            FAMILY_GC_11_5_0 => Self::Gfx11_5,
            FAMILY_GC_12_0_0 => Self::Gfx12,
            _ => Self::Unknown,
        }
    }
}

type AmdgpuDeviceHandle = *mut c_void;
type FnDeviceInitialize =
    unsafe extern "C" fn(i32, *mut u32, *mut u32, *mut AmdgpuDeviceHandle) -> i32;
type FnDeviceDeinitialize = unsafe extern "C" fn(AmdgpuDeviceHandle) -> i32;
// amdgpu_read_mm_registers(dev, dword_offset, count, instance, flags, *values)
type FnReadMmRegisters =
    unsafe extern "C" fn(AmdgpuDeviceHandle, u32, u32, u32, u32, *mut u32) -> i32;
type FnQueryGpuInfo = unsafe extern "C" fn(AmdgpuDeviceHandle, *mut AmdgpuGpuInfo) -> i32;

/// `struct amdgpu_gpu_info` — full 416-byte layout (libdrm amdgpu.h). The full
/// size matters: `amdgpu_query_gpu_info` writes the whole struct, so a truncated
/// buffer would corrupt memory. Only `family_id`/`chip_external_rev` are read.
#[repr(C)]
#[derive(Default)]
struct AmdgpuGpuInfo {
    asic_id: u32,
    chip_rev: u32,
    chip_external_rev: u32,
    family_id: u32,
    ids_flags: u64,
    max_engine_clk: u64,
    max_memory_clk: u64,
    num_shader_engines: u32,
    num_shader_arrays_per_engine: u32,
    avail_quad_shader_pipes: u32,
    max_quad_shader_pipes: u32,
    cache_entries_per_quad_pipe: u32,
    num_hw_gfx_contexts: u32,
    rb_pipes: u32,
    enabled_rb_pipes_mask: u32,
    gpu_counter_freq: u32,
    backend_disable: [u32; 4],
    mc_arb_ramcfg: u32,
    gb_addr_cfg: u32,
    gb_tile_mode: [u32; 32],
    gb_macro_tile_mode: [u32; 16],
    pa_sc_raster_cfg: [u32; 4],
    pa_sc_raster_cfg1: [u32; 4],
    cu_active_number: u32,
    cu_ao_mask: u32,
    cu_bitmap: [[u32; 4]; 4],
    vram_type: u32,
    vram_bit_width: u32,
    ce_ram_size: u32,
    vce_harvest_config: u32,
    pci_rev_id: u32,
}

/// A dlopen'd libdrm_amdgpu holding the handful of symbols the sampler needs.
/// Kept alive (via `Arc`) for as long as any [`AmdgpuRegDevice`] opened from it.
pub struct AmdgpuRegLib {
    lib: *mut c_void,
    device_initialize: FnDeviceInitialize,
    device_deinitialize: FnDeviceDeinitialize,
    read_mm_registers: FnReadMmRegisters,
    query_gpu_info: FnQueryGpuInfo,
}

impl AmdgpuRegLib {
    /// dlopen libdrm_amdgpu and resolve the register-sampling symbols. `None`
    /// if the driver library or any symbol is absent.
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = dlopen_first(&["libdrm_amdgpu.so.1", "libdrm_amdgpu.so"])?;
            let out = Self {
                device_initialize: std::mem::transmute::<*mut c_void, FnDeviceInitialize>(
                    dlsym_required(lib, "amdgpu_device_initialize")?,
                ),
                device_deinitialize: std::mem::transmute::<*mut c_void, FnDeviceDeinitialize>(
                    dlsym_required(lib, "amdgpu_device_deinitialize")?,
                ),
                read_mm_registers: std::mem::transmute::<*mut c_void, FnReadMmRegisters>(
                    dlsym_required(lib, "amdgpu_read_mm_registers")?,
                ),
                query_gpu_info: std::mem::transmute::<*mut c_void, FnQueryGpuInfo>(dlsym_required(
                    lib,
                    "amdgpu_query_gpu_info",
                )?),
                lib,
            };
            Some(Arc::new(out))
        }
    }

    /// Open a persistent device handle on a render node (e.g.
    /// `/dev/dri/renderD128`). The device keeps the library alive.
    pub fn open_device(self: &Arc<Self>, render_node: &Path) -> Option<AmdgpuRegDevice> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(render_node)
            .ok()?;
        let fd = OwnedFd::from(file);
        unsafe {
            let mut major = 0u32;
            let mut minor = 0u32;
            let mut handle: AmdgpuDeviceHandle = std::ptr::null_mut();
            if (self.device_initialize)(fd.as_raw_fd(), &mut major, &mut minor, &mut handle) != 0
                || handle.is_null()
            {
                return None;
            }
            let mut info = AmdgpuGpuInfo::default();
            let chip_class = if (self.query_gpu_info)(handle, &mut info) == 0 {
                ChipClass::from_family(info.family_id, info.chip_external_rev)
            } else {
                ChipClass::Unknown
            };
            Some(AmdgpuRegDevice {
                lib: Arc::clone(self),
                handle,
                _fd: fd,
                chip_class,
            })
        }
    }
}

impl Drop for AmdgpuRegLib {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.lib);
        }
    }
}

// The library handle + resolved fn pointers are immutable after load and only
// used for read-only ioctls; safe to share across threads.
unsafe impl Send for AmdgpuRegLib {}
unsafe impl Sync for AmdgpuRegLib {}

/// A persistent amdgpu device handle for repeated MMIO register reads.
pub struct AmdgpuRegDevice {
    lib: Arc<AmdgpuRegLib>,
    handle: AmdgpuDeviceHandle,
    _fd: OwnedFd, // owns the render-node fd for the device's lifetime
    chip_class: ChipClass,
}

impl AmdgpuRegDevice {
    /// The device's GFX generation (for register-layout selection).
    pub fn chip_class(&self) -> ChipClass {
        self.chip_class
    }

    /// Read a single 32-bit MMIO register at `dword_offset`. `None` if the
    /// kernel rejects the offset (not on the allow-list) or the read fails.
    pub fn read_mm_register(&self, dword_offset: u32) -> Option<u32> {
        unsafe {
            let mut out = 0u32;
            let r = (self.lib.read_mm_registers)(
                self.handle,
                dword_offset,
                1,           // count
                0xFFFF_FFFF, // instance mask (broadcast)
                0,           // flags
                &mut out,
            );
            (r == 0).then_some(out)
        }
    }
}

impl Drop for AmdgpuRegDevice {
    fn drop(&mut self) {
        // Deinit before the OwnedFd drop closes the fd (libdrm does not own it).
        unsafe {
            (self.lib.device_deinitialize)(self.handle);
        }
    }
}

// Bound to a single device fd; the handle is only read through &self ioctls.
unsafe impl Send for AmdgpuRegDevice {}

unsafe fn dlopen_first(names: &[&str]) -> Option<*mut c_void> {
    for name in names {
        let c_name = CString::new(*name).ok()?;
        let lib = libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if !lib.is_null() {
            return Some(lib);
        }
    }
    None
}

unsafe fn dlsym_required(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    let c_name = CString::new(name).ok()?;
    let symbol = libc::dlsym(lib, c_name.as_ptr());
    (!symbol.is_null()).then_some(symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_class_ordering_matches_grbm2_selection() {
        // GRBM2 tables branch on `>=`; ordering must be monotonic by generation.
        assert!(ChipClass::Gfx9 < ChipClass::Gfx10);
        assert!(ChipClass::Gfx10 < ChipClass::Gfx10_3);
        assert!(ChipClass::Gfx10_3 < ChipClass::Gfx11);
        assert!(ChipClass::Gfx11 < ChipClass::Gfx11_5);
        assert!(ChipClass::Gfx11_5 < ChipClass::Gfx12);
        // GFX11 must fall into the GFX10_3 branch (no dedicated GFX11 table).
        assert!(ChipClass::Gfx10_3 <= ChipClass::Gfx11);
    }

    #[test]
    fn family_maps_to_expected_class() {
        // Fleet: Vega20 (AI/GFX9), Phoenix gfx1103 (GC_11_0_1/GFX11),
        // Strix Halo gfx1151 (GC_11_5_0/GFX11_5), W7800 gfx1201 (GC_12_0_0/GFX12).
        assert_eq!(ChipClass::from_family(FAMILY_AI, 0x28), ChipClass::Gfx9);
        assert_eq!(
            ChipClass::from_family(FAMILY_GC_11_0_1, 0x01),
            ChipClass::Gfx11
        );
        assert_eq!(
            ChipClass::from_family(FAMILY_GC_11_5_0, 0xC0),
            ChipClass::Gfx11_5
        );
        assert_eq!(
            ChipClass::from_family(FAMILY_GC_12_0_0, 0x50),
            ChipClass::Gfx12
        );
        // NV split: Navi14 (GFX10) vs Navi21 (GFX10_3).
        assert_eq!(ChipClass::from_family(FAMILY_NV, 0x14), ChipClass::Gfx10);
        assert_eq!(ChipClass::from_family(FAMILY_NV, 0x28), ChipClass::Gfx10_3);
        assert_eq!(ChipClass::from_family(999, 0), ChipClass::Unknown);
    }
}
