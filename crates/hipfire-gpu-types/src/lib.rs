// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Leaf GPU value types shared across the workspace.
//!
//! [`GpuTensor`] (a device buffer + shape + [`DType`]) and the [`DType`]
//! element-type enum. Extracted from `hipfire-rdna` (review 3.11b) so crates
//! that only need to *name* a device tensor or element type depend on this
//! ~200-line leaf (over `hip-bridge`) instead of the whole 63k-LOC compute
//! crate. `hipfire-rdna` re-exports both, so `hipfire_rdna::{GpuTensor, DType}`
//! keeps resolving unchanged.

use hip_bridge::DeviceBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// BF16L3: lossless 3-bit-LUT exponent coding, ~11.6 bits/element, decoded
    /// in-kernel so the ratio applies to bandwidth. Byte length is NOT a stride
    /// — see `hipfire_primitives::bf16_lut3` for the plane layout.
    Bf16L3,
    Q4K,       // 144 bytes per 256 elements
    Q6K,       // 210 bytes per 256 elements
    Q8_0,      // 34 bytes per 32 elements
    Q4F16G64,  // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32,  // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,     // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256,  // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128,  // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256,  // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128,  // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,   // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    MQ4G128,   // MagnumQuant: FWHT-128-rotated INT4 (72 bytes/group, same layout as HFQ4G128)
    MQ8G256,   // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    MQ6G256,   // MagnumQuant: FWHT-rotated HFQ6-G256 (200 bytes/group, same as HFQ6G256)
    MQ3G256,   // MagnumQuant: FWHT-rotated HFQ3-G256 (104 bytes/group, same as HFQ3G256)
    Qtip3G256, // QTIP-3: FWHT-rotated trellis-coded 3-bit (100 bytes/group: f32 scale + 96 B
    // packed symbols). Decoded by gemv_qtip3g256 (computed 1MAD codebook, zero LDS); runtime
    // FWHT-rotates x like MQ3/MQ4. See kernels/src/gemv_qtip3g256.hip / qtip.rs.
    Qtip3G256I3, // QTIP-3 with the 3INST codebook (100 bytes/group, identical layout to
    // Qtip3G256). Only the computed state->value map differs: 3INST (excess kurtosis -0.111)
    // vs 1MAD (-0.312), i.e. closer to the Gaussian the rotated weights follow. A separate
    // DType because the codebook is part of the wire format -- decoding one as the other
    // produces noise while every structural check passes. See gemv_qtip3g256i3.hip.
    Qtip4G256, // QTIP-4: FWHT-rotated trellis-coded 4-bit (132 bytes/group: f32 scale + 128 B
    // nibble-packed symbols). Same 1MAD codebook/12-bit trellis as Qtip3G256, decoded by
    // gemv_qtip4g256. See kernels/src/gemv_qtip4g256.hip / qtip::pack_qtip4_group.
    MQ2G256,      // MagnumQuant: FWHT-rotated HFQ2-G256 (72 bytes/group, same as HFQ2G256)
    MQ2G256Lloyd, // MagnumQuant 2-bit + Lloyd-Max 4-entry fp16 codebook (72 bytes/group)
    MQ3G256Lloyd, // MagnumQuant 3-bit + Lloyd-Max 8-entry fp16 codebook (112 bytes/group)
    MQ4G256Lloyd, // MagnumQuant 4-bit + Lloyd-Max 16-entry fp16 codebook (160 bytes/group)
    HFP4G32,      // HFP4: E2M1 element + UE8M0 g32 block scale + FP16 row scale.
    // Per-row header 16 B; per-block payload 17 B (UE8M0 + 16 packed nibbles).
    // See docs/quant-formats/hfp4.md.
    MFP4G32, // MFP4: HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same byte layout
    // as HFP4G32; format_flags bit 0 + bits 2-3 = 01 stamps the rotation kind.
    // Runtime applies the matching FWHT to x via mq_rotate_x; the kernel itself
    // is shared with HFP4G32.
    HFQ2G256,   // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128,   // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    HFQ6G256,   // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    ParoQ4G128, // ParoQuant: AWQ-packed INT4 G128 repacked to HFQ4G128 layout at load.
    // Weights are standard HFQ4G128 (72 bytes/group); the ParoQuant distinction
    // is that weight_gemv applies Givens rotation to activations before GEMV.
    // Rotation metadata (pairs, theta, channel_scales) lives on WeightTensor::paro.
    Oq4G256, // Opus Quant W4A4: symmetric signed-INT4, FWHT-rotated, per-group f32 scale.
    Oq8G256, // Opus Quant W8A8: symmetric signed-INT8, FWHT-rotated, per-group f32 scale (iu8 WMMA).
    // Same device layout as Oq8G256 ([int8 m*k | f32 scales m*ng]) but ng = k/128
    // and the activation takes the FWHT-128 basis. Distinct DType because the
    // group is NOT recoverable from the buffer — only from the type.
    Oq8G128,
    // Opus Quant W8A8, compact-resident: the on-disk OqPlusCompact (qt=36) blocks
    // kept as-is on device instead of expanded to one int8 per weight at load.
    //   [f16 scale | 128 packed signed-int4 | N_out * (u8 idx, i8 val)]
    //   block_stride = 130 + 2*N_out, one block per (row, 256-group)
    // FWHT-rotated like Oq8G256 and consumed by the same W8A8 math — the kernel
    // decodes the nibbles and applies the overlay per tile — but the byte layout
    // is block-structured with an in-block f16 scale rather than a flat int8
    // plane plus a split f32 scale plane, so it must be a distinct dtype for the
    // same reason the DflashOq*Plain variants are: a loader or GEMM that treated
    // it as Oq8G256 would read 4-bit data as 8-bit and silently produce garbage.
    // `block_stride` is not carried here — it is derivable as
    // `bytes / (M * K/256)` at the call site.
    OqCompactG256,
    // Same block structure as OqCompactG256 at a 128-element group: header is
    // `2 + 128/2` = 66 bytes rather than 130, and `block_stride` derives as
    // `bytes / (M * K/128)`. Distinct dtype rather than a parameter because the
    // group also selects the FWHT length (128-point, sign seeds 43/1043, the
    // one MQ4G128 already uses) — reading it as G256 would rotate by the wrong
    // transform. It exists for COVERAGE of models whose K is a multiple of 128
    // but not 256, not to beat G=256 on size; see
    // docs/experiments/2026-08-06-oq-compact-group-size.md.
    OqCompactG128,
    // DFLASH plain-basis Opus storage. These variants preserve the arch-20
    // sidecar's original interleaved blocks on device:
    //   Oq8Plain        [f16 scale | 256 i8]                       (258 B)
    //   Oq4Plain        [f16 scale | 128 packed signed nibbles]    (130 B)
    //   Oq4MixedPlain   Oq4Plain + (u8 position, i8 replacement)*N
    // They are deliberately distinct from Oq{4,8}G256: those primary-model
    // formats are FWHT-rotated and use split f32 scales. Treating these as the
    // same dtype would silently rotate DFLASH activations into the wrong basis.
    DflashOq8Plain,
    DflashOq4Plain,
    DflashOq4MixedPlain,
    // On-disk storage is [f16 scale][128 nibbles]/256-group (130 B/group, codec
    // `quantize_oq4g256`). The loader repacks to the kernel layout: packed nibbles
    // [M,K/2] followed by per-group f32 scales [M,K/256] in one buffer. The forward
    // quantizes activations to int4 at runtime (`quantize_act_oq4`) and dispatches
    // `gemm_oq4_grouped_wmma` — the only W4A4 (int4-activation) path in the engine.
    W8A8Ref, // Reference kernel layer W8A8: per-channel symmetric int8 weights followed by
    // per-channel f32 scales in one buffer ([M*K int8 | M f32]). A8 activations are
    // quantized per-token at runtime; iu8 WMMA + dequant by w_scale·x_scale. Boring
    // reference (no grouping/rotation) — produced by quantize-on-load (HIPFIRE_W8A8=1).
    Raw, // raw bytes, no element interpretation
}

/// Byte stride of one `Q8HFQ` weight row: `[f16 scale x K/32 | int8 value x K]`
/// padded up to a 128-byte boundary.
///
/// The single definition of that layout. `Q8HFQ` is the ONLY dtype whose GEMV
/// kernel indexes with a caller-supplied row stride (`row_base = A + row *
/// row_stride`, `kernels/src/gemv_q8hfq*.hip`), so a caller that guesses the
/// value — or leaves it 0, the default a hand-rolled `WeightRef` literal
/// carries — makes every output row dot weight row 0 and still returns success.
/// The loaders and the dispatch guard both derive it here so they cannot drift.
pub const fn q8hfq_row_stride(k: usize) -> usize {
    ((k / 32) * 2 + k + 127) & !127
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4K
            | DType::Q6K
            | DType::Q8_0
            | DType::Q4F16G64
            | DType::Q4F16G32
            | DType::Q8HFQ
            | DType::HFQ4G256
            | DType::HFQ4G128
            | DType::HFQ3G256
            | DType::HFQ3G128
            | DType::HFQ2G256
            | DType::HFQ2G128
            | DType::HFQ6G256
            | DType::MQ4G256
            | DType::MQ4G128
            | DType::MQ6G256
            | DType::MQ8G256
            | DType::MQ3G256
            | DType::Qtip3G256
            | DType::Qtip3G256I3
            | DType::Qtip4G256
            | DType::MQ2G256
            | DType::MQ2G256Lloyd
            | DType::MQ3G256Lloyd
            | DType::MQ4G256Lloyd
            | DType::HFP4G32
            | DType::MFP4G32
            | DType::ParoQ4G128
            | DType::Oq4G256
            | DType::DflashOq8Plain
            | DType::DflashOq4Plain
            | DType::DflashOq4MixedPlain
            | DType::W8A8Ref
            | DType::Oq8G256
            | DType::Oq8G128
            // OqCompactG256 is block-structured with a variable-width overlay
            // table, so like Bf16L3 it has no per-element stride: a length is
            // `M * (K/256) * (130 + 2*N_out)`, never `n * size()`.
            | DType::OqCompactG256
            | DType::OqCompactG128
            // BF16L3's payload is planar with a variable escape plane, so
            // there is no per-element stride at all — byte-level, like Raw.
            // Anything computing a length as `n * size()` is wrong for it and
            // must use the format's own layout.
            | DType::Bf16L3
            | DType::Raw => 1, // byte-level
        }
    }

    /// Whether a `WeightTensor` of this dtype should have the
    /// `<weight>.awq_scale.weight` F16 sidecar attached at load time.
    ///
    /// Centralizes the gate that previously lived inline at every
    /// loader call site (qwen35.rs `load_weight_tensor`, etc.). The
    /// motivation is the May 2026 regression where `qwen35.rs:907`
    /// gated on `matches!(wt.gpu_dtype, DType::MQ4G256)` and silently
    /// dropped AWQ sidecars for `MQ3G256`-quantized Qwen3.5 weights,
    /// producing fluent-but-nonsensical token soup for ~5 hours
    /// before the missing arm was traced. Adding a new AWQ-eligible
    /// dtype is now a one-line edit here instead of two scattered
    /// edits per loader.
    ///
    /// Current allow-list = the empirical truth of which dtypes ship
    /// AWQ sidecars from the quantizer AND have an AWQ-aware forward
    /// path (`rotate_x_mq_for` etc., wired through `awq_scale.is_some()`).
    ///
    /// **Forward-path-ready candidates not currently in the allow-list**
    /// (forward kernels exist but no `.hfq` file in tree ships sidecars
    /// for them — widen only after the quantizer side is verified to
    /// emit sidecars and at least one coherence-gate row exercises the
    /// combination):
    /// - `MQ6G256`
    /// - `MQ2G256`, `MQ2G256Lloyd`
    /// - `MQ3G256Lloyd`
    /// - `MFP4G32` (forward path has explicit `awq_scale.is_some()`
    ///   branching at llama.rs:609 but the quantizer comment says
    ///   "AWQ is gated to MQ4G256 today" — confirm before widening)
    ///
    /// `MQ8G256` is explicitly **not** a candidate: it uses its own
    /// INT8-quantized scratch path (`gemv_mq8g256_with_rotate`,
    /// `rotate_quantize_x_mq8`) and does not flow through
    /// `rotate_x_mq_for`, so there is no AWQ-aware kernel to dispatch
    /// to.
    ///
    /// **lm_head / embed_tokens callers:** as of the lm_head-AWQ
    /// runtime PR, this helper IS safe for the `output` weight in
    /// `qwen35.rs::load_weights` / `load_weights_vl`. Both dispatch
    /// paths that consume `weights.output` now route through
    /// AWQ-aware rotations when a sidecar is attached:
    /// - Decode: `weight_gemv` → `rotate_x_mq_for` (llama.rs)
    /// - Spec-decode verify: `speculative.rs::rotate_x_mq_batched_for`
    ///
    /// Pre-runtime-fix, attaching a sidecar on lm_head would have
    /// produced `(W·s)·x ≠ W·x` via the spec-verify path's plain
    /// `rotate_x_mq_batched` and driven the KLD 0.67 → 13.5
    /// corruption documented at `docs/plans/awq_fix_claude.md`. The
    /// quantizer-side `awq_eligible` whitelist
    /// (`hipfire-quantize/src/main.rs:3849`) still gates which
    /// tensors actually receive `W' = W·s` pre-multiplication at
    /// quant time — this helper governs only whether the loader
    /// attaches an already-emitted sidecar.
    pub fn supports_awq_sidecar(self) -> bool {
        // MQ3G256Lloyd / MQ2G256Lloyd added 2026-05-28: they are "forward-path-ready"
        // (flow through rotate_x_mq_for, which applies x/=awq_scale when a sidecar is
        // attached) — see the doc block above. Enables AWQ×Lloyd composition once the
        // quantizer emits sidecars for the Lloyd arms.
        matches!(
            self,
            DType::MQ4G256
                | DType::MQ3G256
                | DType::MQ2G256
                | DType::MQ3G256Lloyd
                | DType::MQ2G256Lloyd
                // Opus Quant W4A4: SmoothQuant migrates a per-channel scale into the
                // weight offline (W·s); the forward divides x by the awq_scale
                // sidecar before FWHT+int4-quantize, completing (W·s)·(x/s) = W·x.
                | DType::Oq4G256
                // OQ+ / Opus Plus W4A8 loads as Oq8G256 (int4 weights upcast to
                // int8); same AWQ contract — the forward divides x by the sidecar
                // before FWHT+int8-quantize. (Real W8A8 Oq8 has no sidecar → no-op.)
                | DType::Oq8G256
                | DType::Oq8G128
                // QTIP-3 trellis: the pack applies W*s before the FWHT exactly as
                // the Opus arms do, so the forward must divide x by the sidecar to
                // complete (W*s).(x/s) = W.x. Without this arm the quantizer would
                // emit awq_scale sidecars that nothing attaches — weights smoothed
                // on one side only, which is worse than no AWQ at all.
                | DType::Qtip3G256
                // Same contract as Qtip3G256 — the 3INST codebook changes the
                // decoded VALUES, not whether the weights were AWQ-smoothed.
                // Omitting this arm left the sidecar unattached on qtip3+ I3
                // artifacts: W*s served against un-divided x, which scored KLD
                // 8.27 while the non-AWQ path scored 1.61.
                | DType::Qtip3G256I3
                // Compact-RESIDENT Opus (HIPFIRE_OQ_COMPACT_RESIDENT): identical
                // AWQ contract to Oq8G256 above — it is the same artifact, just
                // left in 4.25 b/w blocks instead of expanded to int8 at load.
                // Omitting these two is why whole-model compact residency served
                // garbage: expanding hit the Oq8G256 arm and got x/s, staying
                // compact silently dropped it and served W*s against undivided
                // x. Same failure the Qtip3G256I3 arm above records.
                | DType::OqCompactG256
                | DType::OqCompactG128
        )
    }
}

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// A `GpuTensor` whose buffer is a null pointer of size 0, for CPU-only unit
    /// tests in **dependent crates** that read only tensor metadata (shape/dtype/op)
    /// and never touch the device.
    ///
    /// CONTRACT: the returned tensor must NEVER be passed to a HIP call — its buffer
    /// is null and dereferencing it on the GPU is undefined behavior. It exists only
    /// so cross-crate tests can borrow a `&GpuTensor` for metadata-only logic.
    ///
    /// Not `#[cfg(test)]`-gated on purpose: `#[cfg(test)]` here would only be active
    /// when `hipfire-rdna`'s own tests build, making this invisible to dependent
    /// crates' tests (e.g. `hipfire-dispatch`). `#[doc(hidden)]` keeps it out of the
    /// public API surface while remaining reachable cross-crate, matching the
    /// `FeatureFlags::from_env_for_test` precedent.
    #[doc(hidden)]
    pub fn null_for_test() -> Self {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: DType::F32,
        }
    }

    /// Create a non-owning sub-view at a byte offset. For F32 tensors,
    /// `offset_elems` is the number of f32 elements to skip.
    /// The returned tensor is a view — do NOT free it.
    pub fn sub_offset(&self, offset_elems: usize, len_elems: usize) -> GpuTensor {
        let byte_off = offset_elems * self.dtype.size();
        let ptr = unsafe { (self.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void };
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, len_elems * self.dtype.size()) },
            shape: vec![len_elems],
            dtype: self.dtype,
        }
    }
}

#[cfg(test)]
mod q8hfq_layout_tests {
    #[test]
    fn q8hfq_row_stride_pads_scales_plus_values_to_128b() {
        assert_eq!(super::q8hfq_row_stride(4096), 4352);
        assert_eq!(super::q8hfq_row_stride(896), 1024);
        for k in [256usize, 896, 1024, 4096, 8192] {
            let s = super::q8hfq_row_stride(k);
            assert_eq!(s % 128, 0, "K={k} stride {s} must be 128B-aligned");
            assert!(s >= (k / 32) * 2 + k, "K={k} stride {s} must hold the row");
        }
    }
}
