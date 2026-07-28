// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lossless BF16 exponent compression: 3-bit LUT + escape (`BF16L3`).
//!
//! BF16 spends 8 bits on an exponent whose realised distribution in a trained
//! weight tensor is narrow. Coding it as a 3-bit index into a per-block LUT
//! (index 7 = escape → literal byte) takes the per-element cost from 16 bits to
//! ~11.6, a measured **1.38×** on weights and vision towers.
//!
//! The transform is **exactly lossless**: every input `u16` is reproduced
//! bit-for-bit, including zeros, denormals, infinities and NaN payloads.
//!
//! # Layout — planar, so the GPU can decode it in place
//!
//! This format is decoded *in the kernel*, so weights stay compressed in VRAM
//! and the ratio applies to bandwidth, not just file size. That rules out both
//! a tensor-wide escape stream (decodable only front-to-back) and variable-length
//! blocks laid out back-to-back (block starts land on arbitrary alignments).
//!
//! Instead every fixed-size plane is stored contiguously at a computed, 16-byte
//! aligned base, and only the genuinely variable escape plane is indexed:
//!
//! ```text
//! [16 B] header    — u32 n_blocks, u32 n_elems, u32 escape-plane length, u32 pad
//! [     ] esc_tab  — u32 × n_blocks: block's start index into the escape plane
//! [     ] lut      —  8 B × n_blocks: exponent for codes 0..=6; byte 7 padding
//! [     ] mant     — 256 B × n_blocks: sign << 7 | mantissa[6:0]
//! [     ] codes    —  96 B × n_blocks: 3-bit codes, 8 per 3 bytes, LSB-first
//! [     ] escapes  — literal exponents, grouped by block, in element order
//! ```
//!
//! The mantissa and code planes are padded to full blocks so element `i` of
//! block `b` is at `mant[256*b + i]` and its code at `codes[96*b + ...]` — no
//! indirection and no dependent load. Only `esc_tab[b]` is an indexed read, one
//! per block, issuable early. Cost is 364 B per 256 elements (12 B of it LUT +
//! escape-table overhead), giving a **1.4066× ceiling** at zero escapes.
//!
//! # What a kernel does
//!
//! In a wave32, thread `t` owns elements `[8t, 8t+8)` of a block — exactly the
//! 3 code bytes at `codes[96*b + 3t]`, which is why the packing is 8-per-3-bytes.
//! Each thread counts its own escapes, one wave-wide exclusive prefix sum turns
//! those into per-thread escape-plane cursors, and the block decodes with no
//! per-element search. See `kernels/src/gemv_bf16l3.hip`.
//!
//! # Where it does NOT pay
//!
//! Measured 1.075× on a MedGemma-27B Hessian (`.calib.hfq` qt=130 bf16 tril):
//! `XᵀX` accumulations span far more octaves than weights, so nearly every
//! element escapes. Do not apply this to Hessians without re-measuring;
//! [`encode_if_smaller`] is the backstop.

/// Elements per independently-decodable block. 256 matches the G256 grouping
/// every other rotated/quantized format in the tree already uses.
pub const BLOCK: usize = 256;

/// The code reserved to mean "the exponent follows as a literal byte".
const ESCAPE: u8 = 7;
/// LUT entries usable for direct coding (code 7 is [`ESCAPE`]).
const LUT_LEN: usize = 7;
/// Per-block LUT plane stride; entry 7 is padding to keep the stride a power of 2.
const LUT_STRIDE: usize = 8;
/// Per-block code plane stride: 256 codes × 3 bits.
const CODE_STRIDE: usize = BLOCK * 3 / 8;
/// Fixed header: n_blocks, n_elems, escape-plane length, padding.
const HEADER: usize = 16;

#[inline]
const fn align16(x: usize) -> usize {
    x.div_ceil(16) * 16
}

/// Byte offsets of each plane, derived purely from the block count. The GPU
/// kernel recomputes these identically from the same two numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub n_blocks: usize,
    /// u32 per block: start index into the escape plane.
    pub esc_tab: usize,
    /// 8 B per block.
    pub lut: usize,
    /// 256 B per block (padded past the tail).
    pub mant: usize,
    /// 96 B per block (padded past the tail).
    pub codes: usize,
    /// Variable-length escape literals.
    pub escapes: usize,
}

impl Layout {
    /// Plane offsets for a tensor of `n` elements.
    pub const fn for_elems(n: usize) -> Self {
        let nb = n.div_ceil(BLOCK);
        let esc_tab = HEADER;
        let lut = esc_tab + align16(4 * nb);
        let mant = lut + align16(LUT_STRIDE * nb);
        let codes = mant + BLOCK * nb;
        let escapes = codes + CODE_STRIDE * nb;
        Self {
            n_blocks: nb,
            esc_tab,
            lut,
            mant,
            codes,
            escapes,
        }
    }

    /// Total packed length given the number of escaped elements.
    pub const fn packed_len(&self, n_esc: usize) -> usize {
        self.escapes + n_esc
    }
}

/// Choose a block's LUT: the `LUT_LEN` most frequent exponents, descending by
/// population. Ties break toward the smaller exponent, so the encoding is
/// deterministic across runs and platforms.
fn choose_lut(hist: &[u64; 256]) -> [u8; LUT_STRIDE] {
    let mut order: Vec<u8> = (0..=255u8).collect();
    order.sort_by_key(|&e| (std::cmp::Reverse(hist[e as usize]), e));
    let mut lut = [0u8; LUT_STRIDE];
    lut[..LUT_LEN].copy_from_slice(&order[..LUT_LEN]);
    lut
}

/// Compress raw little-endian BF16 bytes to the planar `BF16L3` layout.
///
/// A trailing odd byte is ignored. The result is smaller than the input for
/// weight-shaped data but **not** guaranteed so in general — an input using
/// many exponents uniformly costs 19 bits/element. Use [`encode_if_smaller`]
/// where that matters.
pub fn encode(bf16_le: &[u8]) -> Vec<u8> {
    let n = bf16_le.len() / 2;
    let lay = Layout::for_elems(n);
    let at = |i: usize| u16::from_le_bytes([bf16_le[2 * i], bf16_le[2 * i + 1]]);

    let mut out = vec![0u8; lay.escapes];
    let mut escapes: Vec<u8> = Vec::new();

    for blk in 0..lay.n_blocks {
        let start = blk * BLOCK;
        let len = BLOCK.min(n - start);

        let mut hist = [0u64; 256];
        for i in start..start + len {
            hist[((at(i) >> 7) as u8) as usize] += 1;
        }
        let lut = choose_lut(&hist);
        let mut code_of = [ESCAPE; 256];
        for (code, &e) in lut[..LUT_LEN].iter().enumerate() {
            code_of[e as usize] = code as u8;
        }

        // This block's escapes begin here in the escape plane.
        let esc_start = escapes.len() as u32;
        out[lay.esc_tab + 4 * blk..lay.esc_tab + 4 * blk + 4]
            .copy_from_slice(&esc_start.to_le_bytes());
        out[lay.lut + LUT_STRIDE * blk..lay.lut + LUT_STRIDE * blk + LUT_STRIDE]
            .copy_from_slice(&lut);

        let mant = lay.mant + BLOCK * blk;
        let codes = lay.codes + CODE_STRIDE * blk;
        for i in 0..len {
            let bits = at(start + i);
            out[mant + i] = (((bits >> 15) as u8) << 7) | (bits as u8 & 0x7f);
        }
        // 8 codes per 24-bit little-endian group — one group per thread lane.
        for grp in 0..len.div_ceil(8) {
            let mut acc = 0u32;
            for slot in 0..8 {
                let i = grp * 8 + slot;
                if i >= len {
                    break;
                }
                let e = (at(start + i) >> 7) as u8;
                let code = code_of[e as usize];
                if code == ESCAPE {
                    escapes.push(e);
                }
                acc |= (code as u32) << (3 * slot);
            }
            out[codes + 3 * grp..codes + 3 * grp + 3].copy_from_slice(&acc.to_le_bytes()[..3]);
        }
    }

    out[..4].copy_from_slice(&(lay.n_blocks as u32).to_le_bytes());
    out[4..8].copy_from_slice(&(n as u32).to_le_bytes());
    out[8..12].copy_from_slice(&(escapes.len() as u32).to_le_bytes());
    out.extend_from_slice(&escapes);
    out
}

/// [`encode`], but `None` when the packed form is not smaller than the plain
/// BF16 input — the caller should then store plain BF16 (`QuantType::BF16`).
pub fn encode_if_smaller(bf16_le: &[u8]) -> Option<Vec<u8>> {
    let packed = encode(bf16_le);
    (packed.len() < bf16_le.len()).then_some(packed)
}

/// Element count recorded in the header.
pub fn n_elems(packed: &[u8]) -> Option<usize> {
    Some(u32::from_le_bytes(packed.get(4..8)?.try_into().ok()?) as usize)
}

/// Decode one block to BF16 bit patterns — the random-access primitive the GPU
/// kernel mirrors. Returns `None` if the payload is truncated or `blk` is out
/// of range.
pub fn decode_block(packed: &[u8], blk: usize, n: usize) -> Option<Vec<u16>> {
    let lay = Layout::for_elems(n);
    if blk >= lay.n_blocks {
        return None;
    }
    let start = blk * BLOCK;
    let len = BLOCK.min(n - start);

    let esc_start = u32::from_le_bytes(
        packed
            .get(lay.esc_tab + 4 * blk..lay.esc_tab + 4 * blk + 4)?
            .try_into()
            .ok()?,
    ) as usize;
    let lut = packed.get(lay.lut + LUT_STRIDE * blk..lay.lut + LUT_STRIDE * blk + LUT_STRIDE)?;
    let mant = packed.get(lay.mant + BLOCK * blk..lay.mant + BLOCK * blk + len)?;
    let codes =
        packed.get(lay.codes + CODE_STRIDE * blk..lay.codes + CODE_STRIDE * blk + CODE_STRIDE)?;
    let escapes = packed.get(lay.escapes + esc_start..)?;

    let mut out = Vec::with_capacity(len);
    let mut esc = escapes.iter();
    for i in 0..len {
        let g = &codes[3 * (i / 8)..3 * (i / 8) + 3];
        let acc = u32::from_le_bytes([g[0], g[1], g[2], 0]);
        let code = ((acc >> (3 * (i % 8))) & 7) as u8;
        let e = if code == ESCAPE {
            *esc.next()?
        } else {
            lut[code as usize]
        };
        let sm = mant[i];
        out.push(((sm as u16 & 0x80) << 8) | ((e as u16) << 7) | (sm as u16 & 0x7f));
    }
    Some(out)
}

/// Decompress a whole `BF16L3` payload back to raw little-endian BF16 bytes.
/// `n` is the element count from the tensor shape.
pub fn decode(packed: &[u8], n: usize) -> Option<Vec<u8>> {
    if n_elems(packed)? != n {
        return None;
    }
    let mut out = Vec::with_capacity(n * 2);
    for blk in 0..Layout::for_elems(n).n_blocks {
        for bits in decode_block(packed, blk, n)? {
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift — a fixed corpus beats a rand dependency in a
    /// zero-dependency leaf crate.
    fn xorshift(seed: &mut u32) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed
    }

    fn roundtrips(bits: &[u16]) -> Vec<u8> {
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let packed = encode(&raw);
        assert_eq!(
            decode(&packed, bits.len()).as_deref(),
            Some(raw.as_slice()),
            "lossless roundtrip"
        );
        packed
    }

    #[test]
    fn roundtrip_is_bit_exact_over_every_u16() {
        // The strongest statement available: all 65536 BF16 bit patterns,
        // covering zeros, denormals, infinities and NaN payloads.
        roundtrips(&(0..=u16::MAX).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_handles_ragged_tails_and_empty() {
        // Exercise partial code groups (n % 8) and partial blocks (n % BLOCK).
        for n in [0usize, 1, 5, 7, 8, 9, 17, 255, 256, 257, 700, 1024] {
            let bits: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(2654)).collect();
            roundtrips(&bits);
        }
    }

    #[test]
    fn planes_are_16_byte_aligned_and_stride_addressed() {
        // The kernel indexes mant/codes as blk*stride with no indirection, so
        // these bases must stay aligned and the strides exact.
        for n in [256usize, 1024, 8192, 1_000_000] {
            let lay = Layout::for_elems(n);
            assert_eq!(lay.mant % 16, 0, "mant base for n={n}");
            assert_eq!(lay.codes % 16, 0, "codes base for n={n}");
            assert_eq!(lay.lut % 16, 0, "lut base for n={n}");
            assert_eq!(lay.codes - lay.mant, BLOCK * lay.n_blocks);
            assert_eq!(lay.escapes - lay.codes, CODE_STRIDE * lay.n_blocks);
        }
    }

    #[test]
    fn blocks_decode_independently_and_out_of_order() {
        // The point of the layout: block b is reachable without touching b-1.
        let mut seed = 0xfeed_1234u32;
        let bits: Vec<u16> = (0..2_000).map(|_| xorshift(&mut seed) as u16).collect();
        let packed = roundtrips(&bits);
        let n = bits.len();
        for blk in (0..Layout::for_elems(n).n_blocks).rev() {
            let got = decode_block(&packed, blk, n).expect("block decodes");
            let start = blk * BLOCK;
            assert_eq!(got, &bits[start..start + got.len()], "block {blk}");
        }
        assert!(decode_block(&packed, Layout::for_elems(n).n_blocks, n).is_none());
    }

    #[test]
    fn gaussian_weights_compress_about_1_38x() {
        // Sum of 4 uniforms ≈ Gaussian: the realistic weight-tensor case the
        // ratio claim rests on. Guards the LUT choice, not just the codec.
        let mut seed = 0x1234_5678u32;
        let bits: Vec<u16> = (0..64_000)
            .map(|_| {
                let s: i64 = (0..4).map(|_| (xorshift(&mut seed) % 2048) as i64).sum();
                super::super::conv::f32_to_bf16_bits((s - 4094) as f32 * 1e-4)
            })
            .collect();
        let packed = roundtrips(&bits);
        let ratio = (bits.len() * 2) as f64 / packed.len() as f64;
        assert!(
            ratio > 1.33,
            "expected ~1.38x on Gaussian weights, got {ratio:.3}x"
        );
    }

    #[test]
    fn single_exponent_tensor_hits_the_ceiling() {
        // One exponent everywhere: no escapes, so only the fixed planes remain.
        // 512 B raw per block vs 364 B stored = 1.4066x, the layout's maximum.
        let n = 8_192;
        let bits: Vec<u16> = (0..n).map(|i| 0x3f80 | (i as u16 & 0x7f)).collect();
        let packed = roundtrips(&bits);
        assert_eq!(packed.len(), Layout::for_elems(n).packed_len(0));
        let ratio = (n * 2) as f64 / packed.len() as f64;
        assert!(
            (1.40..1.408).contains(&ratio),
            "expected the 1.4066x ceiling, got {ratio:.4}x"
        );
    }

    #[test]
    fn adversarial_input_is_rejected_rather_than_inflated() {
        // 256 exponents cycled uniformly: >7 per block means a ~97% escape
        // rate, i.e. 19 bits/element. encode_if_smaller must decline.
        let bits: Vec<u16> = (0..8_192u32).map(|i| ((i % 256) as u16) << 7).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        assert!(encode_if_smaller(&raw).is_none());
        roundtrips(&bits); // still lossless, just not smaller
    }

    #[test]
    fn truncated_payload_decodes_to_none() {
        let n = 1_000;
        let bits: Vec<u16> = (0..n).map(|i| 0x3f80 | (i as u16 & 0x7f)).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let packed = encode(&raw);
        for cut in [0usize, 2, 8, 40, packed.len() / 2, packed.len() - 1] {
            assert!(decode(&packed[..cut], n).is_none(), "cut at {cut}");
        }
        // A short escape plane is only catchable mid-decode: force an escape in
        // the final block, then truncate exactly that literal away.
        let mut odd = bits.clone();
        odd[n - 1] = 0x7f80; // an exponent no other element uses
        let raw: Vec<u8> = odd.iter().flat_map(|b| b.to_le_bytes()).collect();
        let packed = encode(&raw);
        assert!(decode(&packed[..packed.len() - 1], n).is_none());
    }

    #[test]
    fn wrong_element_count_is_rejected() {
        // A shape/payload mismatch must not silently decode a short tensor.
        let bits: Vec<u16> = (0..600).map(|i| 0x3f80 | (i as u16 & 0x7f)).collect();
        let packed = roundtrips(&bits);
        assert!(decode(&packed, 600 + BLOCK).is_none());
        assert!(decode(&packed, 100).is_none());
    }
}
