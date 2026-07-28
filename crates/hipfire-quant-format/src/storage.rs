// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lossless recodings: stored encoding vs the encoding a reader should see.
//!
//! Most [`QuantType`]s mean exactly what they say — a reader takes the bytes and
//! dispatches on the byte. A few are *lossless recodings* of another type: the
//! payload reconstructs some other encoding bit-for-bit, and every reader is
//! supposed to expand it and carry on as if it had read that other type.
//!
//! That "every reader" is the problem this module exists to solve. The `.hfq`
//! index is parsed independently in nine places across the tree (the runtime
//! loader, the quantizer's re-quantize input, the sidecar tools, …), because
//! `hipfire-runtime`'s parser pulls in the GPU stack and leaf crates cannot
//! reach it. When BF16L3/BF16H were added, the expansion was hand-written into
//! two of those readers and silently missing from the rest — re-quantizing a
//! compressed artifact would have failed. Homing the rule here means a reader
//! calls [`expand`] and is correct for every present and future recoding.
//!
//! Adding a recoding therefore means adding one arm to [`QuantType::logical`]
//! and one to [`expand`] — and `every_recoding_expands` fails the build's tests
//! until you do.

use crate::QuantType;
use std::borrow::Cow;

impl QuantType {
    /// The encoding a reader should present for this stored type.
    ///
    /// The identity for everything except a lossless recoding, which reports the
    /// type it reconstructs. An index entry should be rewritten to this before
    /// consumers see it, so nothing downstream needs a per-codec branch.
    pub const fn logical(self) -> Self {
        match self {
            Self::Bf16Lut3 | Self::Bf16Huff => Self::BF16,
            other => other,
        }
    }

    /// Whether the stored bytes must be decoded before use — i.e. whether
    /// `logical()` differs from the type itself.
    pub const fn is_lossless_recoding(self) -> bool {
        !matches!(self.logical(), s if s as u8 == self as u8)
    }

    /// Byte length of the *expanded* payload for `n_elems`, or `None` when the
    /// logical type has no fixed per-element width.
    pub fn logical_byte_len(self, n_elems: usize) -> Option<usize> {
        let logical = self.logical();
        match logical {
            QuantType::BF16 | QuantType::F16 => Some(n_elems * 2),
            QuantType::F32 => Some(n_elems * 4),
            _ => logical.tensor_bytes(n_elems),
        }
    }
}

/// Expand a stored payload to its logical bytes.
///
/// Returns the input borrowed for every ordinary type, and owned decoded bytes
/// for a lossless recoding. `n_elems` is the element count from the tensor
/// shape. `threads` parallelises a bit-serial decode across the format's chunk
/// table; 1 decodes inline.
///
/// `None` means the payload is corrupt or truncated — never "unknown type",
/// which is why callers can treat `None` as a hard error.
pub fn expand(
    stored: QuantType,
    raw: &[u8],
    n_elems: usize,
    threads: usize,
) -> Option<Cow<'_, [u8]>> {
    match stored {
        QuantType::Bf16Lut3 => hipfire_primitives::bf16_lut3::decode(raw, n_elems).map(Cow::Owned),
        QuantType::Bf16Huff => {
            hipfire_primitives::bf16_huff::decode_par(raw, n_elems, threads).map(Cow::Owned)
        }
        _ => Some(Cow::Borrowed(raw)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every stored byte that maps to a different logical type must round-trip
    /// through `expand`. This is the guard that makes a new recoding impossible
    /// to add without wiring its decode — the failure mode that motivated this
    /// module.
    #[test]
    fn every_recoding_expands() {
        let src: Vec<u8> = (0..4096u32)
            .flat_map(|i| ((0x3f80u16) | (i as u16 & 0x7f)).to_le_bytes())
            .collect();
        let n = src.len() / 2;

        let mut seen = 0;
        for code in 0u8..=255 {
            let Some(qt) = QuantType::from_code(code) else {
                continue;
            };
            if !qt.is_lossless_recoding() {
                continue;
            }
            seen += 1;
            // Known recodings are all BF16 today; encode with the matching codec.
            let packed = match qt {
                QuantType::Bf16Lut3 => hipfire_primitives::bf16_lut3::encode(&src),
                QuantType::Bf16Huff => hipfire_primitives::bf16_huff::encode(&src),
                other => panic!(
                    "{other:?} is a lossless recoding with no encode/expand wiring — \
                     add it to `expand()` (and to this test)"
                ),
            };
            for threads in [1usize, 4] {
                assert_eq!(
                    expand(qt, &packed, n, threads).as_deref(),
                    Some(src.as_slice()),
                    "{qt:?} did not expand losslessly (threads={threads})"
                );
            }
            assert_eq!(qt.logical(), QuantType::BF16);
            assert_eq!(qt.logical_byte_len(n), Some(src.len()));
        }
        assert!(seen >= 2, "expected the BF16 recodings to be registered");
    }

    #[test]
    fn ordinary_types_pass_through_borrowed() {
        let raw = vec![7u8; 64];
        for qt in [
            QuantType::BF16,
            QuantType::F16,
            QuantType::F32,
            QuantType::Oq4G256,
        ] {
            assert!(!qt.is_lossless_recoding(), "{qt:?}");
            assert_eq!(qt.logical(), qt);
            let out = expand(qt, &raw, 32, 1).expect("passthrough");
            assert!(matches!(out, Cow::Borrowed(_)), "{qt:?} should not copy");
            assert_eq!(&*out, raw.as_slice());
        }
    }

    #[test]
    fn corrupt_payload_is_an_error_not_a_silent_pass() {
        let src: Vec<u8> = (0..2048u32)
            .flat_map(|i| (0x3f80u16 | (i as u16 & 0x7f)).to_le_bytes())
            .collect();
        let n = src.len() / 2;
        for qt in [QuantType::Bf16Lut3, QuantType::Bf16Huff] {
            let packed = match qt {
                QuantType::Bf16Lut3 => hipfire_primitives::bf16_lut3::encode(&src),
                _ => hipfire_primitives::bf16_huff::encode(&src),
            };
            assert!(
                expand(qt, &packed[..packed.len() / 2], n, 1).is_none(),
                "{qt:?}"
            );
        }
    }
}
