// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lossless BF16 exponent compression by Huffman coding (`BF16H`).
//!
//! The sibling of [`crate::bf16_lut3`], trading that format's fixed 3-bit code
//! for a variable-length Huffman code over the exponent. Same core idea — the
//! sign and mantissa are left as a raw byte and only the exponent is coded —
//! but a Huffman code spends ~2.66 bits on an exponent whose entropy is 2.60,
//! where the fixed 3-bit code plus escapes and per-block tables spends 3.57.
//!
//! Measured on Qwen3-1.7B: **1.50×** vs 1.38× for `bf16_lut3`, against an
//! order-0 entropy floor of 1.52×. This is the format to write to **disk**.
//!
//! The trade is decode cost. Variable-length codes are bit-serial, so this is
//! decoded once on load and expanded to plain BF16 in RAM. When weights must
//! stay compressed *in VRAM* and be decoded inside a GEMV, use `bf16_lut3`:
//! its byte-aligned fixed-width code is what makes in-kernel decode affordable.
//!
//! # Alphabet: top-15 exponents + escape
//!
//! Coding all 256 possible exponents directly measures 1.4997× but produces
//! codes up to **26 bits** deep. Restricting the alphabet to the 15 most
//! frequent exponents plus an escape (which carries a raw 8-bit exponent)
//! measures 1.4995× — statistically identical — while capping the alphabet at
//! 16 symbols and observed code depth at 13. A shallow code is what allows a
//! small decode table, so the restriction is free compression-wise and pure
//! win decode-wise.
//!
//! # Layout
//!
//! ```text
//! [16 B] header    — u32 n_elems, u32 n_chunks, u32 bitstream_bytes,
//!                    u8 n_symbols, u8 max_code_len, u8 n_direct, u8 version
//! [16 B] symbols   — exponent value per symbol; the last is ESCAPE when present
//! [16 B] lengths   — Huffman code length per symbol (canonical code rebuild)
//! [     ] chunks   — BIT offset of each chunk in the bitstream: u32 × n_chunks
//!                    when version is 0, u64 × n_chunks when it is 1. The width
//!                    is chosen from the exact bit total, so it is 32-bit for
//!                    everything under 2^32 bits and only widens where a 32-bit
//!                    offset would wrap. See [`VERSION_OFF`].
//! [  n B] mant     — sign << 7 | mantissa[6:0], directly indexed by element
//! [     ] bits     — Huffman codes, MSB-first; an escape is followed by 8 raw
//!                    bits holding the literal exponent
//! ```
//!
//! # Why there is no 4-stream interleaving
//!
//! `huf_decompress.c` decodes 4 bitstreams at once because it resolves one
//! symbol per step, making the bit-cursor dependency chain the bottleneck. The
//! multi-symbol table above already breaks that chain — a lookup advances the
//! cursor once per ~4 symbols — so the two are substitutes, not complements.
//! Measured: interleaving 4 chunks in one thread is **-11%** on real weights
//! and **+23%** only on an escape-heavy corpus where the table constantly
//! misses. Note this would need no format change if ever wanted: the chunk
//! table already provides independent streams, which is what FSE must
//! manufacture.
//!
//! Chunking every [`CHUNK`] elements costs 4 B per 8192 weights (0.004 b/w,
//! i.e. nothing) and makes decode embarrassingly parallel: a chunk's codes start
//! at a known bit offset and its mantissa bytes are directly indexed. That
//! matters because a full BF16 artifact is tens of GB to expand at load.

/// Elements per independently-decodable chunk.
pub const CHUNK: usize = 8192;
/// 15 exponents + 1 escape.
const MAX_SYMBOLS: usize = 16;
/// Exponents coded directly; everything else escapes.
const DIRECT_SYMBOLS: usize = 15;
/// Bits of the primary decode table. Codes this long or shorter resolve in one
/// lookup; longer ones fall back to a canonical compare, which is rare because
/// short codes are by construction the frequent ones.
const PRIMARY_BITS: usize = 8;

const HEADER: usize = 16;
const SYMTAB: usize = HEADER;
const LENTAB: usize = SYMTAB + MAX_SYMBOLS;
const CHUNKTAB: usize = LENTAB + MAX_SYMBOLS;

/// Header byte holding the chunk-offset width version.
///
/// v0 stored chunk bit offsets as `u32`, which silently wraps once a tensor's
/// exponent bitstream passes 2^32 bits — about 1.65 G elements at the measured
/// ~2.61 bits/exponent, i.e. a ~3.3 GB BF16 tensor. Past the wrap every chunk
/// decodes from the wrong bit position, so the mantissa plane (directly
/// indexed) stays correct while the exponents come out plausible-but-wrong.
/// That failure is invisible to a magnitude check: the values look like
/// weights, just the wrong ones.
///
/// v1 widens the offsets to `u64`. The byte was zero-filled and unread in v0,
/// so a v0 artifact keeps decoding through the v0 arm below — which matters,
/// because artifacts written before this fix may outlive their sources.
const VERSION_OFF: usize = 15;
const V0: u8 = 0;
const V1: u8 = 1;

/// Chunk-offset width in bytes for a format version.
#[inline]
const fn off_width(version: u8) -> usize {
    if version == V0 {
        4
    } else {
        8
    }
}

/// Largest element count the `u32` `n_elems` header field can describe. A
/// tensor above this cannot be encoded at all (see [`encode_if_smaller`]),
/// rather than being written with a truncated count.
pub const MAX_ELEMS: usize = u32::MAX as usize;

/// Byte offset of the chunk table's end / mantissa plane start.
#[inline]
const fn mant_off_v(n_chunks: usize, version: u8) -> usize {
    CHUNKTAB + off_width(version) * n_chunks
}

// ── bit I/O, MSB-first ──────────────────────────────────────────────────────

/// Zero bytes appended after the bitstream so [`peek32`] can always load 8
/// bytes without a bounds test. A symbol needs at most 15 code bits + 8 literal
/// bits, so an 8-byte window always covers the next symbol.
const TAIL_PAD: usize = 8;

struct BitWriter {
    buf: Vec<u8>,
    /// Bits not yet flushed, held in the low `nbits` positions.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    /// Total bits written so far — the bit offset a chunk starts at.
    #[inline]
    fn bit_pos(&self) -> usize {
        self.buf.len() * 8 + self.nbits as usize
    }
    /// Append the low `len` bits of `code`, most significant first. `nbits`
    /// stays under 8 between calls and `len <= 23`, so the accumulator can
    /// never overflow 64 bits.
    #[inline]
    fn push(&mut self, code: u32, len: u8) {
        self.acc = (self.acc << len) | (code as u64 & ((1u64 << len) - 1));
        self.nbits += len as u32;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.buf.push((self.acc >> self.nbits) as u8);
        }
    }
    /// Flush the partial byte and append the read-ahead padding.
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push((self.acc << (8 - self.nbits)) as u8);
        }
        self.buf.resize(self.buf.len() + TAIL_PAD, 0);
        self.buf
    }
}

/// The 32 bits starting at bit `pos`, MSB-first.
///
/// One unaligned 8-byte load and two shifts. The bit-at-a-time loop this
/// replaced was the single biggest cost in decode. Safe for any `pos` within
/// the stream because the encoder appends [`TAIL_PAD`] bytes.
#[inline(always)]
fn peek32(bits: &[u8], pos: usize) -> u32 {
    let byte = pos / 8;
    let w = u64::from_be_bytes(bits[byte..byte + 8].try_into().unwrap());
    ((w << (pos % 8)) >> 32) as u32
}

// ── Huffman ─────────────────────────────────────────────────────────────────

/// Huffman code lengths for a set of positive symbol weights. O(n²) is fine at
/// n ≤ 16 and keeps this dependency-free.
fn huffman_lengths(counts: &[u64]) -> Vec<u8> {
    let m = counts.len();
    if m == 1 {
        return vec![1];
    }
    let mut weight: Vec<u64> = counts.to_vec();
    let mut parent: Vec<Option<usize>> = vec![None; m];
    let mut alive: Vec<usize> = (0..m).collect();
    while alive.len() > 1 {
        // Deterministic tie-break on index so encoder and decoder agree.
        alive.sort_by_key(|&i| (weight[i], i));
        let (a, b) = (alive[0], alive[1]);
        weight.push(weight[a] + weight[b]);
        parent.push(None);
        let node = weight.len() - 1;
        parent[a] = Some(node);
        parent[b] = Some(node);
        alive.drain(0..2);
        alive.push(node);
    }
    (0..m)
        .map(|i| {
            let mut d = 0u32;
            let mut cur = i;
            while let Some(p) = parent[cur] {
                d += 1;
                cur = p;
            }
            d.max(1) as u8
        })
        .collect()
}

/// Canonical codes from code lengths: sort by (length, symbol index), then
/// assign consecutive codes per length. Rebuildable from lengths alone, which
/// is why only the lengths are stored.
fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let max = lengths.iter().copied().max().unwrap_or(1);
    let mut codes = vec![0u32; lengths.len()];
    let mut code = 0u32;
    for len in 1..=max {
        for (sym, &l) in lengths.iter().enumerate() {
            if l == len {
                codes[sym] = code;
                code += 1;
            }
        }
        code <<= 1;
    }
    codes
}

/// Compress raw little-endian BF16 bytes. A trailing odd byte is ignored.
pub fn encode(bf16_le: &[u8]) -> Vec<u8> {
    encode_impl(bf16_le, None)
}

/// [`encode`] with the chunk-offset width pinned, so the 64-bit arm can be
/// covered without building a multi-gigabyte tensor. Production always lets the
/// width be derived from the real bit total.
#[cfg(test)]
fn encode_forced(bf16_le: &[u8], version: u8) -> Vec<u8> {
    encode_impl(bf16_le, Some(version))
}

fn encode_impl(bf16_le: &[u8], force_version: Option<u8>) -> Vec<u8> {
    let n = bf16_le.len() / 2;
    let at = |i: usize| u16::from_le_bytes([bf16_le[2 * i], bf16_le[2 * i + 1]]);

    // Alphabet: the DIRECT_SYMBOLS most frequent exponents, plus an escape
    // symbol iff anything falls outside them.
    let mut hist = [0u64; 256];
    for i in 0..n {
        hist[((at(i) >> 7) as u8) as usize] += 1;
    }
    let mut order: Vec<u8> = (0..=255u8).collect();
    order.sort_by_key(|&e| (std::cmp::Reverse(hist[e as usize]), e));
    let direct: Vec<u8> = order
        .iter()
        .copied()
        .take(DIRECT_SYMBOLS)
        .filter(|&e| hist[e as usize] > 0)
        .collect();

    let mut sym_of = [u8::MAX; 256]; // exponent -> symbol index
    let mut counts: Vec<u64> = Vec::with_capacity(MAX_SYMBOLS);
    for (s, &e) in direct.iter().enumerate() {
        sym_of[e as usize] = s as u8;
        counts.push(hist[e as usize]);
    }
    let escape_count: u64 = hist.iter().sum::<u64>() - counts.iter().sum::<u64>();
    let escape_sym = direct.len();
    if escape_count > 0 {
        counts.push(escape_count);
    }

    let n_chunks = n.div_ceil(CHUNK);
    if counts.is_empty() {
        // Empty tensor: header + empty tables only.
        let mut out = vec![0u8; mant_off_v(0, V0)];
        out[..4].copy_from_slice(&0u32.to_le_bytes());
        return out;
    }

    let lengths = huffman_lengths(&counts);
    let codes = canonical_codes(&lengths);
    let max_len = lengths.iter().copied().max().unwrap_or(1);

    // Exact bitstream length, known before a single bit is written: every
    // symbol costs its code length, and an escape costs 8 more for the literal
    // exponent. That is what lets the offset width be chosen up front instead
    // of discovered after the chunk table has already been sized.
    let total_bits: u64 = counts
        .iter()
        .zip(lengths.iter())
        .map(|(&c, &l)| c * l as u64)
        .sum::<u64>()
        + 8 * escape_count;
    let version = force_version.unwrap_or(if total_bits > u32::MAX as u64 { V1 } else { V0 });

    let mut out = vec![0u8; mant_off_v(n_chunks, version) + n];
    for (s, &e) in direct.iter().enumerate() {
        out[SYMTAB + s] = e;
    }
    for (s, &l) in lengths.iter().enumerate() {
        out[LENTAB + s] = l;
    }

    // Mantissa plane: sign bit plus the low 7 bits, one byte per element.
    let mant = mant_off_v(n_chunks, version);
    for i in 0..n {
        let bits = at(i);
        out[mant + i] = (((bits >> 15) as u8) << 7) | (bits as u8 & 0x7f);
    }

    let w = off_width(version);
    let mut bw = BitWriter::new();
    for c in 0..n_chunks {
        let pos = bw.bit_pos() as u64;
        debug_assert!(
            version == V1 || pos <= u32::MAX as u64,
            "v0 chunk offset {pos} exceeds u32; total_bits estimate was wrong"
        );
        out[CHUNKTAB + w * c..CHUNKTAB + w * c + w].copy_from_slice(&pos.to_le_bytes()[..w]);
        let start = c * CHUNK;
        for i in start..(start + CHUNK).min(n) {
            let e = (at(i) >> 7) as u8;
            let s = sym_of[e as usize];
            if s == u8::MAX {
                bw.push(codes[escape_sym], lengths[escape_sym]);
                bw.push(e as u32, 8); // literal exponent
            } else {
                bw.push(codes[s as usize], lengths[s as usize]);
            }
        }
    }

    out[..4].copy_from_slice(&(n as u32).to_le_bytes());
    out[4..8].copy_from_slice(&(n_chunks as u32).to_le_bytes());
    let stream = bw.finish();
    out[8..12].copy_from_slice(&(stream.len() as u32).to_le_bytes());
    out[12] = counts.len() as u8;
    out[13] = max_len;
    // Directly-coded exponent count. The alphabet carries an escape symbol iff
    // n_symbols > n_direct, and it is then symbol index n_direct. Storing this
    // explicitly beats inferring it — with fewer than 15 distinct exponents and
    // nothing escaping, the last symbol IS a real exponent.
    out[14] = direct.len() as u8;
    out[VERSION_OFF] = version;
    out.extend_from_slice(&stream);
    out
}

/// [`encode`], but `None` when the packed form is not smaller than the plain
/// BF16 input — the caller should then store plain BF16 (`QuantType::BF16`).
///
/// Also `None` when the tensor cannot be described by the header's 32-bit
/// `n_elems` / `bitstream_bytes` fields. Declining is the point: a truncated
/// count would produce an artifact that decodes to silent garbage, and storing
/// plain BF16 costs only the compression, not the data. The chunk-offset limit
/// that used to sit alongside these is gone — that one is now widened to 64-bit
/// rather than refused (see [`VERSION_OFF`]).
pub fn encode_if_smaller(bf16_le: &[u8]) -> Option<Vec<u8>> {
    if bf16_le.len() / 2 > MAX_ELEMS {
        return None;
    }
    let packed = encode(bf16_le);
    // The recorded stream length must survive the u32 field; otherwise the
    // truncation check in `View::new` compares against a wrapped value.
    let n_chunks = (bf16_le.len() / 2).div_ceil(CHUNK);
    let version = packed.get(VERSION_OFF).copied().unwrap_or(V0);
    let stream_len = packed
        .len()
        .saturating_sub(mant_off_v(n_chunks, version) + bf16_le.len() / 2);
    if stream_len > u32::MAX as usize {
        return None;
    }
    (packed.len() < bf16_le.len()).then_some(packed)
}

/// Element count recorded in the header.
pub fn n_elems(packed: &[u8]) -> Option<usize> {
    Some(u32::from_le_bytes(packed.get(..4)?.try_into().ok()?) as usize)
}

/// Number of independently decodable chunks.
pub fn n_chunks(packed: &[u8]) -> Option<usize> {
    Some(u32::from_le_bytes(packed.get(4..8)?.try_into().ok()?) as usize)
}

/// Bits of the multi-symbol table window (see [`Multi`]). 2^11 entries × 8 B =
/// 16 KB, which stays L1-resident.
const MULTI_BITS: usize = 11;
/// Exponents a single multi-symbol lookup can emit.
const MULTI_MAX: usize = 4;

/// One multi-symbol table entry: the exponents decodable from the window, and
/// the bits they consume. `n == 0` means "take the slow path" — the window
/// starts with an escape (whose literal lives in the stream, not the table) or
/// no whole code fits.
///
/// This is the `huf_decompress.c` X2/X4 idea. It pays unusually well here: the
/// exponent alphabet has ~2.66-bit average codes, so an 11-bit window normally
/// holds four whole symbols and one lookup replaces four decode steps.
#[derive(Clone, Copy, Default)]
struct Multi {
    exps: [u8; MULTI_MAX],
    n: u8,
    bits: u8,
}

/// Decoder tables rebuilt from the stored code lengths.
struct Decoder {
    /// `PRIMARY_BITS`-wide direct table: (symbol, code length), length 0 = miss.
    primary: Vec<(u8, u8)>,
    /// Canonical fallback for codes longer than `PRIMARY_BITS`.
    lengths: Vec<u8>,
    codes: Vec<u32>,
    symbols: [u8; MAX_SYMBOLS],
    /// Symbol index of the escape, or `None` when the alphabet has no escape.
    escape_sym: Option<usize>,
    max_len: u8,
    /// Multi-symbol acceleration table, indexed by the top [`MULTI_BITS`] bits.
    multi: Vec<Multi>,
}

impl Decoder {
    fn new(packed: &[u8]) -> Option<Self> {
        let n_symbols = *packed.get(12)? as usize;
        let max_len = *packed.get(13)?;
        if n_symbols == 0 || n_symbols > MAX_SYMBOLS || max_len == 0 || max_len > 31 {
            return None;
        }
        let mut symbols = [0u8; MAX_SYMBOLS];
        symbols.copy_from_slice(packed.get(SYMTAB..SYMTAB + MAX_SYMBOLS)?);
        let lengths: Vec<u8> = packed.get(LENTAB..LENTAB + n_symbols)?.to_vec();
        if lengths.iter().any(|&l| l == 0 || l > max_len) {
            return None;
        }
        let codes = canonical_codes(&lengths);

        // Any code of at most PRIMARY_BITS resolves in one lookup: fill every
        // table slot whose leading bits match the code.
        let mut primary = vec![(0u8, 0u8); 1 << PRIMARY_BITS];
        for (sym, &l) in lengths.iter().enumerate() {
            if l as usize <= PRIMARY_BITS {
                let shift = PRIMARY_BITS - l as usize;
                let base = (codes[sym] as usize) << shift;
                for slot in base..base + (1 << shift) {
                    primary[slot] = (sym as u8, l);
                }
            }
        }
        let n_direct = *packed.get(14)? as usize;
        if n_direct > n_symbols || n_symbols > n_direct + 1 {
            return None;
        }
        let escape_sym = (n_symbols > n_direct).then_some(n_direct);
        let mut dec = Self {
            primary,
            lengths,
            codes,
            symbols,
            escape_sym,
            max_len,
            multi: Vec::new(),
        };
        dec.build_multi();
        Some(dec)
    }

    /// Precompute the multi-symbol table: for every possible window value,
    /// greedily decode whole codes until the window runs out, four symbols are
    /// found, or an escape appears.
    fn build_multi(&mut self) {
        let mut table = vec![Multi::default(); 1 << MULTI_BITS];
        for (v, slot) in table.iter_mut().enumerate() {
            let w = (v as u32) << (32 - MULTI_BITS);
            let mut used = 0usize;
            let mut e = Multi::default();
            while (e.n as usize) < MULTI_MAX && used < MULTI_BITS {
                let Some((sym, l)) = self.symbol(w << used) else {
                    break;
                };
                // Only accept a code lying wholly inside the window: a prefix
                // code is determined by its own bits, so the symbol is then
                // correct no matter what follows.
                if used + l > MULTI_BITS || Some(sym) == self.escape_sym {
                    break;
                }
                e.exps[e.n as usize] = self.symbols[sym];
                e.n += 1;
                used += l;
            }
            e.bits = used as u8;
            *slot = e;
        }
        self.multi = table;
    }

    /// Resolve a code longer than [`PRIMARY_BITS`] from a 32-bit window.
    /// Cold: short codes are by construction the frequent ones.
    #[cold]
    fn slow(&self, w: u32) -> Option<(usize, usize)> {
        for l in (PRIMARY_BITS + 1)..=self.max_len as usize {
            let v = w >> (32 - l);
            for (sym, &sl) in self.lengths.iter().enumerate() {
                if sl as usize == l && self.codes[sym] == v {
                    return Some((sym, l));
                }
            }
        }
        None
    }

    /// Decode one symbol from a 32-bit window; returns (symbol, bits consumed).
    #[inline(always)]
    fn symbol(&self, w: u32) -> Option<(usize, usize)> {
        let (sym, len) = self.primary[(w >> (32 - PRIMARY_BITS)) as usize];
        if len != 0 {
            return Some((sym as usize, len as usize));
        }
        self.slow(w)
    }
}

/// Shared, immutable view of a payload — built once, usable from any thread.
struct View<'a> {
    dec: Decoder,
    mant: &'a [u8],
    bits: &'a [u8],
    offsets: &'a [u8],
    /// Bytes per chunk-table entry: 4 for a v0 artifact, 8 for v1.
    off_width: usize,
    n: usize,
    n_chunks: usize,
}

impl<'a> View<'a> {
    fn new(packed: &'a [u8], n: usize) -> Option<Self> {
        if n_elems(packed)? != n {
            return None;
        }
        let nc = n_chunks(packed)?;
        if nc != n.div_ceil(CHUNK) {
            return None;
        }
        let version = *packed.get(VERSION_OFF)?;
        if version != V0 && version != V1 {
            return None;
        }
        let w = off_width(version);
        let mant_base = mant_off_v(nc, version);
        let mant = packed.get(mant_base..mant_base + n)?;
        let bits = packed.get(mant_base + n..)?;
        // `peek` zero-pads past the end, so a short bitstream would otherwise
        // decode to plausible garbage rather than failing. The header records
        // its exact length precisely so truncation is detectable here.
        let want = u32::from_le_bytes(packed.get(8..12)?.try_into().ok()?) as usize;
        if bits.len() < want {
            return None;
        }
        Some(Self {
            dec: Decoder::new(packed)?,
            mant,
            bits,
            offsets: packed.get(CHUNKTAB..CHUNKTAB + w * nc)?,
            off_width: w,
            n,
            n_chunks: nc,
        })
    }

    /// Decode chunk `c` into `out`, which must be exactly that chunk's output
    /// bytes (`2 *` its element count). Chunks are independent: this reads only
    /// the chunk's own bit range and mantissa bytes.
    fn chunk(&self, c: usize, out: &mut [u8]) -> Option<()> {
        let start = c * CHUNK;
        let end = (start + CHUNK).min(self.n);
        if out.len() != 2 * (end - start) {
            return None;
        }
        let w = self.off_width;
        let raw = self.offsets.get(w * c..w * c + w)?;
        let mut buf = [0u8; 8];
        buf[..w].copy_from_slice(raw);
        let mut pos = u64::from_le_bytes(buf) as usize;
        let esc = self.dec.escape_sym;
        let mut i = start;
        while i < end {
            // One load covers a code (<=15 bits) plus any literal (8 bits), or
            // a whole multi-symbol group.
            let w = peek32(self.bits, pos);
            let m = &self.dec.multi[(w >> (32 - MULTI_BITS)) as usize];
            let n = m.n as usize;
            if n > 0 && i + n <= end {
                // Fast path: several exponents from a single lookup.
                for k in 0..n {
                    let o = 2 * (i + k - start);
                    put_bf16(&mut out[o..o + 2], self.mant[i + k], m.exps[k]);
                }
                pos += m.bits as usize;
                i += n;
                continue;
            }
            // Slow path: an escape, or too few elements left for the group.
            let (sym, used) = self.dec.symbol(w)?;
            pos += used;
            let e = if Some(sym) == esc {
                pos += 8;
                ((w << used) >> 24) as u8
            } else {
                self.dec.symbols[sym]
            };
            let o = 2 * (i - start);
            put_bf16(&mut out[o..o + 2], self.mant[i], e);
            i += 1;
        }
        Some(())
    }
}

/// Reconstruct one BF16 from its mantissa byte and exponent.
#[inline(always)]
fn put_bf16(slot: &mut [u8], sm: u8, e: u8) {
    let bits = ((sm as u16 & 0x80) << 8) | ((e as u16) << 7) | (sm as u16 & 0x7f);
    slot.copy_from_slice(&bits.to_le_bytes());
}

/// Decompress a `BF16H` payload back to raw little-endian BF16 bytes.
/// `n` is the element count from the tensor shape.
pub fn decode(packed: &[u8], n: usize) -> Option<Vec<u8>> {
    if n == 0 {
        return (n_elems(packed)? == 0).then(Vec::new);
    }
    let view = View::new(packed, n)?;
    let mut out = vec![0u8; n * 2];
    for (c, slot) in out.chunks_mut(CHUNK * 2).enumerate() {
        view.chunk(c, slot)?;
    }
    Some(out)
}

/// [`decode`], spread over `threads` OS threads.
///
/// Huffman codes are bit-serial, so a single core decodes at only ~350 MB/s —
/// minutes for a full BF16 artifact. Chunks are independently addressable
/// precisely so that expansion at load parallelises; this is the reason the
/// chunk table is in the format at all. `threads <= 1` decodes inline.
pub fn decode_par(packed: &[u8], n: usize, threads: usize) -> Option<Vec<u8>> {
    if threads <= 1 || n <= CHUNK {
        return decode(packed, n);
    }
    let view = View::new(packed, n)?;
    let mut out = vec![0u8; n * 2];
    // Hand each thread a contiguous run of whole chunks; the slices are disjoint
    // so no synchronisation is needed beyond the scope join. Give every thread
    // at least two chunks of work — spawning costs more than it saves on the
    // many small tensors (norms, biases) a real artifact is full of.
    let threads = threads.min(view.n_chunks.div_ceil(2)).max(1);
    let per = view.n_chunks.div_ceil(threads);
    let ok = std::sync::atomic::AtomicBool::new(true);
    std::thread::scope(|s| {
        for (t, slot) in out.chunks_mut(per * CHUNK * 2).enumerate() {
            let view = &view;
            let ok = &ok;
            s.spawn(move || {
                for (j, sub) in slot.chunks_mut(CHUNK * 2).enumerate() {
                    if view.chunk(t * per + j, sub).is_none() {
                        ok.store(false, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                }
            });
        }
    });
    ok.load(std::sync::atomic::Ordering::Relaxed).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // All 65536 patterns: zeros, denormals, infinities, NaN payloads — and
        // 256 distinct exponents, so the escape path is heavily exercised.
        roundtrips(&(0..=u16::MAX).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_handles_ragged_tails_and_empty() {
        for n in [0usize, 1, 5, 8, 255, 256, 8191, 8192, 8193, 20_000] {
            let bits: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(2654)).collect();
            roundtrips(&bits);
        }
    }

    #[test]
    fn single_symbol_tensor_roundtrips() {
        // Degenerate alphabet: one exponent, so one symbol of length 1.
        let bits: Vec<u16> = (0..5000).map(|i| 0x3f80 | (i as u16 & 0x7f)).collect();
        roundtrips(&bits);
    }

    #[test]
    fn beats_lut3_on_gaussian_weights() {
        // The reason this format exists: on realistic weights it must compress
        // materially better than the fixed-3-bit sibling.
        let mut seed = 0x1234_5678u32;
        let bits: Vec<u16> = (0..200_000)
            .map(|_| {
                let s: i64 = (0..4).map(|_| (xorshift(&mut seed) % 2048) as i64).sum();
                crate::conv::f32_to_bf16_bits((s - 4094) as f32 * 1e-4)
            })
            .collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let huff = roundtrips(&bits);
        let lut3 = crate::bf16_lut3::encode(&raw);
        let (rh, rl) = (
            raw.len() as f64 / huff.len() as f64,
            raw.len() as f64 / lut3.len() as f64,
        );
        assert!(rh > rl, "huff {rh:.4}x must beat lut3 {rl:.4}x");
        assert!(
            rh > 1.44,
            "expected ~1.5x on Gaussian weights, got {rh:.4}x"
        );
    }

    #[test]
    fn chunks_start_at_recorded_bit_offsets() {
        // The chunk table is what makes parallel decode possible; a wrong offset
        // would still decode sequentially, so assert it directly.
        let mut seed = 0xabcd_1234u32;
        let bits: Vec<u16> = (0..30_000).map(|_| xorshift(&mut seed) as u16).collect();
        let packed = roundtrips(&bits);
        let nc = n_chunks(&packed).unwrap();
        assert_eq!(nc, 30_000usize.div_ceil(CHUNK));
        let dec = Decoder::new(&packed).unwrap();
        // This fixture is small, so it is a v0 artifact with a 32-bit table —
        // which is exactly what the offsets are read as just below.
        assert_eq!(packed[VERSION_OFF], V0);
        let mant_base = mant_off_v(nc, V0);
        let stream = &packed[mant_base + 30_000..];
        // Decoding CHUNK symbols from chunk c must land exactly on chunk c+1.
        for c in 0..nc - 1 {
            let off = |i: usize| {
                u32::from_le_bytes(
                    packed[CHUNKTAB + 4 * i..CHUNKTAB + 4 * i + 4]
                        .try_into()
                        .unwrap(),
                ) as usize
            };
            let mut pos = off(c);
            for _ in 0..CHUNK {
                let (sym, used) = dec.symbol(peek32(stream, pos)).unwrap();
                pos += used;
                if Some(sym) == dec.escape_sym {
                    pos += 8;
                }
            }
            assert_eq!(
                pos,
                off(c + 1),
                "chunk {c} does not end where {} starts",
                c + 1
            );
        }
    }

    #[test]
    fn parallel_decode_matches_sequential() {
        // Chunk boundaries are where a parallel decode goes wrong; check several
        // thread counts that do and do not divide the chunk count evenly.
        let mut seed = 0x5150_7777u32;
        let n = 5 * CHUNK + 137;
        let bits: Vec<u16> = (0..n).map(|_| xorshift(&mut seed) as u16).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let packed = encode(&raw);
        let seq = decode(&packed, n).expect("sequential");
        assert_eq!(seq, raw);
        for t in [1usize, 2, 3, 4, 8, 64] {
            assert_eq!(
                decode_par(&packed, n, t).as_deref(),
                Some(raw.as_slice()),
                "threads={t}"
            );
        }
    }

    #[test]
    fn truncated_payload_decodes_to_none() {
        let n = 10_000;
        let bits: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(7919)).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let packed = encode(&raw);
        for cut in [0usize, 4, 16, 64, packed.len() / 2] {
            assert!(decode(&packed[..cut], n).is_none(), "cut at {cut}");
        }
    }

    #[test]
    fn wrong_element_count_is_rejected() {
        let bits: Vec<u16> = (0..600).map(|i| 0x3f80 | (i as u16 & 0x7f)).collect();
        let packed = roundtrips(&bits);
        assert!(decode(&packed, 601).is_none());
        assert!(decode(&packed, 100).is_none());
    }
}

#[cfg(test)]
mod overflow_regression {
    use super::*;

    /// Weight-shaped BF16: exponents in a narrow band, mantissas varied.
    fn weights(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut v = Vec::with_capacity(n * 2);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let exp = 0x3Cu16 + ((s >> 3) & 0x7) as u16; // few distinct exponents
            let bits = (exp << 8) | ((s >> 11) & 0xFF) as u16;
            v.extend_from_slice(&bits.to_le_bytes());
        }
        v
    }

    /// Exponent-diverse BF16, so nearly every element escapes and the stream
    /// costs ~12 bits/element instead of ~2.6. That is what makes crossing the
    /// 2^32-bit boundary affordable to test at all.
    fn escape_heavy(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut v = Vec::with_capacity(n * 2);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.extend_from_slice(&(s as u16).to_le_bytes()); // exponent spans all 256
        }
        v
    }

    /// A small tensor must still be written as v0, byte-for-byte as before this
    /// fix. Artifacts already on disk may outlive their sources, so the v0 arm
    /// is not free to drift.
    #[test]
    fn small_tensors_stay_v0_and_round_trip() {
        let raw = weights(100_000, 0xABCD);
        let packed = encode(&raw);
        assert_eq!(packed[VERSION_OFF], V0, "small tensor must stay v0");
        assert_eq!(decode(&packed, raw.len() / 2).unwrap(), raw);
    }

    /// The 64-bit-offset arm, exercised without a multi-GB allocation: same
    /// data, width pinned to v1, must decode identically and lay its chunk
    /// table out 8 bytes per entry.
    #[test]
    fn v1_offsets_round_trip_and_widen_the_chunk_table() {
        let raw = weights(100_000, 0x1234);
        let n = raw.len() / 2;
        let v0 = encode_forced(&raw, V0);
        let v1 = encode_forced(&raw, V1);

        assert_eq!(v0[VERSION_OFF], V0);
        assert_eq!(v1[VERSION_OFF], V1);

        // Same payload, wider table: exactly 4 extra bytes per chunk.
        let nc = n.div_ceil(CHUNK);
        assert_eq!(v1.len() - v0.len(), 4 * nc);

        assert_eq!(decode(&v1, n).unwrap(), raw, "v1 must decode exactly");
        assert_eq!(decode(&v0, n).unwrap(), raw, "v0 must still decode exactly");
        // Parallel decode walks the chunk table directly, so cover it too.
        assert_eq!(decode_par(&v1, n, 4).unwrap(), raw);
    }

    /// A corrupt/unknown version must be refused rather than decoded with a
    /// guessed offset width.
    #[test]
    fn unknown_version_is_refused() {
        let raw = weights(20_000, 7);
        let mut packed = encode(&raw);
        packed[VERSION_OFF] = 99;
        assert!(decode(&packed, raw.len() / 2).is_none());
    }

    /// The actual bug: a bitstream past 2^32 bits.
    ///
    /// v0 truncated each chunk's bit offset to `u32`, so every chunk after the
    /// wrap decoded from the wrong position — correct mantissas, wrong
    /// exponents, no error reported. Measured on gemma-4-E2B's 4.7 GB
    /// `embed_tokens_per_layer`: byte-identical for the first 70%, then 1.41 GB
    /// in which 100% of differing elements kept their sign and mantissa and
    /// differed only in exponent.
    ///
    /// Ignored by default: escape-heavy data costs ~8.5 bits/element, so it
    /// still takes 600 M elements to pass 2^32 bits, peaking near 3.6 GB of RAM.
    /// Build in release — a debug encode of this size is minutes, not seconds.
    #[test]
    #[ignore = "allocates ~3.6 GB; run with --release --ignored"]
    fn bitstream_past_2_32_bits_round_trips() {
        const N: usize = 600_000_000;
        let raw = escape_heavy(N, 0xFEED);
        let packed = encode(&raw);

        assert_eq!(
            packed[VERSION_OFF], V1,
            "a stream past 2^32 bits must select the 64-bit offset format"
        );

        // The last chunk must genuinely start beyond what a u32 could hold,
        // otherwise this test would pass without reaching the old bug.
        let nc = n_chunks(&packed).unwrap();
        let last = {
            let base = CHUNKTAB + 8 * (nc - 1);
            u64::from_le_bytes(packed[base..base + 8].try_into().unwrap())
        };
        assert!(
            last > u32::MAX as u64,
            "final chunk offset {last} did not exceed u32::MAX; test data too small"
        );

        let out = decode(&packed, N).unwrap();
        assert_eq!(out.len(), raw.len());
        assert!(out == raw, "round trip past the 2^32-bit boundary must be exact");
    }
}
