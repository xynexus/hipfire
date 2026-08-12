// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

// Expand a BF16H (Huffman-coded exponent) payload back to plain BF16 on device.
//
// Why this exists: `bf16_huff` is the format we write to DISK (1.50x vs
// bf16_lut3's 1.38x), but its codes are variable-length, so until now it could
// only be expanded on the host at load. On a UMA APU that means the whole
// tensor crosses the CPU before the GPU sees it. This kernel gives the caller
// the choice — expand in the loader, or hand the packed bytes to the device and
// expand here.
//
// This is NOT the bf16_lut3 story. LUT3 is byte-aligned and block-addressable,
// so `gemv_bf16l3` decodes it in registers and the weights never materialise as
// BF16 at all — that saves BANDWIDTH. Huffman cannot do that: a variable-length
// code has no random access, so this kernel materialises plain BF16 in VRAM
// exactly like the host path would. What it saves is the host round-trip and
// the host-side decode, not weight bandwidth.
//
// Parallelism comes from the format's chunk table: every CHUNK elements start
// at a known bit offset with directly-indexed mantissa bytes, so chunks are
// independent. Decode within a chunk is inherently serial (that is what
// variable-length means), so the mapping is ONE THREAD PER CHUNK. A 262M-element
// embedding table is ~32k chunks, which is ample occupancy.
//
// The cost of that mapping is write coalescing: thread t writes the contiguous
// 16 KB region [t*2*CHUNK, (t+1)*2*CHUNK), so neighbouring lanes touch addresses
// 16 KB apart. That is inherent — output positions are fixed by the format — and
// acceptable for a one-shot load-time expansion.
//
// Layout constants MUST mirror hipfire_primitives::bf16_huff. A silent
// divergence here decodes to plausible-but-wrong weights (right magnitude,
// wrong values), which no magnitude check catches — the same failure mode the
// v0/v1 chunk-offset note in that module describes.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define BF16H_CHUNK         8192
#define BF16H_MAX_SYMBOLS     16
#define BF16H_PRIMARY_BITS     8
#define BF16H_HEADER          16
#define BF16H_SYMTAB          16   // HEADER
#define BF16H_LENTAB          32   // SYMTAB + MAX_SYMBOLS
#define BF16H_CHUNKTAB        48   // LENTAB + MAX_SYMBOLS

// Header byte offsets.
#define BF16H_OFF_NELEMS       0   // u32
#define BF16H_OFF_NCHUNKS      4   // u32
#define BF16H_OFF_BITBYTES     8   // u32
#define BF16H_OFF_NSYMBOLS    12   // u8
#define BF16H_OFF_MAXLEN      13   // u8
#define BF16H_OFF_NDIRECT     14   // u8
#define BF16H_OFF_VERSION     15   // u8

// Read 32 bits MSB-first starting at bit `pos`. Mirrors `bf16_huff::peek32`:
// a big-endian 8-byte load, shifted left by the intra-byte offset, top 32 taken.
// Over-reading up to 8 bytes past the last code is safe because the encoder
// appends TAIL_PAD = 8 zero bytes.
// Two ALIGNED dword loads, not eight byte loads: the bit cursor moves by a few
// bits per symbol, so this is the hottest line in the kernel and a dependent
// byte-at-a-time assembly serialises it completely.
__device__ __forceinline__ uint32_t bf16h_peek32(const uint8_t* bits, uint64_t pos) {
    // Align on the ABSOLUTE address, not on an offset within `bits`: the
    // bitstream starts at `mant + n` and n is the element count, so `bits`
    // itself is odd-aligned whenever n is odd. Aligning relative to it would
    // hand the dword load a misaligned pointer.
    const uintptr_t addr = (uintptr_t)bits + (uintptr_t)(pos >> 3);
    const uintptr_t base = addr & ~(uintptr_t)3;
    const uint32_t* w32 = (const uint32_t*)base;
    // Byte-swap each dword so the pair reads as the big-endian byte order the
    // format's MSB-first codes are written in.
    const uint64_t hi = (uint64_t)__builtin_bswap32(w32[0]);
    const uint64_t lo = (uint64_t)__builtin_bswap32(w32[1]);
    const uint64_t w = (hi << 32) | lo;
    // Bit offset of `pos` from the aligned base: at most 3 bytes + 7 bits = 31,
    // so the 32-bit window always fits inside the 64 bits loaded.
    const uint32_t off = (uint32_t)(((addr - base) << 3) + (pos & 7));
    return (uint32_t)((w << off) >> 32);
}

// Canonical Huffman codes from lengths, exactly as `canonical_codes` builds
// them: walk lengths ascending, assign sequentially, shift left per level.
__device__ __forceinline__ void bf16h_canonical(const uint8_t* lengths, int n_symbols,
                                                uint8_t max_len, uint32_t* codes) {
    uint32_t code = 0;
    for (uint32_t len = 1; len <= (uint32_t)max_len; ++len) {
        for (int s = 0; s < n_symbols; ++s) {
            if ((uint32_t)lengths[s] == len) {
                codes[s] = code;
                ++code;
            }
        }
        code <<= 1;
    }
}

// Decoder state, small enough to live in LDS per workgroup.
struct Bf16hDec {
    uint16_t primary[1 << BF16H_PRIMARY_BITS];  // (sym << 8) | len, len 0 = miss
    uint32_t codes[BF16H_MAX_SYMBOLS];
    uint8_t  lengths[BF16H_MAX_SYMBOLS];
    uint8_t  symbols[BF16H_MAX_SYMBOLS];        // exponent per symbol
    int      n_symbols;
    int      escape_sym;                        // -1 when the alphabet has none
    uint8_t  max_len;
};

// Resolve one symbol from a 32-bit window. Returns bits consumed; writes the
// symbol index. Short codes — which are by construction the frequent ones —
// resolve in a single primary lookup; longer ones fall back to the canonical
// compare, mirroring `Decoder::symbol` / `Decoder::slow`.
__device__ __forceinline__ int bf16h_symbol(const Bf16hDec* d, uint32_t w, int* sym) {
    const uint16_t e = d->primary[w >> (32 - BF16H_PRIMARY_BITS)];
    const int len = e & 0xff;
    if (len != 0) {
        *sym = e >> 8;
        return len;
    }
    for (int l = BF16H_PRIMARY_BITS + 1; l <= (int)d->max_len; ++l) {
        const uint32_t v = w >> (32 - l);
        for (int s = 0; s < d->n_symbols; ++s) {
            if ((int)d->lengths[s] == l && d->codes[s] == v) {
                *sym = s;
                return l;
            }
        }
    }
    *sym = -1;   // malformed payload; caller stops
    return 0;
}

// Reassemble one BF16 from its mantissa byte and exponent, matching `put_bf16`:
// sign from bit 7 of the mantissa byte, then exponent, then mantissa[6:0].
__device__ __forceinline__ uint16_t bf16h_put(uint8_t sm, uint8_t e) {
    return (uint16_t)((((uint16_t)sm & 0x80) << 8) | ((uint16_t)e << 7) |
                      ((uint16_t)sm & 0x7f));
}

// One workgroup builds the shared decode tables once, then each thread decodes
// whole chunks. `n` is the element count from the tensor shape.
//
// `packed` is the whole BF16H payload; `out` receives n little-endian BF16.
extern "C" __global__ void bf16_huff_decode(const uint8_t* __restrict__ packed,
                                            uint16_t* __restrict__ out,
                                            uint32_t n) {
    __shared__ Bf16hDec d;

    const uint32_t n_chunks = *(const uint32_t*)(packed + BF16H_OFF_NCHUNKS);
    const int n_symbols     = packed[BF16H_OFF_NSYMBOLS];
    const uint8_t max_len   = packed[BF16H_OFF_MAXLEN];
    const int n_direct      = packed[BF16H_OFF_NDIRECT];
    const uint8_t version   = packed[BF16H_OFF_VERSION];
    // v0 stored chunk bit offsets as u32; v1 widened them to u64 so a bitstream
    // past 2^32 bits stops wrapping. Both must keep decoding — artifacts written
    // before that fix outlive their sources.
    const uint32_t off_w = (version == 0) ? 4u : 8u;

    if (threadIdx.x == 0) {
        d.n_symbols = n_symbols;
        d.max_len   = max_len;
        // The escape symbol exists only when the alphabet carries one more
        // symbol than it codes directly.
        d.escape_sym = (n_symbols > n_direct) ? n_direct : -1;
        for (int s = 0; s < BF16H_MAX_SYMBOLS; ++s) {
            d.symbols[s] = packed[BF16H_SYMTAB + s];
            d.lengths[s] = (s < n_symbols) ? packed[BF16H_LENTAB + s] : 0;
            d.codes[s]   = 0;
        }
        bf16h_canonical(d.lengths, n_symbols, max_len, d.codes);
        for (int i = 0; i < (1 << BF16H_PRIMARY_BITS); ++i) d.primary[i] = 0;
        // Every slot whose leading bits match a short code resolves to it.
        for (int s = 0; s < n_symbols; ++s) {
            const int l = d.lengths[s];
            if (l != 0 && l <= BF16H_PRIMARY_BITS) {
                const int shift = BF16H_PRIMARY_BITS - l;
                const uint32_t base = d.codes[s] << shift;
                for (uint32_t k = 0; k < (1u << shift); ++k) {
                    d.primary[base + k] = (uint16_t)((s << 8) | l);
                }
            }
        }
    }
    __syncthreads();

    const uint64_t mant_base = (uint64_t)BF16H_CHUNKTAB + (uint64_t)off_w * n_chunks;
    const uint8_t* mant = packed + mant_base;
    const uint8_t* bits = mant + n;
    const uint8_t* offs = packed + BF16H_CHUNKTAB;

    // Each thread takes a CONTIGUOUS run of chunks, and the caller caps the
    // thread count. Both matter, and the reason is cache residency rather than
    // occupancy.
    //
    // Measured: at 2048 chunks (~23 MB of bitstream, LLC-resident) this decodes
    // at 0.3 ns/element; at 32069 chunks (~370 MB) a grid-strided one-chunk-per-
    // thread mapping fell to 6.9 ns/element — 20x worse than linear. Every live
    // thread is an independent bit cursor walking its own region, so concurrency
    // IS working-set size here. Fewer, longer-lived cursors keep the aggregate
    // window inside the last-level cache; a grid-stride loop would instead give
    // each thread a set of chunks scattered across the whole stream.
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_thread = (n_chunks + total_threads - 1) / total_threads;
    const uint32_t c_begin = tid * per_thread;
    uint32_t c_end = c_begin + per_thread;
    if (c_end > n_chunks) c_end = n_chunks;

    for (uint32_t c = c_begin; c < c_end; ++c) {
        const uint32_t start = c * BF16H_CHUNK;
        if (start >= n) continue;
        uint32_t end = start + BF16H_CHUNK;
        if (end > n) end = n;

        // Chunk bit offset, width per format version.
        uint64_t pos = 0;
        for (uint32_t b = 0; b < off_w; ++b) {
            pos |= (uint64_t)offs[(uint64_t)off_w * c + b] << (8 * b);  // little-endian
        }

        for (uint32_t i = start; i < end; ++i) {
            const uint32_t w = bf16h_peek32(bits, pos);
            int sym = -1;
            const int used = bf16h_symbol(&d, w, &sym);
            if (sym < 0) return;   // malformed payload
            pos += (uint64_t)used;
            uint8_t e;
            if (sym == d.escape_sym) {
                // An escape is followed by 8 raw bits holding the literal
                // exponent, read from the same window past the code.
                e = (uint8_t)((w << used) >> 24);
                pos += 8;
            } else {
                e = d.symbols[sym];
            }
            out[i] = bf16h_put(mant[i], e);
        }
    }
}
