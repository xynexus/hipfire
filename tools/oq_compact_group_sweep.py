#!/usr/bin/env python3
"""OqPlusCompact group-size viability sweep: G in {64,128,256,512,1024}.

Faithful port of hipfire-quantize/src/codecs.rs `quantize_oqplus_compact` +
`mixed_clipsearch`, generalised from the hardcoded G=256 to arbitrary G:

  symmetric_clipsearch(±7) -> overlay indices by gain (err4^2 - err8^2)
  -> refit_mixed_scale -> indices again -> refit again.
  Bulk clamps to ±7, overlay to ±127, ONE shared f16 scale.

Block bytes = 2 (f16 scale) + G/2 (nibbles) + N_out * entry, where entry is 2
for G<=256 (u8 idx + i8 val) and 3 for G>256, since a u8 index cannot address a
position >= 256. That widening is part of the economics of going bigger.

Experiment only; not production tooling.
"""
import sys, math, json, struct
import numpy as np

SNAP = ("/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17/"
        "model.safetensors-00001-of-00001.safetensors")

CLIP_SYM = np.array([1.0, .95, .9, .85, .8, .75, .7, .65, .6])
CLIP_REFIT = np.array([1.0, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35])


def load_bf16(path, names):
    out = {}
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for name in names:
            if name not in hdr:
                continue
            info = hdr[name]
            s, e = info["data_offsets"]
            f.seek(base + s)
            u16 = np.frombuffer(f.read(e - s), dtype=np.uint16)
            out[name] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    return out


def fwht_rows(x):
    """Normalised Walsh-Hadamard over the last axis (power-of-two length)."""
    x = x.astype(np.float64).copy()
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        u = x[:, :, 0, :].copy()
        v = x[:, :, 1, :].copy()
        x[:, :, 0, :] = u + v
        x[:, :, 1, :] = u - v
        x = x.reshape(-1, n)
        h *= 2
    return x / math.sqrt(n)


def _sse4(R, scale):
    q = np.clip(np.rint(R / scale), -7, 7)
    d = R - q * scale
    return (d * d).sum(axis=1)


def _mixed_err(R, scale, mask8):
    q4 = np.clip(np.rint(R / scale), -7, 7)
    q8 = np.clip(np.rint(R / scale), -127, 127)
    q = np.where(mask8, q8, q4)
    d = R - q * scale
    return (d * d).sum(axis=1)


def _overlay_mask(R, scale, n_out):
    q4 = np.clip(np.rint(R / scale), -7, 7)
    q8 = np.clip(np.rint(R / scale), -127, 127)
    e4 = R - q4 * scale
    e8 = R - q8 * scale
    gain = e4 * e4 - e8 * e8
    idx = np.argpartition(-gain, n_out - 1, axis=1)[:, :n_out]
    mask = np.zeros(R.shape, dtype=bool)
    mask[np.arange(R.shape[0])[:, None], idx] = True
    return mask


def _best_scale(R, amax, grid, scorer):
    best_s = None
    best_e = None
    for c in grid:
        s = np.maximum(c * amax / 7.0, 1e-12)
        e = scorer(s)
        if best_e is None:
            best_s, best_e = s.copy(), e.copy()
        else:
            take = (e < best_e)[:, None]      # e is (rows,), s is (rows,1)
            best_s = np.where(take, s, best_s)
            best_e = np.where(take[:, 0], e, best_e)
    return best_s


def quant_compact_snr(W, G, n_out, rng):
    M, K = W.shape
    assert K % G == 0
    X = W.reshape(-1, G)
    s1 = rng.choice([-1.0, 1.0], size=G)
    s2 = rng.choice([-1.0, 1.0], size=G)
    R = fwht_rows(X * s1) * s2                       # randomised Hadamard, as in cpu_fwht_256
    amax = np.abs(R).max(axis=1, keepdims=True)

    scale = _best_scale(R, amax, CLIP_SYM, lambda s: _sse4(R, s))
    for _ in range(2):                               # indices -> refit, twice
        mask = _overlay_mask(R, scale, n_out)
        scale = _best_scale(R, amax, CLIP_REFIT, lambda s: _mixed_err(R, s, mask))

    mask = _overlay_mask(R, scale, n_out)
    q = np.where(mask, np.clip(np.rint(R / scale), -127, 127),
                 np.clip(np.rint(R / scale), -7, 7))
    rec = q * scale
    num = float((R ** 2).sum())
    den = float(((R - rec) ** 2).sum())
    return 10.0 * math.log10(num / den) if den > 0 else float("inf")


def bpw(G, n_out):
    entry = 2 if G <= 256 else 3
    return (2 + G // 2 + n_out * entry) * 8.0 / G


TENSORS = [
    "model.language_model.layers.11.self_attn.q_proj.weight",
    "model.language_model.layers.11.self_attn.k_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]
GROUPS = [64, 128, 256, 512, 1024]


MAX_ROWS = 256   # ~256*K weights per tensor is a large sample for an SNR mean


def main():
    W = load_bf16(SNAP, TENSORS)
    W = {k: v[:MAX_ROWS] for k, v in W.items()}
    if not W:
        print("no tensors loaded", file=sys.stderr)
        return 2
    rng = np.random.default_rng(42)

    print("== hard constraint: K % G == 0 ==")
    for name, w in W.items():
        K = w.shape[1]
        bad = [G for G in GROUPS if K % G != 0]
        print(f"  {name.split('.')[-2]:<11} K={K:<6} fails: {bad if bad else 'none'}")
    print()

    print("== SNR at matched cost (largest N_out fitting <= 4.25 bits/weight) ==")
    print(f"  {'G':<7}{'N_out':<7}{'bits/w':<9}{'SNR dB':<9}{'vs G=256':<10}")
    base = {}
    rows = []
    for G in GROUPS:
        entry = 2 if G <= 256 else 3
        n_out = max(1, int((4.25 * G / 8.0 - 2 - G // 2) // entry))
        snrs = [quant_compact_snr(w.astype(np.float32), G, n_out, rng)
                for w in W.values() if w.shape[1] % G == 0]
        rows.append((G, n_out, bpw(G, n_out), np.mean(snrs) if snrs else None, len(snrs)))
    ref = next(r[3] for r in rows if r[0] == 256)
    for G, n_out, b, s, ntens in rows:
        if s is None:
            print(f"  {G:<7}{n_out:<7}{b:<9.3f}{'N/A':<9}{'(no divisible tensor)':<10}")
        else:
            note = f"{s-ref:+.2f} dB" + ("" if ntens == len(W) else f"  [{ntens}/{len(W)} tensors]")
            print(f"  {G:<7}{n_out:<7}{b:<9.3f}{s:<9.2f}{note:<10}")
    print()

    print("== cost to MATCH G=256 quality (sweep N_out until SNR >= G=256's) ==")
    for G in GROUPS:
        if G == 256:
            print(f"  {G:<7}baseline  {bpw(256,3):.3f} bits/w   {ref:.2f} dB")
            continue
        usable = [w for w in W.values() if w.shape[1] % G == 0]
        if not usable:
            print(f"  {G:<7}N/A (no tensor divisible by G)")
            continue
        hit = None
        ladder = sorted({max(1, round(G * f / 256)) for f in
                         (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)})
        for n_out in ladder:
            s = np.mean([quant_compact_snr(w.astype(np.float32), G, n_out, rng) for w in usable])
            if s >= ref:
                hit = (n_out, bpw(G, n_out), s)
                break
        if hit:
            n_out, b, s = hit
            print(f"  {G:<7}N_out={n_out:<4} {b:.3f} bits/w   {s:.2f} dB   ({b-bpw(256,3):+.3f} bits vs G=256)")
        else:
            print(f"  {G:<7}cannot reach {ref:.2f} dB at any N_out < G/2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
