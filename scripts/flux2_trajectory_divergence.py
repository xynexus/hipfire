#!/usr/bin/env python3
"""Confirm whether a quantized FLUX.2 model's rendered-image drift is trajectory
divergence (sampler chaos) rather than weight corruption.

It reads two `HIPFIRE_DUMP_DENOISE_TRACE` directories -- one from the reference
(bf16 / p0) run and one from the candidate (e.g. oq8) run, generated with the
SAME prompt/seed/steps -- and compares the dumped tensors step by step.

Trace format (`HFDT`, written by dump_denoise_trace_tensor in denoise.rs):
    b"HFDT" | u32 rank | rank x u32 dims | f32 data      (all little-endian)
Files: latent_000.bin, conditioning_positive.bin, conditioning_negative.bin,
       velocity_001.bin, latent_001.bin, velocity_002.bin, ...

Logic:
  * latent_000 and conditioning_* MUST match between runs (same seed + bf16 text
    encoder). These are the controls -- if they differ, the premise is wrong.
  * velocity_001 is the FIRST model prediction on identical inputs, so its error
    is the pure numerical effect of quantization, before any trajectory
    divergence has compounded. Small here == faithful quant.
  * latent_NNN divergence growing across steps == chaotic amplification: the same
    small perturbation steers the flow to a different-but-valid mode. This is what
    kills exact pixel comparison of the final image.

Usage:
    python scripts/flux2_trajectory_divergence.py <ref_trace_dir> <cand_trace_dir>
    python scripts/flux2_trajectory_divergence.py ref/ cand/ --vel-threshold 0.02
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_hfdt(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:4] != b"HFDT":
        raise ValueError(f"{path}: not an HFDT tensor (bad magic {raw[:4]!r})")
    (rank,) = struct.unpack_from("<I", raw, 4)
    dims = struct.unpack_from(f"<{rank}I", raw, 8)
    offset = 8 + rank * 4
    data = np.frombuffer(raw, dtype="<f4", offset=offset)
    return data.reshape(dims)


def load_opt(d: Path, name: str):
    p = d / f"{name}.bin"
    return load_hfdt(p) if p.exists() else None


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    diff = a - b
    ref_rms = float(np.sqrt(np.mean(a * a)))
    rms = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms": rms,
        "rel_rms": (rms / ref_rms) if ref_rms > 0 else 0.0,
        "cos": cos,
    }


def detect_steps(d: Path) -> int:
    return len(sorted(d.glob("velocity_*.bin")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir", type=Path, help="reference (bf16/p0) trace dir")
    ap.add_argument("cand_dir", type=Path, help="candidate (quantized) trace dir")
    ap.add_argument("--vel-threshold", type=float, default=0.02,
                    help="rel-L2 on step-1 velocity below which quant is 'faithful' (default 0.02)")
    ap.add_argument("--control-eps", type=float, default=1e-6,
                    help="max abs diff allowed on the identical-input controls (default 1e-6)")
    args = ap.parse_args()

    ref, cand = args.ref_dir, args.cand_dir
    if not ref.is_dir() or not cand.is_dir():
        print(f"error: both trace dirs must exist ({ref}, {cand})", file=sys.stderr)
        return 2

    # --- Controls: identical inputs must match ---
    print("== controls (must match: same seed + bf16 text encoder) ==")
    control_ok = True
    for name in ("latent_000", "conditioning_positive", "conditioning_negative"):
        a, b = load_opt(ref, name), load_opt(cand, name)
        if a is None or b is None:
            print(f"  {name:24s} : MISSING (ref={a is not None}, cand={b is not None})")
            continue
        if a.shape != b.shape:
            print(f"  {name:24s} : SHAPE MISMATCH {a.shape} vs {b.shape}")
            control_ok = False
            continue
        s = stats(a, b)
        flag = "ok" if s["max_abs"] <= args.control_eps else "DIFFERS"
        if flag != "ok":
            control_ok = False
        print(f"  {name:24s} : max_abs={s['max_abs']:.3e}  [{flag}]")
    if not control_ok:
        print("  !! controls differ -- inputs are NOT identical; premise broken.")
        print("     (if conditioning_* differs, the text encoder path changed too.)")

    # --- Per-step divergence ---
    nsteps = min(detect_steps(ref), detect_steps(cand))
    if nsteps == 0:
        print("\nerror: no velocity_*.bin found in one/both dirs; run txt2img with "
              "HIPFIRE_DUMP_DENOISE_TRACE=<dir>", file=sys.stderr)
        return 2

    print(f"\n== per-step divergence ({nsteps} steps) ==")
    hdr = f"  {'step':>4}  {'vel rel_L2':>11}  {'vel cos':>9}  {'lat rel_L2':>11}  {'lat cos':>9}  {'lat max_abs':>12}"
    print(hdr)
    vel1_rel = None
    lat_final_rel = 0.0
    for step in range(1, nsteps + 1):
        vs = stats(load_hfdt(ref / f"velocity_{step:03d}.bin"),
                   load_hfdt(cand / f"velocity_{step:03d}.bin"))
        la = load_opt(ref, f"latent_{step:03d}")
        lb = load_opt(cand, f"latent_{step:03d}")
        ls = stats(la, lb) if (la is not None and lb is not None) else None
        if step == 1:
            vel1_rel = vs["rel_rms"]
        if ls is not None:
            lat_final_rel = ls["rel_rms"]
        lat_cols = (f"{ls['rel_rms']:>11.5f}  {ls['cos']:>9.5f}  {ls['max_abs']:>12.4f}"
                    if ls is not None else f"{'--':>11}  {'--':>9}  {'--':>12}")
        print(f"  {step:>4}  {vs['rel_rms']:>11.5f}  {vs['cos']:>9.5f}  {lat_cols}")

    # --- Verdict ---
    print("\n== verdict ==")
    if vel1_rel is None:
        print("  inconclusive: no step-1 velocity.")
        return 1
    growth = (lat_final_rel / vel1_rel) if vel1_rel > 0 else float("inf")
    print(f"  step-1 velocity rel-L2 : {vel1_rel:.5f}  (threshold {args.vel_threshold})")
    print(f"  final latent  rel-L2   : {lat_final_rel:.5f}")
    if vel1_rel <= args.vel_threshold and lat_final_rel > vel1_rel * 3:
        print("  => TRAJECTORY DIVERGENCE CONFIRMED: the quantized model predicts nearly")
        print("     the same velocity on identical inputs, but the trajectory amplifies")
        print(f"     that tiny difference ~{growth:.1f}x by the final step. The image drift")
        print("     is sampler chaos, not weight corruption. Do NOT gate on exact pixels.")
        return 0
    if vel1_rel > args.vel_threshold:
        print("  => STEP-1 VELOCITY ALREADY DIVERGES: quantization is materially changing")
        print("     the model's prediction on identical inputs. This is (at least partly)")
        print("     a quant-fidelity problem, not pure chaos. Cross-check with")
        print("     `hipfire diffusion quant-diff` to find the offending tensor(s).")
        return 1
    print("  => inconclusive: small step-1 velocity error but little downstream growth.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
