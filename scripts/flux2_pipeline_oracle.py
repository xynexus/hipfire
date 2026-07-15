#!/usr/bin/env python3
"""End-to-end reference oracle: run the FULL BFL FLUX.2 pipeline (DiT denoise +
VAE decode, all reference torch) from hipfire's dumped init noise + conditioning,
and compare to hipfire.

If the reference pipeline also produces a fragmented image from the same inputs,
the fragmentation is genuine klein-base-4B behavior at the given step count — not
a hipfire bug. If the reference is clean, hipfire diverges somewhere.

  python scripts/flux2_pipeline_oracle.py <trace_dir> <out.png> [num_steps]
"""

import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "flux2" / "src"))
from flux2.model import Flux2, Klein4BParams  # noqa: E402
from flux2_vae_oracle import build_reference_vae, to_png, DEFAULT_VAE  # noqa: E402

DIT_WEIGHTS = (
    "/srv/huggingface/models--black-forest-labs--FLUX.2-klein-base-4B/"
    "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d/flux-2-klein-base-4b.safetensors"
)


def load_hfdt(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    assert raw[:4] == b"HFDT", path
    (rank,) = struct.unpack_from("<I", raw, 4)
    dims = struct.unpack_from(f"<{rank}I", raw, 8)
    off = 8 + rank * 4
    return torch.from_numpy(np.frombuffer(raw, "<f4", offset=off).reshape(dims).copy()).float()


def compute_empirical_mu(seq: int, steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq > 4300:
        return a2 * seq + b2
    m200 = a2 * seq + b2
    m10 = a1 * seq + b1
    a = (m200 - m10) / 190.0
    return a * steps + (m200 - 200.0 * a)


def get_schedule(steps: int, seq: int) -> list:
    mu = compute_empirical_mu(seq, steps)
    ts = torch.linspace(1, 0, steps + 1)
    return (math.exp(mu) / (math.exp(mu) + (1 / ts - 1))).tolist()


def ids(dims, seq_axis) -> torch.Tensor:
    axes = [torch.arange(d) if i == seq_axis else torch.arange(1) for i, d in enumerate(dims)]
    return torch.cartesian_prod(*axes).unsqueeze(0).float()


def main() -> int:
    D = Path(sys.argv[1]); out = Path(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lat = load_hfdt(D / "latent_000.bin")             # [1,128,h,w]
    ctx = load_hfdt(D / "conditioning_positive.bin")  # [1,txt,7680]
    _, c, h, w = lat.shape
    xid = ids((1, h, w, 1), 1)  # img: [0,y,x,0], h-major -> token=y*w+x
    # (cartesian over t,h,w,l with h,w varying gives h-major; w inner)
    xid = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)).unsqueeze(0).float()
    cid = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(ctx.shape[1])).unsqueeze(0).float()

    model = Flux2(Klein4BParams()).eval().float()
    sd = load_file(DIT_WEIGHTS)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    sched = get_schedule(steps, h * w)
    print(f"steps={steps}  schedule={[round(t,4) for t in sched]}")

    # Optional CFG (base model wants cfg=4): needs the unconditional embedding.
    cfg = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    neg = None
    if cfg != 1.0:
        neg = load_hfdt(D / "conditioning_negative.bin").float()
    print(f"cfg={cfg}")

    x = rearrange(lat, "b c h w -> b (h w) c").float()
    with torch.no_grad():
        for k in range(steps):
            t_cur, t_next = sched[k], sched[k + 1]
            tv = torch.full((1,), float(t_cur))
            cond = model(x=x, x_ids=xid, timesteps=tv, ctx=ctx.float(), ctx_ids=cid, guidance=None)[:, : h * w]
            if neg is not None:
                ncid = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(neg.shape[1])).unsqueeze(0).float()
                uncond = model(x=x, x_ids=xid, timesteps=tv, ctx=neg, ctx_ids=ncid, guidance=None)[:, : h * w]
                pred = uncond + cfg * (cond - uncond)
            else:
                pred = cond
            x = x + (t_next - t_cur) * pred
            print(f"  step {k}: t {t_cur:.4f}->{t_next:.4f}  |x|={x.norm():.1f}")
    final = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

    # compare to hipfire's final latent if present
    hf_final = D / f"latent_{steps:03d}.bin"
    if hf_final.exists():
        hf = load_hfdt(hf_final)
        r = (final - hf).reshape(-1).double().norm().item() / (hf.reshape(-1).double().norm().item() + 1e-12)
        print(f"reference final latent vs hipfire latent_{steps:03d}: rel-L2={r:.4f}")

    vae = build_reference_vae(Path(DEFAULT_VAE))
    with torch.no_grad():
        decoded = vae.decode(final)
    to_png(decoded, out)
    print(f"wrote reference pipeline image -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
