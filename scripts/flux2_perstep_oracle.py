#!/usr/bin/env python3
"""Per-step DiT-forward parity: for each denoise step k, feed hipfire's dumped
latent_00k to the reference forward at the step-k timestep and compare to
hipfire's velocity_00(k+1). Pinpoints whether the multi-step divergence is in the
forward (on structured inputs) or the integration.

  python scripts/flux2_perstep_oracle.py <trace_dir> [num_steps]
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
sys.path.insert(0, str(REPO / "third_party" / "flux2" / "src"))
from flux2.model import Flux2, Klein4BParams  # noqa: E402

W = ("/srv/huggingface/models--black-forest-labs--FLUX.2-klein-base-4B/"
     "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d/flux-2-klein-base-4b.safetensors")


def load(p):
    raw = Path(p).read_bytes(); (r,) = struct.unpack_from("<I", raw, 4)
    d = struct.unpack_from(f"<{r}I", raw, 8); off = 8 + r * 4
    return torch.from_numpy(np.frombuffer(raw, "<f4", offset=off).reshape(d).copy()).float()


def sched(steps, seq):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    m200 = a2 * seq + b2; m10 = a1 * seq + b1
    a = (m200 - m10) / 190.0; mu = a * steps + (m200 - 200.0 * a)
    ts = torch.linspace(1, 0, steps + 1)
    return (math.exp(mu) / (math.exp(mu) + (1 / ts - 1))).tolist()


def rel(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return ((a - b).norm() / (b.norm() + 1e-12)).item(), (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    D = Path(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    l0 = load(D / "latent_000.bin"); _, c, h, w = l0.shape
    ctx = load(D / "conditioning_positive.bin")
    xid = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)).unsqueeze(0).float()
    cid = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(ctx.shape[1])).unsqueeze(0).float()
    m = Flux2(Klein4BParams()).eval().float()
    miss, unexp = m.load_state_dict(load_file(W), strict=False); assert not miss and not unexp
    ts = sched(steps, h * w)
    print(f"schedule {[round(t,4) for t in ts]}")
    with torch.no_grad():
        for k in range(steps):
            latk = load(D / f"latent_{k:03d}.bin")
            velk = load(D / f"velocity_{k+1:03d}.bin")
            x = rearrange(latk, "b c h w -> b (h w) c").float()
            pred = m(x=x, x_ids=xid, timesteps=torch.full((1,), float(ts[k])),
                     ctx=ctx.float(), ctx_ids=cid, guidance=None)[:, : h * w]
            pred = rearrange(pred, "b (hw) c -> b c hw", hw=h * w).reshape(1, c, h, w)
            r, co = rel(pred, velk)
            print(f"  step {k} (t={ts[k]:.4f}): ref-forward vs hipfire velocity_{k+1:03d}  rel-L2={r:.4f} cos={co:.4f}")


if __name__ == "__main__":
    main()
