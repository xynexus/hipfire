#!/usr/bin/env python3
"""CPU oracle for the FLUX.2 DiT *forward*.

Loads the vendored BFL Flux2 model with the BFL-native klein-base-4B weights
(strict, no remapping), then runs a single step-0 forward on hipfire's dumped
inputs (init noise `latent_000`, text conditioning `conditioning_positive`) and
compares the predicted velocity to hipfire's `velocity_001`.

  match (small rel-L2)  -> hipfire's DiT forward is correct; look at the schedule
  mismatch (large)      -> hipfire's DiT forward is the bug

Because hipfire's exact step-0 timestep may differ, we evaluate the reference at
the reference flow-match schedule's timesteps and report the best match.

Usage:
  python scripts/flux2_dit_oracle.py <trace_dir>
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "flux2" / "src"))
from flux2.model import Flux2, Klein4BParams  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import math


# Inlined from flux2.sampling (avoids its module-level torchvision import).
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * num_steps + b


def get_schedule(num_steps: int, image_seq_len: int) -> list:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    ts = torch.linspace(1, 0, num_steps + 1)
    ts = math.exp(mu) / (math.exp(mu) + (1 / ts - 1) ** 1.0)
    return ts.tolist()

WEIGHTS = (
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


def img_ids(h: int, w: int) -> torch.Tensor:
    t = torch.arange(1); hh = torch.arange(h); ww = torch.arange(w); ll = torch.arange(1)
    return torch.cartesian_prod(t, hh, ww, ll).unsqueeze(0).float()  # [1, h*w, 4]


def txt_ids(seq: int) -> torch.Tensor:
    t = torch.arange(1); hh = torch.arange(1); ww = torch.arange(1); ll = torch.arange(seq)
    return torch.cartesian_prod(t, hh, ww, ll).unsqueeze(0).float()  # [1, seq, 4]


def rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    r = (a - b).norm().item() / (b.norm().item() + 1e-12)
    c = (a @ b).item() / (a.norm().item() * b.norm().item() + 1e-12)
    return r, c


def main() -> int:
    D = Path(sys.argv[1])
    lat = load_hfdt(D / "latent_000.bin")            # [1,128,h,w]
    ctx = load_hfdt(D / "conditioning_positive.bin") # [1,txt,7680]
    hf_vel = load_hfdt(D / "velocity_001.bin")       # [1,128,h,w]
    _, c, h, w = lat.shape
    x = rearrange(lat, "b c h w -> b (h w) c")        # token = y*w + x
    xid = img_ids(h, w)
    cid = txt_ids(ctx.shape[1])
    print(f"latent {tuple(lat.shape)}  ctx {tuple(ctx.shape)}  img_tokens {h*w}")

    model = Flux2(Klein4BParams()).eval()
    sd = load_file(WEIGHTS)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (len(miss), len(unexp))
    model = model.float()

    sched = get_schedule(num_steps=4, image_seq_len=h * w)
    print(f"reference schedule (mu-shift): {[round(t,4) for t in sched]}")

    best = None
    with torch.no_grad():
        for i, tval in enumerate(sched[:-1]):  # candidate step-0 timesteps
            tvec = torch.full((1,), float(tval))
            pred = model(x=x.float(), x_ids=xid, timesteps=tvec,
                         ctx=ctx.float(), ctx_ids=cid, guidance=None)  # [1, h*w, 128]
            pred = rearrange(pred[:, : h * w], "b (h w) c -> b c h w", h=h, w=w)
            r, co = rel(pred, hf_vel)
            print(f"  t={tval:.4f}  vs hipfire velocity_001:  rel-L2={r:.4f}  cos={co:.4f}")
            if best is None or r < best[0]:
                best = (r, co, tval, pred)

    r, co, tval, pred = best
    print(f"\nBEST MATCH: t={tval:.4f}  rel-L2={r:.4f}  cos={co:.4f}")
    if r < 0.15:
        print("=> hipfire DiT forward MATCHES the reference (forward correct; suspect schedule/VAE-order).")
    elif co > 0.9:
        print("=> close direction but scaled/shifted — likely a timestep/scale mismatch, not a structural forward bug.")
    else:
        print("=> hipfire DiT forward DIVERGES from the reference — the forward is the bug.")
    # magnitudes for context
    print(f"   |hipfire vel|={hf_vel.norm():.2f}  |ref vel|={pred.norm():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
