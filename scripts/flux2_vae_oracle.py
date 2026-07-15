#!/usr/bin/env python3
"""CPU oracle: decode a hipfire-dumped DiT latent through the *reference* BFL
FLUX.2 VAE, to ground-truth whether the txt2img corruption is in hipfire's
DiT-generated latent or in hipfire's VAE decode.

Reuses the diffusers->BFL weight mapping from flux2_vae_reference.py, loads the
diffusers VAE safetensors, reads a hipfire HFDT latent (`latent_004.bin`, the
final packed [1,128,h,w] latent), and decodes it to a PNG on CPU torch.

  reference-VAE output CLEAN  -> hipfire's VAE decode is the bug
  reference-VAE output FRAGMENTED -> hipfire's DiT latent is genuinely bad (DiT-forward bug)

Usage:
  python scripts/flux2_vae_oracle.py <latent.bin> <out.png> [--vae <safetensors>]
"""

import argparse
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
from flux2_vae_reference import diffusers_decoder_to_bfl  # noqa: E402

DEFAULT_VAE = (
    "/srv/huggingface/models--black-forest-labs--FLUX.2-klein-base-4B/"
    "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d/vae/"
    "diffusion_pytorch_model.safetensors"
)


def load_hfdt(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if raw[:4] != b"HFDT":
        raise ValueError(f"{path}: not an HFDT tensor")
    (rank,) = struct.unpack_from("<I", raw, 4)
    dims = struct.unpack_from(f"<{rank}I", raw, 8)
    off = 8 + rank * 4
    arr = np.frombuffer(raw, dtype="<f4", offset=off).reshape(dims).copy()
    return torch.from_numpy(arr).float()


def build_reference_vae(vae_path: Path):
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    model = AutoEncoder(AutoEncoderParams()).eval().float()
    source = load_file(str(vae_path))
    mapped = {}
    for name, tensor in source.items():
        mapped_name = diffusers_decoder_to_bfl(name)
        if mapped_name is None:
            continue
        value = tensor.float()
        if (
            mapped_name.startswith("decoder.mid.attn_1.")
            and mapped_name.endswith(".weight")
            and value.ndim == 2
        ):
            value = value[:, :, None, None]
        mapped[mapped_name] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    dec_missing = [n for n in missing if n.startswith("decoder.") or n.startswith("bn.")]
    if dec_missing or unexpected:
        raise RuntimeError(f"VAE mapping incomplete: missing={dec_missing}, unexpected={unexpected}")
    return model


def to_png(decoded: torch.Tensor, path: Path):
    # decoded [1,3,H,W] in ~[-1,1]  ->  RGB u8
    from PIL import Image

    img = decoded[0].detach().cpu().numpy()
    img = np.clip((img.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("latent", type=Path, help="hipfire HFDT latent (packed [1,128,h,w])")
    ap.add_argument("out", type=Path, help="output PNG")
    ap.add_argument("--vae", type=Path, default=Path(DEFAULT_VAE))
    args = ap.parse_args()

    latent = load_hfdt(args.latent)
    print(f"latent shape {tuple(latent.shape)}  mean={latent.mean():+.3f} std={latent.std():.3f}")
    model = build_reference_vae(args.vae)
    with torch.no_grad():
        # model.decode does inv_normalize -> unpatchify -> conv decoder internally.
        decoded = model.decode(latent)
    print(f"decoded shape {tuple(decoded.shape)}  range [{decoded.min():.2f}, {decoded.max():.2f}]")
    to_png(decoded, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
