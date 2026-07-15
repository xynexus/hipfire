#!/usr/bin/env python3
"""Run the vendored BFL FLUX.2 VAE decoder on a deterministic tiny latent."""

import argparse
import json
import sys
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file


def diffusers_decoder_to_bfl(name: str) -> str | None:
    if name.startswith("bn."):
        return name
    if name.startswith("post_quant_conv."):
        return "decoder." + name
    if not name.startswith("decoder."):
        return None
    rest = name.removeprefix("decoder.")
    if rest.startswith("mid_block.resnets.0."):
        return "decoder.mid.block_1." + rest.removeprefix("mid_block.resnets.0.")
    if rest.startswith("mid_block.resnets.1."):
        return "decoder.mid.block_2." + rest.removeprefix("mid_block.resnets.1.")
    if rest.startswith("mid_block.attentions.0."):
        suffix = rest.removeprefix("mid_block.attentions.0.")
        replacements = {
            "group_norm.": "norm.",
            "to_q.": "q.",
            "to_k.": "k.",
            "to_v.": "v.",
            "to_out.0.": "proj_out.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return "decoder.mid.attn_1." + target + suffix.removeprefix(source)
        return None
    if rest.startswith("up_blocks."):
        _, block, tail = rest.split(".", 2)
        bfl_block = 3 - int(block)
        if tail.startswith("resnets."):
            _, layer, suffix = tail.split(".", 2)
            suffix = suffix.replace("conv_shortcut.", "nin_shortcut.", 1)
            return f"decoder.up.{bfl_block}.block.{layer}.{suffix}"
        if tail.startswith("upsamplers.0.conv."):
            return (
                f"decoder.up.{bfl_block}.upsample.conv."
                + tail.removeprefix("upsamplers.0.conv.")
            )
        return None
    if rest.startswith("conv_norm_out."):
        return "decoder.norm_out." + rest.removeprefix("conv_norm_out.")
    return "decoder." + rest


def flattened(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vae", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "third_party" / "flux2" / "src"))
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    model = AutoEncoder(AutoEncoderParams(resolution=16)).eval().float()
    source = load_file(args.vae)
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
            # Diffusers stores the mid-block attention projections as Linear
            # weights while the BFL reference expresses the same operation as
            # a 1x1 Conv2d.
            value = value[:, :, None, None]
        mapped[mapped_name] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    decoder_missing = [name for name in missing if name.startswith("decoder.") or name.startswith("bn.")]
    if decoder_missing or unexpected:
        raise RuntimeError(f"VAE mapping incomplete: missing={decoder_missing}, unexpected={unexpected}")
    latent = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(1, 128, 1, 1)
    with torch.no_grad():
        denormalized = model.inv_normalize(latent)
        unpatchified = rearrange(
            denormalized,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=2,
            pj=2,
        )
        decoded = model.decode(latent)
    args.output.write_text(
        json.dumps(
            {
                "latent": flattened(latent),
                "unpatchified": flattened(unpatchified),
                "decoded": flattened(decoded),
                "decoded_shape": list(decoded.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
