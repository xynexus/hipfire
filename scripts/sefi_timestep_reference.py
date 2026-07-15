#!/usr/bin/env python3
"""Run SeFi's dual timestep embedding from the real checkpoint weights."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--semantic", type=float, default=726.0)
    parser.add_argument("--texture", type=float, default=826.0)
    args = parser.parse_args()
    weights = {}
    wanted = {
        f"dual_time_embed.{stream}_embedder.linear_{layer}.weight"
        for stream in ("semantic", "texture")
        for layer in (1, 2)
    }
    for shard in sorted((args.snapshot / "transformer").glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in wanted.intersection(handle.keys()):
                weights[name] = handle.get_tensor(name).float()
    missing = wanted.difference(weights)
    if missing:
        raise RuntimeError(f"missing dual-time tensors: {sorted(missing)}")
    time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)

    def branch(stream: str, timestep: float) -> torch.Tensor:
        projected = time_proj(torch.tensor([timestep], dtype=torch.float32))
        hidden = F.linear(
            projected,
            weights[f"dual_time_embed.{stream}_embedder.linear_1.weight"],
        )
        hidden = F.silu(hidden)
        return F.linear(
            hidden,
            weights[f"dual_time_embed.{stream}_embedder.linear_2.weight"],
        )

    with torch.no_grad():
        semantic = branch("semantic", args.semantic)
        texture = branch("texture", args.texture)
        output = torch.cat([semantic, texture], dim=-1)
    args.output.write_text(
        json.dumps(
            {
                "semantic_timestep": args.semantic,
                "texture_timestep": args.texture,
                "shape": list(output.shape),
                "output": output.reshape(-1).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
