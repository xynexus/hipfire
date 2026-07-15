#!/usr/bin/env python3
"""Emit SeFi dual-time schedules from the pinned Diffusers scheduler."""

import argparse
import enum
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch


def stub_broken_torchvision() -> None:
    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms", loader=None
    )
    io = types.ModuleType("torchvision.io")
    io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 2
        BILINEAR = 3
        HAMMING = 4
        BICUBIC = 5
        LANCZOS = 6

    transforms.InterpolationMode = InterpolationMode
    torchvision.transforms = transforms
    torchvision.io = io
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.io"] = io


def shifted(u: torch.Tensor, alpha: float) -> torch.Tensor:
    return (alpha * u) / (1.0 + (alpha - 1.0) * u)


def schedule(scheduler, count: int, delta_t: float, alpha: float) -> dict:
    u_base = torch.linspace(0.0, 1.0, count + 1, dtype=torch.float32)
    u_shifted = shifted(u_base, alpha)
    u_sem_raw = u_shifted * (1.0 + delta_t)
    u_sem = u_sem_raw.clamp(max=1.0)
    u_tex = (u_sem_raw - delta_t).clamp(min=0.0, max=1.0)

    def lookup(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = (u * (scheduler.config.num_train_timesteps - 1)).long().clamp(
            0, scheduler.config.num_train_timesteps - 1
        )
        return scheduler.timesteps[indices], scheduler.sigmas[indices]

    timestep_sem, sigma_sem = lookup(u_sem)
    timestep_tex, sigma_tex = lookup(u_tex)
    _, base_sigma = lookup(u_shifted)
    # Deterministic tiny loop oracle. Its velocity depends on the current latent,
    # channel class and both timesteps, so every split/update is independently
    # observable for all supported distilled step counts.
    latents = torch.tensor(
        [[[[0.25, -0.5]], [[0.75, 1.0]], [[-1.25, 0.5]]]],
        dtype=torch.float32,
    )
    trace = []
    for step in range(count):
        velocity = torch.empty_like(latents)
        for channel in range(latents.shape[1]):
            timestep = timestep_sem[step] if channel == 0 else timestep_tex[step]
            for element in range(latents.shape[-1]):
                velocity[0, channel, 0, element] = (
                    latents[0, channel, 0, element] * 0.125
                    + (step + 1) * 0.01
                    + (channel * latents.shape[-1] + element) * 0.05
                    + timestep * 1e-5
                )
        latents[:, :1] += (sigma_sem[step + 1] - sigma_sem[step]) * velocity[:, :1]
        latents[:, 1:] += (sigma_tex[step + 1] - sigma_tex[step]) * velocity[:, 1:]
        trace.append(
            {
                "velocity": velocity.reshape(-1).tolist(),
                "latent": latents.reshape(-1).tolist(),
            }
        )
    return {
        "base_sigmas": base_sigma.tolist(),
        "timestep_sem": timestep_sem[:-1].tolist(),
        "timestep_tex": timestep_tex[:-1].tolist(),
        "sigma_sem": sigma_sem.tolist(),
        "sigma_tex": sigma_tex.tolist(),
        "integration_initial": [0.25, -0.5, 0.75, 1.0, -1.25, 0.5],
        "integration_trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    stub_broken_torchvision()
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.snapshot, subfolder="scheduler"
    )
    payload = {
        str(count): schedule(scheduler, count, args.delta_t, args.alpha)
        for count in (4, 8, 10)
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
