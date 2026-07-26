#!/usr/bin/env python3
"""Emit a deterministic tiny BFL FLUX.2 checkpoint and CPU parity trace."""

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


def tensor_values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "third_party" / "flux2" / "src"))

    # sampling.py only needs torchvision in image preprocessing helpers; the
    # schedule reference itself is text/tensor-only and this host's torchvision
    # build is intentionally not part of the parity environment.
    sys.modules["torchvision"] = types.ModuleType("torchvision")

    from flux2.model import Flux2, Flux2Params, timestep_embedding
    from flux2.sampling import get_schedule

    torch.manual_seed(0xF102)
    params = Flux2Params(
        in_channels=8,
        context_in_dim=6,
        hidden_size=16,
        num_heads=1,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4],
        theta=2000,
        mlp_ratio=3.0,
        use_guidance_embed=False,
    )
    model = Flux2(params).eval()
    x = torch.tensor(
        [[
            [0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.25],
            [-0.75, 0.375, 0.5, -0.125, 1.25, -1.0, 0.625, 0.25],
        ]],
        dtype=torch.float32,
    )
    ctx = torch.tensor(
        [[
            [0.5, -0.25, 0.75, 0.125, -0.5, 1.0],
            [-0.375, 0.625, -0.125, 0.875, 0.25, -0.75],
        ]],
        dtype=torch.float32,
    )
    x_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 0]]], dtype=torch.float32)
    ctx_ids = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32)
    normalized_timestep = torch.tensor([0.25], dtype=torch.float32)

    with torch.no_grad():
        vec = model.time_in(timestep_embedding(normalized_timestep, 256))
        image_mod = model.double_stream_modulation_img(vec)
        text_mod = model.double_stream_modulation_txt(vec)
        single_mod, _ = model.single_stream_modulation(vec)
        image = model.img_in(x)
        text = model.txt_in(ctx)
        image_rope = model.pe_embedder(x_ids)
        text_rope = model.pe_embedder(ctx_ids)
        image, text, _ = model.double_blocks[0].forward_kv_extract(
            image,
            text,
            image_rope,
            text_rope,
            image_mod,
            text_mod,
            num_ref_tokens=0,
        )
        double_image = image.clone()
        double_text = text.clone()
        joint = torch.cat((text, image), dim=1)
        rope = torch.cat((text_rope, image_rope), dim=2)
        joint, _ = model.single_blocks[0].forward_kv_extract(
            joint,
            rope,
            single_mod,
            num_txt_tokens=ctx.shape[1],
            num_ref_tokens=0,
        )
        single_joint = joint.clone()
        output = model.final_layer(joint[:, ctx.shape[1] :, :], vec)
        direct = model(x, x_ids, normalized_timestep, ctx, ctx_ids, None)
        torch.testing.assert_close(output, direct)

    for component in ["text_encoder", "transformer", "vae", "scheduler", "tokenizer"]:
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    (root / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2Transformer2DModel",
                "in_channels": 8,
                "out_channels": 8,
                "patch_size": 1,
                "num_layers": 1,
                "num_single_layers": 1,
                "num_attention_heads": 1,
                "joint_attention_dim": 6,
                "axes_dims_rope": [4, 4, 4, 4],
                "rope_theta": 2000,
                "guidance_embeds": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "text_encoder" / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2}),
        encoding="utf-8",
    )
    (root / "vae" / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLFlux2", "latent_channels": 2}),
        encoding="utf-8",
    )
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}),
        encoding="utf-8",
    )
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        root / "flux-2-tiny.safetensors",
    )
    (root / "reference.json").write_text(
        json.dumps(
            {
                "x": tensor_values(x),
                "ctx": tensor_values(ctx),
                "scheduler_timestep": 250.0,
                "vec": tensor_values(vec),
                "double_image": tensor_values(double_image),
                "double_text": tensor_values(double_text),
                "single_joint": tensor_values(single_joint),
                "output": tensor_values(output),
                "schedule_50_seq1024": get_schedule(50, 1024),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
