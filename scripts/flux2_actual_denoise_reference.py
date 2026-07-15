#!/usr/bin/env python3
"""Compare an actual Klein denoise trace with the vendored BFL implementation."""

import argparse
import enum
import importlib.machinery
import json
import struct
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file


MAX_ABS_LIMIT = 0.5
NRMSE_LIMIT = 0.02


def stub_broken_torchvision() -> None:
    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms", loader=None
    )

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
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms


def read_trace(path: Path) -> torch.Tensor:
    payload = path.read_bytes()
    if payload[:4] != b"HFDT":
        raise ValueError(f"{path}: invalid HFDT magic")
    rank = struct.unpack_from("<I", payload, 4)[0]
    shape = struct.unpack_from(f"<{rank}I", payload, 8)
    offset = 8 + rank * 4
    values = torch.frombuffer(bytearray(payload[offset:]), dtype=torch.float32)
    return values.reshape(shape).clone()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    rmse = delta.square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "nrmse": float(rmse / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path, help="BFL native Klein safetensors")
    parser.add_argument("trace", type=Path, help="HIPFIRE_DUMP_DENOISE_TRACE directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "third_party" / "flux2" / "src"))
    stub_broken_torchvision()
    from flux2.model import Flux2, Klein4BParams, timestep_embedding
    from flux2.sampling import batched_prc_img, batched_prc_txt, get_schedule

    torch.set_num_threads(args.threads)
    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(torch.bfloat16)
    model.load_state_dict(load_file(args.weights, device="cpu"), strict=True, assign=True)
    model = (model.float() if args.dtype == "float32" else model).eval()
    run_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    latent_files = sorted(args.trace.glob("latent_[0-9][0-9][0-9].bin"))
    if len(latent_files) < 2:
        raise ValueError("trace must contain initial and at least one updated latent")
    steps = len(latent_files) - 1
    latent = read_trace(latent_files[0])
    conditioning = read_trace(args.trace / "conditioning_positive.bin")
    image_tokens, image_ids = batched_prc_img(latent)
    text_tokens, text_ids = batched_prc_txt(conditioning)
    schedule = get_schedule(steps, image_tokens.shape[1])

    results: dict[str, object] = {
        "weights": str(args.weights.resolve()),
        "dtype": args.dtype,
        "steps": steps,
        "limits": {"max_abs": MAX_ABS_LIMIT, "nrmse": NRMSE_LIMIT},
        "trace": [],
        "stages": [],
    }
    def compare_stage(name: str, tensor: torch.Tensor) -> None:
        path = args.trace / f"{name}.bin"
        if path.is_file():
            expected = read_trace(path)
            results["stages"].append(
                {"name": name, **metrics(tensor.float(), expected)}
            )
            if name.startswith("flux2_single_"):
                results["stages"].append(
                    {
                        "name": f"{name}_image_tail",
                        **metrics(tensor[:, -1:, :].float(), expected[:, -1:, :]),
                    }
                )

    with torch.no_grad():
        for index, (t_cur, t_next) in enumerate(zip(schedule[:-1], schedule[1:]), 1):
            timestep = torch.full((latent.shape[0],), t_cur, dtype=run_dtype)
            vec = model.time_in(timestep_embedding(timestep, 256))
            image_mod = model.double_stream_modulation_img(vec)
            text_mod = model.double_stream_modulation_txt(vec)
            single_mod, _ = model.single_stream_modulation(vec)
            image = model.img_in(image_tokens.to(run_dtype))
            text = model.txt_in(text_tokens.to(run_dtype))
            compare_stage("flux2_input_image", image)
            compare_stage("flux2_input_text", text)
            image_rope = model.pe_embedder(image_ids)
            text_rope = model.pe_embedder(text_ids)
            for block_index, block in enumerate(model.double_blocks):
                image, text, _ = block.forward_kv_extract(
                    image,
                    text,
                    image_rope,
                    text_rope,
                    image_mod,
                    text_mod,
                    num_ref_tokens=0,
                )
                compare_stage(f"flux2_double_{block_index:02}_image", image)
                compare_stage(f"flux2_double_{block_index:02}_text", text)
            joint = torch.cat((text, image), dim=1)
            joint_rope = torch.cat((text_rope, image_rope), dim=2)
            for block_index, block in enumerate(model.single_blocks):
                joint, _ = block.forward_kv_extract(
                    joint,
                    joint_rope,
                    single_mod,
                    num_txt_tokens=text.shape[1],
                    num_ref_tokens=0,
                )
                compare_stage(f"flux2_single_{block_index:02}", joint)
            prediction = model.final_layer(joint[:, text.shape[1] :, :], vec).float()
            hip_joint_path = args.trace / "flux2_single_19.bin"
            if hip_joint_path.is_file():
                hip_joint = read_trace(hip_joint_path)
                hip_prediction = model.final_layer(
                    hip_joint[:, text.shape[1] :, :].to(run_dtype), vec
                ).float()
                results["stages"].append(
                    {
                        "name": "flux2_final_from_hip_hidden",
                        **metrics(
                            hip_prediction.transpose(1, 2).reshape_as(latent),
                            read_trace(args.trace / f"velocity_{index:03}.bin"),
                        ),
                    }
                )
            prediction_nchw = prediction.transpose(1, 2).reshape_as(latent)
            expected_velocity = read_trace(args.trace / f"velocity_{index:03}.bin")
            velocity_metrics = metrics(prediction_nchw, expected_velocity)
            image_tokens = image_tokens.float() + (t_next - t_cur) * prediction
            latent = image_tokens.transpose(1, 2).reshape_as(latent)
            expected_latent = read_trace(args.trace / f"latent_{index:03}.bin")
            latent_metrics = metrics(latent, expected_latent)
            results["trace"].append(
                {
                    "step": index,
                    "timestep": float(t_cur),
                    "velocity": velocity_metrics,
                    "latent": latent_metrics,
                    "reference_velocity": prediction_nchw.reshape(-1).tolist(),
                    "reference_latent": latent.reshape(-1).tolist(),
                }
            )

    failures = []
    for step in results["trace"]:
        for kind in ("velocity", "latent"):
            item = step[kind]
            if item["max_abs"] > MAX_ABS_LIMIT or item["nrmse"] > NRMSE_LIMIT:
                failures.append({"step": step["step"], "kind": kind, **item})
    results["passed"] = not failures
    results["failures"] = failures
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
