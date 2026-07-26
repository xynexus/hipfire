#!/usr/bin/env python3
"""Survey model architectures by *declared structure* rather than by name.

Diagnostics for `docs/plans/2026-07-26-arch-identity-structure-descriptor.md`
(phase 0). Extracts a structure descriptor from every model config under a
HuggingFace root, canonicalizes it, and counts distinct structures — the input
to deciding whether structure-based identity can replace the numeric `arch_id`.

Run it twice. The difference between the two runs is the finding:

    python3 scripts/arch_structure_survey.py                  # config.json only
    python3 scripts/arch_structure_survey.py --with-manifest  # + tensor manifest

Config alone reports collapses that do not exist: Qwen2's QKV bias and Qwen3's
QK-norm are properties of the HF *implementation*, absent from `config.json`,
and present only as tensors. Reads only; writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

DEFAULT_ROOT = "/srv/huggingface"

# Repos that are not decoder language models. Encoders, ONNX exports and
# sparse-autoencoder probes have no stack to describe.
SKIP_TYPES = {"onnx", "custom", "stt", "topk_sae", "siglip", "clip", "?"}

# `layer_types` value space, which differs per vendor for the same concept.
LAYER_MIXER = {
    "full_attention": "attn",
    "sliding_attention": "attn",
    "linear_attention": "linear",
    "conv": "short_conv",
    "recurrent": "linear",
    "mamba": "mamba2",
}
# nemotron_h `hybrid_override_pattern` chars.
BLOCK_CHAR = {"*": "attn", "M": "mamba2", "-": "ffn.dense", "E": "ffn.moe"}


# --------------------------------------------------------------- discovery
def find_configs(root: str) -> dict[str, str]:
    """One config.json per repo, HF-cache layout or a plain directory."""
    out: dict[str, str] = {}
    for entry in sorted(os.listdir(root)):
        if entry.startswith("datasets--") or entry in (
            "CACHEDIR.TAG",
            "System Volume Information",
        ):
            continue
        base = os.path.join(root, entry)
        if not os.path.isdir(base):
            continue
        cands = []
        snaps = os.path.join(base, "snapshots")
        if os.path.isdir(snaps):
            cands += [
                os.path.join(snaps, sha, "config.json")
                for sha in sorted(os.listdir(snaps))
            ]
        cands.append(os.path.join(base, "config.json"))
        for c in cands:
            if os.path.isfile(c):
                out[entry] = c
                break
    return out


def load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:  # unreadable/corrupt configs are data, not crashes
        return {"__error__": str(exc)}


def text_cfg(cfg: dict) -> dict:
    """Multimodal wrappers nest the decoder; find the stack that has layers."""
    for key in ("text_config", "llm_config", "language_config", "decoder_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict) and nested.get("num_hidden_layers"):
            return nested
    return cfg


def g(cfg: dict, *names, default=None):
    """First present, non-null value among `names`."""
    for n in names:
        if cfg.get(n) is not None:
            return cfg[n]
    return default


# ------------------------------------------------------------- config axes
def base_attn(t: dict) -> str:
    if g(t, "q_lora_rank", "kv_lora_rank") is not None:
        return "mla"
    kv, heads = g(t, "num_key_value_heads"), g(t, "num_attention_heads")
    if kv is not None and heads is not None and kv != heads:
        return "gqa"
    return "mha" if heads else "unknown"


def sliding_active(t: dict) -> bool:
    # Presence of a field is not activation of a feature: Qwen2.5 ships
    # `sliding_window` alongside `use_sliding_window: false`.
    if g(t, "use_sliding_window") is False:
        return False
    return bool(g(t, "sliding_window"))


def mixers(t: dict) -> list[str]:
    """The SET of sequence mixers across the stack. A third of real models are
    hybrid, so a single value cannot describe them."""
    pattern = g(t, "hybrid_override_pattern")
    if pattern:
        got = {BLOCK_CHAR.get(c) for c in str(pattern)}
        return sorted(
            {base_attn(t) if m == "attn" else m
             for m in got if m and not m.startswith("ffn")}
        )
    layer_types = g(t, "layer_types")
    if isinstance(layer_types, list) and layer_types:
        got = set()
        for x in layer_types:
            m = LAYER_MIXER.get(str(x), str(x))
            got.add(base_attn(t) if m == "attn" else m)
        return sorted(got)
    return [base_attn(t)]


def mixer_type(t: dict) -> str:
    layer_types = g(t, "layer_types")
    if isinstance(layer_types, list) and any("slid" in str(x) for x in layer_types):
        return "sliding_periodic"
    if g(t, "sliding_window_pattern"):
        return "sliding_periodic"
    return "sliding" if sliding_active(t) else "generic"


def ffn(t: dict) -> tuple[str, str]:
    experts = g(t, "num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
    if not experts:
        act = str(g(t, "hidden_act", "hidden_activation", default="silu")).lower()
        if "relu2" in act or "relu_sq" in act:
            return "dense", "relu2"
        return "dense", ("gelu" if "gelu" in act else "swiglu")
    shared = g(t, "shared_expert_intermediate_size", "n_shared_experts", "num_shared_experts")
    return "moe", ("shared+routed" if shared else "routed")


def rope(t: dict) -> str:
    if g(t, "partial_rotary_factor") is not None:
        return "partial"
    # `rope_parameters` is the current HF spelling; older configs use
    # `rope_theta` / `rope_scaling`. Both are live in the wild.
    params = g(t, "rope_parameters")
    if isinstance(params, dict):
        kind = params.get("rope_type") or params.get("type")
        return str(kind).lower() if kind and kind != "default" else "linear"
    scaling = g(t, "rope_scaling")
    if isinstance(scaling, dict):
        return str(scaling.get("rope_type") or scaling.get("type") or "scaled").lower()
    if g(t, "rope_local_base_freq") is not None:
        return "dual"
    return "linear" if g(t, "rope_theta") is not None else "none"


def pattern_enc(t: dict) -> str:
    """Which of the four in-the-wild encodings this config uses. A canonical
    descriptor must normalize all of them to one form."""
    if g(t, "hybrid_override_pattern"):
        return "explicit_blocks"
    layer_types = g(t, "layer_types")
    if isinstance(layer_types, list) and layer_types:
        return "uniform" if len(set(map(str, layer_types))) == 1 else "explicit_list"
    return "period" if g(t, "sliding_window_pattern") else "uniform"


def towers(cfg: dict, t: dict) -> str:
    out = []
    if isinstance(cfg.get("vision_config"), dict):
        out.append("vision")
    if isinstance(cfg.get("audio_config"), dict):
        out.append("audio")
    if g(t, "num_nextn_predict_layers", "n_future_tokens"):
        out.append("mtp")
    if isinstance(cfg.get("dflash_config"), dict) or g(cfg, "num_target_layers"):
        out.append("draft")
    return "+".join(out) or "-"


# ----------------------------------------------------------- manifest axes
def manifest_keys(cfg_path: str) -> set[str]:
    """Tensor names from the safetensors index, or a single-file header."""
    d = os.path.dirname(cfg_path)
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            return set(json.load(open(index))["weight_map"])
        except Exception:
            return set()
    single = os.path.join(d, "model.safetensors")
    if os.path.isfile(single):
        try:
            with open(single, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                return set(json.loads(f.read(n))) - {"__metadata__"}
        except Exception:
            return set()
    return set()


def decoder_keys(keys: set[str]) -> set[str]:
    """Drop tower tensors. gemma3 carries 81 QKV-bias tensors that are 100%
    siglip vision encoder and 0% decoder; attributing them to the stack
    invents a structural difference that is not there."""
    drop = ("vision_tower", "vision_model", "visual.", "audio_tower", "audio_model")
    return {k for k in keys if not any(d in k for d in drop)}


def tensor_facts(keys: set[str]) -> dict | None:
    """Structural facts that only the weights reveal."""
    if not keys:
        return None
    dec = decoder_keys(keys)
    blob = "\n".join(dec)
    return {
        "qkv_bias": any(k.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")) for k in dec),
        "qk_norm": "q_norm" in blob or "k_norm" in blob,
        "o_bias": any(k.endswith("o_proj.bias") for k in dec),
        "mlp_bias": any(k.endswith(("gate_proj.bias", "down_proj.bias", "up_proj.bias")) for k in dec),
        "expert_layout": "per_expert" if ".experts." in blob else ("stacked" if "experts" in blob else "-"),
        "shared_expert": "shared_expert" in blob,
        "ssm": "A_log" in blob or "dt_bias" in blob,
        "conv1d": "conv1d" in blob,
        "vision": any("vision" in k or "visual." in k for k in keys),
        "mtp": "nextn" in blob or "mtp" in blob.lower(),
    }


# ----------------------------------------------------------------- descriptor
def descriptor(cfg: dict, facts: dict | None) -> dict:
    t = text_cfg(cfg)
    kind, ftype = ffn(t)
    d = {
        "mixers": "+".join(mixers(t)),
        "mixer.type": mixer_type(t),
        "ffn.kind": kind,
        "ffn.type": ftype,
        "rope": rope(t),
        "pattern": pattern_enc(t),
        "towers": towers(cfg, t),
    }
    if facts is None:
        # Config-only mode: these are unreliable, but they are all we have.
        d["qk_norm"] = any(t.get(k) for k in ("use_qk_norm", "qk_norm", "q_norm", "k_norm"))
        d["bias"] = bool(g(t, "attention_bias", "qkv_bias", default=False))
    else:
        d.update({f"w.{k}": v for k, v in facts.items()})
    return d


def signature(d: dict) -> str:
    return "|".join(f"{k}={d[k]}" for k in sorted(d))


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"HF root (default {DEFAULT_ROOT})")
    ap.add_argument("--with-manifest", action="store_true",
                    help="also derive structure from the tensor manifest (authoritative)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    no_manifest: list[tuple[str, str]] = []
    for label, path in find_configs(args.root).items():
        cfg = load_json(path)
        if "__error__" in cfg:
            skipped.append((label, cfg["__error__"]))
            continue
        model_type = str(g(cfg, "model_type", default="?"))
        t = text_cfg(cfg)
        if model_type in SKIP_TYPES or g(t, "num_hidden_layers", "num_layers", "n_layer") is None:
            skipped.append((label, f"not a decoder LM (model_type={model_type})"))
            continue
        facts = None
        if args.with_manifest:
            facts = tensor_facts(manifest_keys(path))
            if facts is None:
                no_manifest.append((label, model_type))
                continue
        d = descriptor(cfg, facts)
        rows.append({"label": label, "mt": model_type, "desc": d, "sig": signature(d)})

    mode = "config + manifest" if args.with_manifest else "config only"
    print(f"mode: {mode}")
    print(f"{len(rows)} decoder LMs, {len(skipped)} skipped"
          + (f", {len(no_manifest)} without a readable manifest" if args.with_manifest else ""))

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["sig"]].append(r)
    model_types = {r["mt"] for r in rows}
    print(f"\n=== {len(groups)} DISTINCT STRUCTURES / {len(model_types)} model_types ===\n")

    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for i, (_, grp) in enumerate(ordered, 1):
        d = grp[0]["desc"]
        mts = sorted({r["mt"] for r in grp})
        print(f"[S{i:02d}] {len(grp):3d}  {', '.join(mts)}")
        print(f"      mixers={d['mixers']}/{d['mixer.type']}  ffn={d['ffn.kind']}/{d['ffn.type']}"
              f"  rope={d['rope']}  pattern={d['pattern']}  towers={d['towers']}")
        flags = [k[2:] for k, v in d.items() if k.startswith("w.") and v is True]
        if args.with_manifest:
            print(f"      weights: {', '.join(flags) if flags else '(plain)'}"
                  f"  experts={d.get('w.expert_layout')}")

    print("\n=== AXIS COVERAGE ===")
    for axis in ("mixers", "mixer.type", "ffn.kind", "ffn.type", "rope", "pattern", "towers"):
        print(f"{axis:12s} {dict(Counter(r['desc'][axis] for r in rows).most_common())}")

    print("\n=== COLLAPSE: model_types sharing one structure ===")
    found = False
    for i, (_, grp) in enumerate(ordered, 1):
        mts = sorted({r["mt"] for r in grp})
        if len(mts) > 1:
            found = True
            print(f"  S{i:02d}: {' == '.join(mts)}  ({len(grp)} repos)")
    if not found:
        print("  (none) -- structure-based identity does not merge any two families here")

    print("\n=== AMBIGUITY: one model_type spanning many structures ===")
    spans: defaultdict[str, set[str]] = defaultdict(set)
    for r in rows:
        spans[r["mt"]].add(r["sig"])
    for mt, sigs in sorted(spans.items(), key=lambda kv: -len(kv[1])):
        if len(sigs) > 1:
            print(f"  {mt:34s} {len(sigs)} structures")

    print("\n=== HYBRID STACKS (more than one mixer kind) ===")
    hybrids = Counter(r["desc"]["mixers"] for r in rows if "+" in r["desc"]["mixers"])
    total = sum(hybrids.values())
    for m, n in hybrids.most_common():
        print(f"  {m:16s} {n:3d} repos")
    print(f"  -> {total}/{len(rows)} ({100 * total // max(len(rows), 1)}%) of the corpus is hybrid")

    if no_manifest:
        print(f"\n=== no readable manifest ({len(no_manifest)}) -- unverified ===")
        for label, mt in no_manifest:
            print(f"  {label[:56]:58s} {mt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
