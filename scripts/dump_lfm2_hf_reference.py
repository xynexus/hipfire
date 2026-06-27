#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — HF Lfm2ForCausalLM per-layer reference dump for forward bisection.
#
# Loads LFM2.5 from a local snapshot, tokenizes the prompt, and writes:
#   - <prefix>.tokens.json          token ids (feed the SAME ids to the hipfire dump)
#   - <prefix>.hf.hidden.bin        post-layer residuals [pos][layer][hidden] f32
#   - <prefix>.hf.logits.json       final-position logit stats + top-20
#
# HF output_hidden_states is (n_layers+1) tensors; index 0 is post-embedding,
# index L+1 is the residual after layer L — which is what the hipfire capture
# records, so we dump indices 1..=n_layers.
#
# Tooling only (AGENTS.md: Python allowed for tooling/comparison, not hot path).
import argparse, json, struct, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out-prefix", default="/tmp/lfm2")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, torch_dtype=dtype)
    model.eval()
    print(f"model: {model.config.model_type} layers={model.config.num_hidden_layers} "
          f"hidden={model.config.hidden_size} vocab={model.config.vocab_size}", file=sys.stderr)
    lt = getattr(model.config, "layer_types", None)
    if lt:
        print("layer_types:", lt, file=sys.stderr)

    # Raw tokenization (no chat template) to match the hipfire dump's tok.encode.
    ids = tok(args.prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    id_list = ids[0].tolist()
    print(f"tokens ({len(id_list)}): {id_list}", file=sys.stderr)
    with open(f"{args.out_prefix}.tokens.json", "w") as f:
        json.dump(id_list, f)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, hidden]
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    n_pos = ids.shape[1]

    # [pos][layer][hidden], post-layer (HF index L+1) to match hipfire.
    with open(f"{args.out_prefix}.hf.hidden.bin", "wb") as f:
        f.write(b"LFM2HID0")
        f.write(struct.pack("<IIII", n_layers, n_pos, hidden, 0))
        for p in range(n_pos):
            for l in range(n_layers):
                row = hs[l + 1][0, p].float().cpu().numpy().astype(np.float32)
                f.write(row.tobytes())

    logits = out.logits[0, -1].float().cpu().numpy()
    order = np.argsort(-logits)
    top = [{"id": int(j), "logit": float(logits[j]), "text": tok.decode([int(j)])}
           for j in order[:20]]
    payload = {
        "tokens": id_list,
        "logit_min": float(logits.min()),
        "logit_max": float(logits.max()),
        "logit_mean": float(logits.mean()),
        "argmax": int(order[0]),
        "argmax_text": tok.decode([int(order[0])]),
        "top20": top,
    }
    with open(f"{args.out_prefix}.hf.logits.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"argmax={order[0]} ({tok.decode([int(order[0])])!r}) "
          f"logit_range=[{logits.min():.3f},{logits.max():.3f}]", file=sys.stderr)
    print(f"wrote {args.out_prefix}.hf.hidden.bin / .hf.logits.json / .tokens.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
