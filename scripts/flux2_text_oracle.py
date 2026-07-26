#!/usr/bin/env python3
"""Text-conditioning oracle: run the snapshot Qwen3-4B text encoder (reference,
CPU torch) on a prompt with the FLUX.2 extraction (layers [9,18,27] concatenated)
and compare to hipfire's dumped `conditioning_positive`.

This validates the ONE pipeline link never checked against a reference: the
text->conditioning step (the user's suspected "text->image confused" stage).

  match  -> hipfire's text encoding is correct; a bad image is settings/model
  diverge-> hipfire's Qwen3 extraction is the bug

  python scripts/flux2_text_oracle.py <prompt_file> <hipfire_conditioning.bin>
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch

SNAP = Path("/srv/huggingface/models--black-forest-labs--FLUX.2-klein-base-4B/"
            "snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d")
LAYERS = [9, 18, 27]
MAX_LEN = 512


def load_hfdt(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (r,) = struct.unpack_from("<I", raw, 4)
    dims = struct.unpack_from(f"<{r}I", raw, 8)
    off = 8 + r * 4
    return np.frombuffer(raw, "<f4", offset=off).reshape(dims).astype(np.float64)


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = Path(sys.argv[1]).read_text().strip()
    hf = load_hfdt(Path(sys.argv[2]))  # [1, seq, 3*hidden]
    print(f"prompt: {prompt!r}\nhipfire conditioning {hf.shape}")

    tok = AutoTokenizer.from_pretrained(str(SNAP / "tokenizer"))
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    print(f"tokens: {enc['input_ids'].shape[1]}  (nonpad {int(enc['attention_mask'].sum())})")

    model = AutoModelForCausalLM.from_pretrained(str(SNAP / "text_encoder"), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple len num_layers+1; [k] = output of layer k
    stacked = torch.stack([hs[k] for k in LAYERS], dim=1)          # [b, 3, seq, hid]
    ref = stacked.permute(0, 2, 1, 3).reshape(1, MAX_LEN, -1).double().numpy()  # [b, seq, 3*hid]
    print(f"reference conditioning {ref.shape}")

    if ref.shape != hf.shape:
        print(f"!! SHAPE MISMATCH ref {ref.shape} vs hipfire {hf.shape}")
        return 1
    d = ref - hf
    rel = np.linalg.norm(d) / (np.linalg.norm(hf) + 1e-12)
    cos = float((ref.ravel() @ hf.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(hf) + 1e-12))
    print(f"\n=== reference vs hipfire conditioning ===")
    print(f"  rel-L2 = {rel:.4f}   cos = {cos:.4f}")
    # per-layer breakdown (the 3 concatenated blocks)
    hid = ref.shape[2] // 3
    for i, L in enumerate(LAYERS):
        a = ref[..., i * hid:(i + 1) * hid]; b = hf[..., i * hid:(i + 1) * hid]
        r = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
        c = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  layer {L:2d} block: rel-L2={r:.4f} cos={c:.4f}")
    # also compare only the non-padded prefix (padding conventions can differ)
    npad = int(enc["attention_mask"].sum())
    a = ref[:, :npad]; b = hf[:, :npad]
    rp = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
    print(f"  non-padded prefix ({npad} tok): rel-L2={rp:.4f}")
    print("\n=> " + ("MATCH: hipfire text encoding is correct." if rel < 0.15
                      else "DIVERGE: hipfire's Qwen3 conditioning differs from the reference — text-encoder bug."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
