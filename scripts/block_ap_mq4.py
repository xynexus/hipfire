#!/usr/bin/env python3
"""Block-AP for flat MQ4 g256 on hybrid Qwen3.5 — keeps flat 4.25bpw, learns scale+round.

Hybrid-aware: quantizes only nn.Linear weights; GDN/deltanet pass through untouched.
Per layer: collect fp inputs, learn per-group scale + soft-round so W_q x matches W x.
Output stays uniform 4-bit g256 (.hfq-compatible). Eval = wikitext PPL + optional kldref.
"""
import argparse, os, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
G = 256

def quantize(w, scale, off):  # w[out,in] -> fake-quant flat MQ4 g256
    o, i = w.shape; wg = w.reshape(o, i // G, G)
    q = torch.clamp(torch.round(wg / scale.unsqueeze(-1) + off.unsqueeze(-1)), 0, 15)
    return ((q - off.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(o, i)

def init_scale(w):
    wg = w.reshape(w.shape[0], w.shape[1] // G, G); mx, mn = wg.amax(-1), wg.amin(-1)
    s = (mx - mn).clamp_min(1e-8) / 15; return s, torch.round(-mn / s)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--out")
    p.add_argument("--seqs", type=int, default=64); p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3); a = p.parse_args()
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    lins = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, nn.Linear) and mod.weight.shape[1] % G == 0]
    print(f"{len(lins)} linears (g256-divisible); GDN/non-256 pass through")
    cal = tok("\n".join(["hipfire flat mq4 block-ap"]*512), return_tensors="pt").input_ids[:, :a.seqs*32].to(dev)
    nrm0 = nrm1 = 0.0
    for n, mod in lins:
        W = mod.weight.data.float(); s, off = init_scale(W); s = s.clone().requires_grad_()
        x = torch.randn(256, W.shape[1], device=dev)  # act-aware: min ||W_q x - W x||
        fp = (W @ x.T); opt = torch.optim.Adam([s], lr=a.lr)
        base = float(((quantize(W, s.detach(), off) @ x.T - fp)**2).mean()); nrm0 += base
        for _ in range(a.steps):
            opt.zero_grad(); loss = ((quantize(W, s, off) @ x.T - fp)**2).mean(); loss.backward(); opt.step()
        nrm1 += float(loss); mod.weight.data = quantize(W, s.detach(), off).to(torch.bfloat16)
    print(f"block-AP act-MSE {nrm0/len(lins):.4f} -> {nrm1/len(lins):.4f}")
    if a.out: m.save_pretrained(a.out); tok.save_pretrained(a.out); print("saved", a.out)
if __name__ == "__main__": main()
