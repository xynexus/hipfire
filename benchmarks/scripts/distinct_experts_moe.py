#!/usr/bin/env python3
"""Measure distinct_experts(B): growth of the activated MoE expert union with the
number of draft tokens verified together, along a DEPTH axis (consecutive
positions) and a WIDTH axis (alternative next-token candidates at one position),
vs a random-uniform baseline.

Reusable across MoE architectures whose HF module exposes a per-token top-k
router selection. Router capture is done by monkeypatching the routing method of
each sparse-MoE block to record the selected expert ids.
"""
import os, sys, json, argparse, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tv_shim  # noqa: F401  (must precede transformers import)
import numpy as np
import torch

torch.set_grad_enabled(False)

# ---------------------------------------------------------------- corpus
def load_corpus_file(path, n_docs):
    """Read pre-extracted docs: JSONL with a 'text' field, one doc per line."""
    docs = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            docs.append(json.loads(s)["text"])
            if len(docs) >= n_docs:
                break
    return docs

def load_wiki_docs(n_docs, min_chars=4000):
    import pyarrow.parquet as pq
    base = ("/srv/huggingface/datasets--wikimedia--structured-wikipedia/"
            "snapshots/417c267bb457fa645c22eb3b5c77764963194c70/enwiki/data/"
            "enwiki_namespace_0_0.parquet")
    pf = pq.ParquetFile(base)
    docs = []
    for batch in pf.iter_batches(batch_size=64, columns=["abstract", "sections"]):
        for row in batch.to_pylist():
            parts = []
            if row.get("abstract"):
                parts.append(row["abstract"])
            secs = row.get("sections")
            if secs:
                parts.append(_extract_section_text(secs))
            text = "\n\n".join(p for p in parts if p)
            if len(text) >= min_chars:
                docs.append(text)
            if len(docs) >= n_docs:
                return docs
    return docs

def _extract_section_text(secs):
    out = []
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "paragraph" and node.get("value"):
                out.append(node["value"])
            for hp in node.get("has_parts", []) or []:
                walk(hp)
        elif isinstance(node, list):
            for x in node:
                walk(x)
    try:
        import json as _j
        walk(secs if not isinstance(secs, str) else _j.loads(secs))
    except Exception:
        pass
    return "\n".join(out)

# ---------------------------------------------------------------- router capture
class Capture:
    """Patches each sparse-MoE block's routing method to record selected experts.
    Records a list per forward: layer_idx -> selected_experts tensor [tokens, k]."""
    def __init__(self, model):
        self.model = model
        self.blocks = []          # (layer_idx, module)
        self.records = {}         # layer_idx -> np.ndarray [tokens, k]
        self._orig = {}
        self._enabled = False
        self._discover()

    def _discover(self):
        # Two supported HF shapes:
        #  (A) sparse block with a `route_tokens_to_experts(router_logits)` method
        #      that returns (selected_experts, weights)  [LFM2-MoE]
        #  (B) a TopK router submodule whose forward returns a tuple containing
        #      the integer selected-experts tensor  [Qwen3.5-MoE and family]
        idx = 0
        seen = set()
        for name, mod in self.model.named_modules():
            if hasattr(mod, "route_tokens_to_experts") and callable(getattr(mod, "route_tokens_to_experts")):
                self.blocks.append((idx, name, mod, "route_tokens_to_experts")); idx += 1
        if not self.blocks:
            for name, mod in self.model.named_modules():
                cn = type(mod).__name__.lower()
                if ("router" in cn or cn.endswith("gate")) and hasattr(mod, "forward"):
                    # skip vision routers if any
                    if "vision" in name.lower():
                        continue
                    self.blocks.append((idx, name, mod, "forward")); idx += 1
        if not self.blocks:
            raise RuntimeError("no routing modules found; add a router hook shape")

    @staticmethod
    def _extract_sel(out):
        """Find the integer [N,k] top-k index tensor among a call's outputs."""
        cands = out if isinstance(out, (tuple, list)) else (out,)
        best = None
        for t in cands:
            if torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32, torch.long) and t.dim() == 2:
                # prefer the smaller last-dim (k) over anything huge
                if best is None or t.shape[1] < best.shape[1]:
                    best = t
        return best

    def __enter__(self):
        self._enabled = True
        self.records = {}
        for li, name, mod, fn in self.blocks:
            orig = getattr(mod, fn)
            self._orig[(li, fn)] = orig
            def make(li, orig):
                def wrapper(*a, **k):
                    out = orig(*a, **k)
                    if self._enabled:
                        sel = self._extract_sel(out)
                        if sel is not None:
                            self.records[li] = sel.detach().to("cpu").numpy().astype(np.int16)
                    return out
                return wrapper
            setattr(mod, fn, make(li, orig))
        return self

    def __exit__(self, *exc):
        self._enabled = False
        for li, name, mod, fn in self.blocks:
            setattr(mod, fn, self._orig[(li, fn)])
        self._orig = {}

    def snapshot(self):
        """Return {layer_idx: [tokens,k] int array} for the last forward."""
        return {li: v.copy() for li, v in self.records.items()}

# ---------------------------------------------------------------- union math
def union_growth_depth(per_pos, Bs):
    """per_pos: list over layers of arrays [S, k]; average over sliding windows.
    Returns {layer_idx: {B: mean_union}} plus aggregate."""
    res = {}
    for li, arr in per_pos.items():
        S = arr.shape[0]
        row = {}
        for B in Bs:
            if B > S:
                continue
            sizes = []
            # subsample starts if too many
            starts = list(range(0, S - B + 1))
            if len(starts) > 512:
                starts = random.sample(starts, 512)
            for s in starts:
                u = np.unique(arr[s:s+B])
                sizes.append(len(u))
            row[B] = float(np.mean(sizes))
        res[li] = row
    return res

def union_growth_width(cand_layers, Bs, reps, rng):
    """cand_layers: {layer_idx: array [W, k]} routing of W candidates at one branch.
    Average union over random W-subsets of each size B."""
    res = {}
    for li, arr in cand_layers.items():
        W = arr.shape[0]
        row = {}
        for B in Bs:
            if B > W:
                continue
            sizes = []
            n = min(reps, 256)
            for _ in range(n):
                sel = rng.choice(W, size=B, replace=False)
                sizes.append(len(np.unique(arr[sel])))
            row[B] = float(np.mean(sizes))
        res[li] = row
    return res

def analytic_random(E, k, B):
    return E * (1.0 - (1.0 - k / E) ** B)

def aggregate(per_layer, Bs):
    """Mean across layers for each B."""
    out = {}
    for B in Bs:
        vals = [d[B] for d in per_layer.values() if B in d]
        if vals:
            out[B] = float(np.mean(vals))
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-docs", type=int, default=6)
    ap.add_argument("--depth-len", type=int, default=576)
    ap.add_argument("--width-prefixes", type=int, default=4)
    ap.add_argument("--width-prefix-len", type=int, default=40)
    ap.add_argument("--width-max", type=int, default=256)
    ap.add_argument("--width-batch", type=int, default=64, help="candidate microbatch")
    ap.add_argument("--max-B", type=int, default=512)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--corpus-file", default=None,
                    help="JSONL of pre-extracted docs (field 'text'); else read wiki parquet")
    ap.add_argument("--max-layers", type=int, default=0,
                    help="load only the first N transformer layers (memory relief on "
                         "unified-memory boxes). Routing for layers 0..N-1 stays exact.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    t0 = time.time()
    dtype = getattr(torch, args.dtype)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tcfg = getattr(cfg, "text_config", cfg)  # VL wrappers nest the LM config
    if args.max_layers:
        n = args.max_layers
        cfgs = [cfg] if cfg is tcfg else [cfg, tcfg]
        for c in cfgs:
            if getattr(c, "num_hidden_layers", None):
                c.num_hidden_layers = min(c.num_hidden_layers, n)
            # truncate any per-layer list configs to match
            for attr in ("layer_types", "mlp_only_layers", "full_attention_idx"):
                v = getattr(c, attr, None)
                if isinstance(v, list) and len(v) > n:
                    setattr(c, attr, v[:n])
        print(f"[cfg] truncated to first {n} layers", flush=True)
    def g(name):
        return getattr(tcfg, name, None) or getattr(cfg, name, None)
    E = g("num_experts") or g("n_routed_experts")
    k = g("num_experts_per_tok") or g("num_experts_per_token")
    print(f"[cfg] E={E} k={k} layers={g('num_hidden_layers')} hidden={g('hidden_size')}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    # A plain device string ("cuda") does NOT engage accelerate's streaming
    # big-model loader, so HF materializes a full CPU copy first — fatal on a
    # unified-memory APU where GPU (GTT) and CPU share the RAM pool. Use
    # device_map="auto" for GPU so shards stream straight to the device.
    device_map = "auto" if args.device.startswith("cuda") else args.device
    def _load(cls):
        return cls.from_pretrained(
            args.model, config=cfg, dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True, device_map=device_map)
    try:
        model = _load(AutoModelForCausalLM)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = _load(AutoModelForImageTextToText)
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    cap = Capture(model)
    print(f"[capture] {len(cap.blocks)} MoE routing modules", flush=True)

    Bs = [b for b in [1,2,4,8,16,32,64,128,256,512] if b <= args.max_B]

    # ---------------- DEPTH ----------------
    if args.corpus_file:
        docs = load_corpus_file(args.corpus_file, args.depth_docs + 3)
    else:
        docs = load_wiki_docs(args.depth_docs + 3)
    depth_layer_acc = {}   # layer -> list of per-doc {B:val}
    tok_pool = []          # for empirical random baseline: flat list of [k] arrays per layer
    pool_layers = {}
    d_used = 0
    with cap:
        for text in docs:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.depth_len).input_ids.to(args.device)
            if ids.shape[1] < 64:
                continue
            model(ids, use_cache=False)
            snap = cap.snapshot()   # layer -> [S,k]
            g = union_growth_depth(snap, Bs)
            for li, row in g.items():
                depth_layer_acc.setdefault(li, []).append(row)
                pool_layers.setdefault(li, []).append(snap[li])
            d_used += 1
            print(f"[depth] doc {d_used} len={ids.shape[1]} t={time.time()-t0:.0f}s", flush=True)
            if d_used >= args.depth_docs:
                break

    # average depth over docs
    depth_layer = {}
    for li, rows in depth_layer_acc.items():
        agg = {}
        for B in Bs:
            vals = [r[B] for r in rows if B in r]
            if vals:
                agg[B] = float(np.mean(vals))
        depth_layer[li] = agg
    depth_agg = aggregate(depth_layer, Bs)

    # ---------------- empirical random baseline (real tokens, shuffled) ----------------
    rand_layer = {}
    for li, arrs in pool_layers.items():
        allsel = np.concatenate(arrs, axis=0)  # [Ntok, k]
        N = allsel.shape[0]
        row = {}
        for B in Bs:
            if B > N:
                continue
            sizes = []
            for _ in range(200):
                sel = rng.choice(N, size=B, replace=False)
                sizes.append(len(np.unique(allsel[sel])))
            row[B] = float(np.mean(sizes))
        rand_layer[li] = row
    rand_agg = aggregate(rand_layer, Bs)

    # ---------------- WIDTH ----------------
    width_layer_acc = {}
    wp_used = 0
    for text in docs:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.width_prefix_len).input_ids.to(args.device)
        if ids.shape[1] < 8:
            continue
        # next-token distribution at the prefix end
        with cap:  # (routing of the prefix forward is ignored)
            out = model(ids, use_cache=False)
        logits = out.logits[0, -1]
        topW = torch.topk(logits, k=args.width_max).indices.to(args.device)  # [W]
        W = topW.shape[0]
        prefix = ids[0]  # [P]
        # build [W, P+1] batch: prefix + candidate
        cand_rows = {}  # layer -> [W,k]
        for start in range(0, W, args.width_batch):
            cbatch = topW[start:start+args.width_batch]
            b = cbatch.shape[0]
            seq = prefix.unsqueeze(0).repeat(b, 1)            # [b,P]
            seq = torch.cat([seq, cbatch.unsqueeze(1)], dim=1)  # [b,P+1]
            with cap:
                model(seq, use_cache=False)
            snap = cap.snapshot()  # layer -> [b*(P+1), k]
            P1 = seq.shape[1]
            for li, arr in snap.items():
                arr3 = arr.reshape(b, P1, -1)
                last = arr3[:, -1, :]  # [b,k] routing of the candidate token
                cand_rows.setdefault(li, []).append(last)
        cand_layers = {li: np.concatenate(v, axis=0) for li, v in cand_rows.items()}
        g = union_growth_width(cand_layers, Bs, reps=200, rng=rng)
        for li, row in g.items():
            width_layer_acc.setdefault(li, []).append(row)
        wp_used += 1
        print(f"[width] prefix {wp_used} W={W} t={time.time()-t0:.0f}s", flush=True)
        if wp_used >= args.width_prefixes:
            break

    width_layer = {}
    for li, rows in width_layer_acc.items():
        agg = {}
        for B in Bs:
            vals = [r[B] for r in rows if B in r]
            if vals:
                agg[B] = float(np.mean(vals))
        width_layer[li] = agg
    width_agg = aggregate(width_layer, Bs)

    analytic = {B: analytic_random(E, k, B) for B in Bs}

    result = dict(
        model=args.model, E=E, k=k,
        num_moe_layers=len(cap.blocks),
        Bs=Bs,
        depth_aggregate=depth_agg,
        width_aggregate=width_agg,
        random_empirical_aggregate=rand_agg,
        analytic_random=analytic,
        depth_per_layer=depth_layer,
        width_per_layer=width_layer,
        random_per_layer=rand_layer,
        meta=dict(depth_docs=d_used, depth_len=args.depth_len,
                  width_prefixes=wp_used, width_prefix_len=args.width_prefix_len,
                  width_max=args.width_max, seed=args.seed,
                  elapsed_s=round(time.time()-t0, 1)),
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("[done]", args.out, f"{time.time()-t0:.0f}s", flush=True)
    # quick console table
    print(f"\nB      depth   width   rand_emp  analytic   (E={E},k={k})")
    for B in Bs:
        print(f"{B:<5}  {depth_agg.get(B,float('nan')):6.2f}  "
              f"{width_agg.get(B,float('nan')):6.2f}  "
              f"{rand_agg.get(B,float('nan')):7.2f}   {analytic[B]:7.2f}")

if __name__ == "__main__":
    main()
