# Golden vectors for the DFlash2 conv + selector, produced by the REFERENCE
# code (z-lab/dflash dflash/model.py) against the REAL checkpoint weights.
# A Rust port checked against these is validated; one checked against a
# re-derivation of the same guess is not.
import sys, json, struct
sys.path.insert(0, "dflash-ref")
import torch
from safetensors.torch import load_file
from dflash.model import _grouped_dynamic_convolve

torch.manual_seed(0)
SNAP = "/srv/huggingface/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
cfg = json.load(open(f"{SNAP}/config.json"))
dc = cfg["dflash_config"]
H, GS, KS = cfg["hidden_size"], dc["conv_group_size"], dc["conv_kernel_size"]
RANK, TOPK = dc["selector_rank"], dc["selector_top_k"]
st = load_file(f"{SNAP}/model.safetensors")

# --- conv: exercise the real base_kernel + kernel_projection of layer 0 -----
base = st["layers.0.attention_conv.base_kernel"].float()          # [2, KS, H]
proj = st["layers.0.attention_conv.kernel_projection.weight"].float()  # [1280, H]
L = 4
hidden = torch.randn(1, L, H) * 0.05
groups = H // GS
dyn = (hidden @ proj.T).view(1, L, 2, KS, groups)
conv_prepare = _grouped_dynamic_convolve(hidden, dyn[..., 0, :, :], base[0], GS)
conv_finish  = _grouped_dynamic_convolve(hidden, dyn[..., 1, :, :], base[1], GS)

# --- selector: real codebooks + hidden_projection, small vocab slice --------
hp = st["candidate_selector.hidden_projection.weight"].float()     # [RANK, H]
pre = st["candidate_selector.predecessor_codebook"].float()        # [V, RANK]
suc = st["candidate_selector.successor_codebook"].float()          # [V, RANK]
V = pre.shape[0]
hp_h = hidden @ hp.T                                              # [1, L, RANK]
cand = torch.randint(0, V, (1, L, TOPK))
unary = torch.randn(1, L, TOPK)
anchor = torch.randint(0, V, (1,))
predecessor, path, all_scores = anchor, [], []
for pos in range(L):
    scores = unary[:, pos] + torch.einsum(
        "br,bkr->bk", pre[predecessor] * hp_h[:, pos], suc[cand[:, pos]])
    all_scores.append(scores)
    idx = torch.argmax(scores, dim=-1)
    predecessor = cand[:, pos].gather(-1, idx[:, None])[:, 0]
    path.append(int(predecessor))

def w(f, t):
    a = t.detach().contiguous().view(-1).float().numpy()
    f.write(struct.pack("<Q", a.size)); f.write(a.tobytes())

with open("dflash2_golden.bin", "wb") as f:
    f.write(struct.pack("<6q", H, GS, KS, RANK, TOPK, L))
    w(f, hidden); w(f, base); w(f, proj)
    w(f, conv_prepare); w(f, conv_finish)
    w(f, hp_h); w(f, cand.float()); w(f, unary)
    w(f, pre[anchor]); w(f, torch.stack([pre[c] for c in cand[0]]))
    w(f, torch.stack([suc[c] for c in cand[0]]))
    w(f, torch.cat(all_scores)); w(f, torch.tensor(path, dtype=torch.float))
print(f"H={H} GS={GS} KS={KS} rank={RANK} topk={TOPK} L={L} groups={groups}")
print("conv_prepare[0,0,:4] =", conv_prepare[0,0,:4].tolist())
print("conv_finish [0,0,:4] =", conv_finish[0,0,:4].tolist())
print("selector scores[0,:4] =", all_scores[0][0,:4].tolist())
print("selector path =", path)
print("wrote dflash2_golden.bin")
