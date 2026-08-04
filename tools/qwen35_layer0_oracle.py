#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Independent numpy recomputation of one Qwen3.5 linear_attn layer.

Third implementation, deliberately. `hipfire-train`'s hybrid walk disagrees
with the runtime's own forward, and the question that matters first is WHICH
kind of wrong: a coding mistake in the Rust, or a wrong understanding of the
architecture. Those need different fixes, and a second implementation written
from the same understanding separates them in one run.

It reads the bf16 safetensors directly and follows the formulas as documented,
then compares against both `mine_hidden.bin` (the walk, via
HIPFIRE_GH_ORACLE_DIR) and `ref_hidden.bin` (the runtime, via
dump_qwen35_hidden_states).

Answer as of writing: numpy agrees with the Rust at cos 0.995 and both differ
from the runtime at 0.45 — so the Rust is faithful and the UNDERSTANDING is
what is wrong. Variants are cheap to try here (seconds, no rebuild), which is
what the `VAR` argument is for.

Usage: qwen35_layer0_oracle.py [base|nogate|sigmoid|l2|gate_norm]
Paths at the top are local scratch; edit them for another host.
"""
import json, struct, numpy as np
F='/home/sadara/.claude/jobs/9a1047b0/tmp/q08b/model.safetensors-00001-of-00001.safetensors'
fh=open(F,'rb'); n=int.from_bytes(fh.read(8),'little'); HDR=json.loads(fh.read(n)); BASE=8+n
def T(name):
    i=HDR[name]; o=i['data_offsets']; fh.seek(BASE+o[0]); raw=fh.read(o[1]-o[0])
    if i['dtype']=='F32':
        return np.frombuffer(raw,dtype='<f4').reshape(i['shape'])
    assert i['dtype']=='BF16', i['dtype']
    u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16
    return np.frombuffer(u.astype('<u4').tobytes(),dtype='<f4').reshape(i['shape'])
P='model.language_model.'
d='/home/sadara/.claude/jobs/9a1047b0/tmp/oracle/'
b=open(d+'mine_hidden.bin','rb').read(); nl,npos,hd,_=struct.unpack('<4I',b[8:24])
mine=np.frombuffer(b[24:24+nl*npos*hd*4],dtype=np.float32).reshape(nl,npos,hd)
b=open(d+'ref_hidden.bin','rb').read(); rl,_,_,_=struct.unpack('<4I',b[8:24])
ref=np.frombuffer(b[24:24+rl*npos*hd*4],dtype=np.float32).reshape(rl,npos,hd)
toks=np.frombuffer(open(d+'tokens.hfkldr','rb').read()[32:],dtype=np.uint32)[:npos]

eps=1e-6
x=T(P+'embed_tokens.weight')[toks].astype(np.float32)
def rms(v,w): return v/np.sqrt((v**2).mean(-1,keepdims=True)+eps)*w
L=P+'layers.0.'
import sys
VAR=sys.argv[1] if len(sys.argv)>1 else 'base'
xn1=x.copy() if VAR=='nonorm1' else rms(x,T(L+'input_layernorm.weight'))
if VAR=='norm1_plus1': xn1=rms(x,1.0+T(L+'input_layernorm.weight'))
qkv=xn1@T(L+'linear_attn.in_proj_qkv.weight').T
cw=T(L+'linear_attn.conv1d.weight')            # [C,1,4]
C=qkv.shape[1]; K=cw.shape[-1]
pre=np.zeros_like(qkv)
for t in range(npos):
    acc=np.zeros(C,dtype=np.float32)
    for j in range(K):
        s=t-(K-1-j)
        jj=(K-1-j) if VAR=='convrev' else j
        if s>=0: acc+=cw[:,0,jj]*qkv[s]
    pre[t]=acc
silu=lambda v: v/(1+np.exp(-v))
nh=T(L+'linear_attn.A_log').shape[0]; hv=T(L+'linear_attn.norm.weight').shape[0]
hk=(C//nh-hv)//2
if VAR=='qk_swap':
    k=silu(pre[:,:nh*hk]); q=silu(pre[:,nh*hk:2*nh*hk]); v=silu(pre[:,2*nh*hk:])
elif VAR=='vqk':
    v=silu(pre[:,:nh*hv]); q=silu(pre[:,nh*hv:nh*hv+nh*hk]); k=silu(pre[:,nh*hv+nh*hk:])
elif VAR=='noconv':
    q=silu(qkv[:,:nh*hk]); k=silu(qkv[:,nh*hk:2*nh*hk]); v=silu(qkv[:,2*nh*hk:])
elif VAR=='nosilu':
    q=pre[:,:nh*hk]; k=pre[:,nh*hk:2*nh*hk]; v=pre[:,2*nh*hk:]
else:
    q=silu(pre[:,:nh*hk]); k=silu(pre[:,nh*hk:2*nh*hk]); v=silu(pre[:,2*nh*hk:])
def l2h(a,n,dh,scale=1.0):
    a=a.reshape(npos,n,dh); g=1/np.sqrt((a**2).sum(-1,keepdims=True)+eps)
    return (a*g*scale).reshape(npos,n*dh)
q=l2h(q,nh,hk,1/np.sqrt(hk)); k=l2h(k,nh,hk)
a_raw=xn1@T(L+'linear_attn.in_proj_a.weight').T; b_raw=xn1@T(L+'linear_attn.in_proj_b.weight').T
dt=T(L+'linear_attn.dt_bias'); alog=T(L+'linear_attn.A_log')
sp=np.logaddexp(0,a_raw+dt); gate=sp*(-np.exp(alog)); alpha=np.exp(gate)
beta=1/(1+np.exp(-b_raw))
S=np.zeros((nh,hv,hk),dtype=np.float32); out=np.zeros((npos,nh*hv),dtype=np.float32)
for t in range(npos):
    kt=k[t].reshape(nh,hk); vt=v[t].reshape(nh,hv); qt=q[t].reshape(nh,hk)
    kv=np.einsum('hvk,hk->hv',S,kt)
    delta=(vt-alpha[t][:,None]*kv)*beta[t][:,None]
    S=alpha[t][:,None,None]*S+delta[:,:,None]*kt[:,None,:]
    out[t]=np.einsum('hvk,hk->hv',S,qt).reshape(-1)
z=xn1@T(L+'linear_attn.in_proj_z.weight').T
nw=T(L+'linear_attn.norm.weight')
o=out.reshape(npos,nh,hv); g=1/np.sqrt((o**2).mean(-1,keepdims=True)+eps)
import sys
base=(o*g*nw).reshape(npos,nh*hv)
if VAR=='nogate': normed=base
elif VAR=='sigmoid': normed=base/(1+np.exp(-z))
elif VAR=='l2':   normed=(o*(1/np.sqrt((o**2).sum(-1,keepdims=True)+eps))*nw).reshape(npos,nh*hv)*silu(z)
elif VAR=='gate_norm': normed=(o*g*nw).reshape(npos,nh*hv)*silu(z)*np.sqrt(hv)
else: normed=base*silu(z)   # every non-gated-norm variant keeps the base gate
attn=normed@T(L+'linear_attn.out_proj.weight').T
x_mid=x+attn
xn2=rms(x_mid,T(L+'post_attention_layernorm.weight'))
mlp=(silu(xn2@T(L+'mlp.gate_proj.weight').T)*(xn2@T(L+'mlp.up_proj.weight').T))@T(L+'mlp.down_proj.weight').T
y=x_mid+mlp
R=lambda v: float(np.sqrt((np.asarray(v)**2).mean()))
print('--- stage magnitudes (rms) ---')
for nm,v in [('x(embed)',x),('xn1',xn1),('qkv',qkv),('conv_pre',pre),('q(post-l2)',q),('k(post-l2)',k),('v',v),
             ('alpha',alpha),('beta',beta),('recur_out',out),('z',z),('normed',normed),('attn',attn),
             ('x_mid',x_mid),('xn2',xn2),('mlp',mlp),('y',y)]:
    print('  %-11s %.5f'%(nm,R(v)))
cos=lambda a,b: float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print('numpy layer0 out rms %.4f | ref rms %.4f | mine rms %.4f'%(np.sqrt((y**2).mean()),np.sqrt((ref[0]**2).mean()),np.sqrt((mine[1]**2).mean())))
print('cos(numpy, ref[0])  =', round(cos(y,ref[0]),4))
print('cos(numpy, mine[1]) =', round(cos(y,mine[1]),4))
print('cos(mine[1], ref[0])=', round(cos(mine[1],ref[0]),4))
pc=[round(float((y[i]*ref[0][i]).sum()/(np.linalg.norm(y[i])*np.linalg.norm(ref[0][i])+1e-12)),3) for i in range(npos)]
print('per-position cos(numpy, ref) first 12:', pc[:12])
print('  pos0 %.3f  mean %.3f  last %.3f'%(pc[0], float(np.mean(pc)), pc[-1]))
xm=[round(float((x_mid[i]*ref[0][i]).sum()/(np.linalg.norm(x_mid[i])*np.linalg.norm(ref[0][i])+1e-12)),3) for i in range(4)]
print('per-position cos(x_mid, ref) first 4:', xm)
print('  attn branch rms %.4f  mlp branch rms %.4f  x rms %.4f'%(np.sqrt((attn**2).mean()),np.sqrt((mlp**2).mean()),np.sqrt((x**2).mean())))
