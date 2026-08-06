#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Whole-model numpy forward for Qwen3.5-0.8B, checked against the RUNTIME.

Independent implementation of all 24 layers — linear_attn and full attention,
QK-norm, partial rope, GQA, the attention output gate — from bf16 safetensors.

The oracle is `dump_logits_qwen35`, which runs the real prefill path on the
deterministic prompt 0,1,2,... and dumps the last position's logits. That path
is trustworthy: generating from the same artifact through `hipfire chat`
produces coherent text. (The per-layer hidden-state dumper is NOT trustworthy —
see docs/linear-attn-real-model-status.md.)

    cargo run --release -p hipfire-runtime --features deltanet \
      --example dump_logits_qwen35 -- <model.hfq> ref_logits.f32 --prefill 64

Result as of writing: cos 0.566. The runtime's top-2 after 0..63 includes 64 —
it continues the count. This implementation's top-2 includes 63, the CURRENT
token, which is what a tied-embedding model returns when the layers barely move
the residual stream. Measured branch magnitudes agree: attention 0.0135 and MLP
0.0011 against a residual of 0.0206, where the runtime roughly triples the
stream in layer 0 alone.

Slow on purpose — plain numpy, ~6 min for 64 tokens. A correctness oracle, not
a benchmark.
"""

import json

import numpy as np

F='/home/sadara/.claude/jobs/9a1047b0/tmp/q08b/model.safetensors-00001-of-00001.safetensors'
fh=open(F,'rb'); n=int.from_bytes(fh.read(8),'little'); H=json.loads(fh.read(n)); B=8+n
def T(nm):
    i=H[nm]; o=i['data_offsets']; fh.seek(B+o[0]); raw=fh.read(o[1]-o[0])
    if i['dtype']=='F32': return np.frombuffer(raw,dtype='<f4').reshape(i['shape'])
    u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16
    return np.frombuffer(u.astype('<u4').tobytes(),dtype='<f4').reshape(i['shape'])
cfg=json.load(open('/home/sadara/.claude/jobs/9a1047b0/tmp/q08b/config.json'))['text_config']
P='model.language_model.'; eps=cfg['rms_norm_eps']
d='/home/sadara/.claude/jobs/9a1047b0/tmp/oracle/'
toks=np.arange(64,dtype=np.int64)  # deterministic prompt 0..63, matching dump_logits_qwen35
S=len(toks)
silu=lambda v: v/(1+np.exp(-v))
rms=lambda v,w: v/np.sqrt((v**2).mean(-1,keepdims=True)+eps)*w
E=T(P+'embed_tokens.weight'); x=E[toks].astype(np.float32)
hd=cfg['head_dim']; nH=cfg['num_attention_heads']; nKV=cfg['num_key_value_heads']
rp=cfg['rope_parameters']; theta=rp['rope_theta']; nrot=int(hd*rp['partial_rotary_factor'])
for li,lt in enumerate(cfg['layer_types']):
    L=f'{P}layers.{li}.'
    xn1=rms(x,T(L+'input_layernorm.weight'))
    if lt=='linear_attention':
        A=L+'linear_attn.'
        qkv=xn1@T(A+'in_proj_qkv.weight').T; cw=T(A+'conv1d.weight'); C=qkv.shape[1]; K=cw.shape[-1]
        pre=np.zeros_like(qkv)
        for t in range(S):
            a=np.zeros(C,dtype=np.float32)
            for j in range(K):
                s0=t-(K-1-j)
                if s0>=0: a+=cw[:,0,j]*qkv[s0]
            pre[t]=a
        nh=T(A+'A_log').shape[0]; hv=T(A+'norm.weight').shape[0]; hk=(C//nh-hv)//2
        q=silu(pre[:,:nh*hk]); k=silu(pre[:,nh*hk:2*nh*hk]); v=silu(pre[:,2*nh*hk:])
        def l2(a,dh,sc=1.0):
            a=a.reshape(S,nh,dh); g=1/np.sqrt((a**2).sum(-1,keepdims=True)+eps); return (a*g*sc).reshape(S,nh*dh)
        q=l2(q,hk,1/np.sqrt(hk)); k=l2(k,hk)
        gate=np.logaddexp(0,xn1@T(A+'in_proj_a.weight').T+T(A+'dt_bias'))*(-np.exp(T(A+'A_log')))
        al=np.exp(gate); be=1/(1+np.exp(-(xn1@T(A+'in_proj_b.weight').T)))
        St=np.zeros((nh,hv,hk),dtype=np.float32); o=np.zeros((S,nh*hv),dtype=np.float32)
        for t in range(S):
            kt=k[t].reshape(nh,hk); vt=v[t].reshape(nh,hv); qt=q[t].reshape(nh,hk)
            kv=np.einsum('hvk,hk->hv',St,kt); dl=(vt-al[t][:,None]*kv)*be[t][:,None]
            St=al[t][:,None,None]*St+dl[:,:,None]*kt[:,None,:]
            o[t]=np.einsum('hvk,hk->hv',St,qt).reshape(-1)
        z=xn1@T(A+'in_proj_z.weight').T; nw=T(A+'norm.weight')
        oo=o.reshape(S,nh,hv); g=1/np.sqrt((oo**2).mean(-1,keepdims=True)+eps)
        br=((oo*g*nw).reshape(S,nh*hv)*silu(z))@T(A+'out_proj.weight').T
    else:
        A=L+'self_attn.'
        qf=xn1@T(A+'q_proj.weight').T
        qh=qf.reshape(S,nH,2,hd)[:,:,0,:]; gt=qf.reshape(S,nH,2,hd)[:,:,1,:]
        kk=(xn1@T(A+'k_proj.weight').T).reshape(S,nKV,hd); vv=(xn1@T(A+'v_proj.weight').T).reshape(S,nKV,hd)
        qh=qh/np.sqrt((qh**2).mean(-1,keepdims=True)+eps)*T(A+'q_norm.weight')
        kk=kk/np.sqrt((kk**2).mean(-1,keepdims=True)+eps)*T(A+'k_norm.weight')
        pos=np.arange(S)[:,None]; inv=1.0/(theta**(np.arange(0,nrot,2)/nrot))
        ang=pos*inv; cs=np.cos(ang); sn=np.sin(ang)
        def rope(a):
            a=a.copy(); e=a[...,0:nrot:2].copy(); o2=a[...,1:nrot:2].copy()
            a[...,0:nrot:2]=e*cs[:,None,:]-o2*sn[:,None,:]; a[...,1:nrot:2]=e*sn[:,None,:]+o2*cs[:,None,:]
            return a
        qh=rope(qh); kk=rope(kk)
        rep=nH//nKV; kx=np.repeat(kk,rep,axis=1); vx=np.repeat(vv,rep,axis=1)
        sc=1.0/np.sqrt(hd); att=np.einsum('thd,shd->hts',qh,kx)*sc
        mask=np.triu(np.full((S,S),-1e30,dtype=np.float32),1); att=att+mask[None]
        att=np.exp(att-att.max(-1,keepdims=True)); att/=att.sum(-1,keepdims=True)
        ctx=np.einsum('hts,shd->thd',att,vx).reshape(S,nH*hd)
        ctx=ctx*(1/(1+np.exp(-gt.reshape(S,nH*hd))))
        br=ctx@T(A+'o_proj.weight').T
    x=x+br
    xn2=rms(x,T(L+'post_attention_layernorm.weight'))
    x=x+((silu(xn2@T(L+'mlp.gate_proj.weight').T)*(xn2@T(L+'mlp.up_proj.weight').T))@T(L+'mlp.down_proj.weight').T)
hn=rms(x,T(P+'norm.weight')); lg=hn@E.T
lp=lg-lg.max(-1,keepdims=True); ls=lp-np.log(np.exp(lp).sum(-1,keepdims=True))
ref=np.fromfile('/home/sadara/.claude/jobs/9a1047b0/tmp/oracle/ref_logits.f32',dtype=np.float32)
mine=lg[-1]  # runtime dumps the LAST position's logits
c=float((mine*ref).sum()/(np.linalg.norm(mine)*np.linalg.norm(ref)+1e-12))
print(f'last-position logits: cos(mine, runtime) = {c:.4f}')
print(f'  mine  top5 ids {np.argsort(-mine)[:5].tolist()}  rms {float(np.sqrt((mine**2).mean())):.4f}')
print(f'  runtime top5 ids {np.argsort(-ref)[:5].tolist()}  rms {float(np.sqrt((ref**2).mean())):.4f}')
