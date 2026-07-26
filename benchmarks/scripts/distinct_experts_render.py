#!/usr/bin/env python3
"""Render the distinct_experts(B) results doc from one or more result JSONs."""
import json, sys, math

def load(p):
    with open(p) as f: return json.load(f)

def frac(v, E): return v / E

def get(agg, B):
    return agg.get(str(B), agg.get(B))

def table(res):
    E = res["E"]; k = res["k"]; Bs = res["Bs"]
    d = res["depth_aggregate"]; w = res["width_aggregate"]
    r = res["random_empirical_aggregate"]; a = res["analytic_random"]
    lines = []
    lines.append(f"| B | depth | depth %E | width | width %E | rand-emp | analytic-uniform |")
    lines.append("|---|---|---|---|---|---|---|")
    for B in Bs:
        dv=get(d,B); wv=get(w,B); rv=get(r,B); av=get(a,B)
        def f(x): return f"{x:.2f}" if x is not None else "—"
        def fp(x): return f"{100*x/E:.0f}%" if x is not None else "—"
        lines.append(f"| {B} | {f(dv)} | {fp(dv) if dv else '—'} | {f(wv)} | {fp(wv) if wv else '—'} | {f(rv)} | {f(av)} |")
    return "\n".join(lines)

def verdict(res):
    E=res["E"]; Bs=res["Bs"]
    d=res["depth_aggregate"]; w=res["width_aggregate"]
    out=[]
    # ratio width/depth at largest common B
    common=[B for B in Bs if get(d,B) and get(w,B)]
    Bmax=max(common)
    dv=get(d,Bmax); wv=get(w,Bmax)
    out.append(f"At B={Bmax}: depth touches {dv:.1f}/{E} ({100*dv/E:.0f}%), "
               f"width touches {wv:.1f}/{E} ({100*wv/E:.0f}%). "
               f"width/depth = {wv/dv:.2f}.")
    # per-token marginal: new experts added per doubling
    return "\n".join(out)

def perlayer_spread(res, per_key, Bs_show):
    import statistics
    pl = res[per_key]  # {layer: {B: val}}
    lines=[f"| B | min | median | max | (over {len(pl)} MoE layers) |",
           "|---|---|---|---|---|"]
    for B in Bs_show:
        vals=[get(d,B) for d in pl.values() if get(d,B) is not None]
        if not vals: continue
        lines.append(f"| {B} | {min(vals):.2f} | {statistics.median(vals):.2f} | {max(vals):.2f} | |")
    return "\n".join(lines)

NAME={}
def main():
    args=sys.argv[1:]
    paths=[]
    for a in args:
        if "=" in a:
            n,p=a.split("=",1); NAME[p]=n; paths.append(p)
        else:
            paths.append(a)
    for p in paths:
        res=load(p)
        nm=NAME.get(p, res['model'].split('/')[-1])
        print(f"### {nm}  (E={res['E']}, k={res['k']}, {res['num_moe_layers']} MoE layers)\n")
        print(table(res)); print()
        print(verdict(res)); print()
        Bshow=[b for b in [8,16,32,64,128] if b in res["Bs"]]
        print("Depth per-layer spread:\n"+perlayer_spread(res,"depth_per_layer",Bshow)); print()
        print("Width per-layer spread:\n"+perlayer_spread(res,"width_per_layer",Bshow)); print()

if __name__=="__main__":
    main()
