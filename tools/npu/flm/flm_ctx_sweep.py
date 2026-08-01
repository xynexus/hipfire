"""Sweep FLM's decode rate against CONTEXT DEPTH, out to the full window.

`flm-benchmarks.md` stops at 3135 tokens -- 2.4% of a 131072 window -- and its
"~13% decay" is the shallow end of a curve that reaches -82%. This measures the
rest of it. Results and what they imply are in docs/npu/next-phase-goals.md.

FLM picks its port; it is NOT 11434 here. `ss -ltnp | grep flm` finds it, and the
server is user-owned: query it, never restart it. Each deep point costs minutes
(TTFT alone is 232 s at 98K), so the whole sweep is a ~10 minute run.

A refusal is instant and returns {"error": "Max length reached"} -- that is the
window being FULL (prompt + generation must fit), not a lower runtime cap.
"""

import json, urllib.request, time, sys
U = "http://127.0.0.1:52625/api/generate"   # `ss -ltnp | grep flm`
para=("The quick brown fox jumps over the lazy dog near the riverbank while "
      "seventeen curious badgers observe the proceedings with mild interest. ")
# Calibrated: 24.9 tokens per paragraph copy, plus FLM's 36-token chat template
# (which carries the DATE, so it shifts day to day -- see decode.py).
TOK_PER = 24.9; OVER = 36
def ask(prompt, npred=32, timeout=1200):
    body=json.dumps({"model":"llama3.2:1b","prompt":prompt,"stream":False,
                     "options":{"temperature":0,"num_predict":npred}}).encode()
    t0=time.time()
    req=urllib.request.Request(U,body,{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=timeout)), time.time()-t0
print(f"{'target':>8} {'p_tok':>7} {'c_tok':>6} {'ttft_s':>7} {'pf_tps':>8} {'dec_tps':>8} {'wall_s':>7}")
for target in (1024,2048,4096,8192,16384,32768,65536,98304,131072):
    n=max(1,int((target-OVER)/TOK_PER))
    try:
        r,wall=ask(para*n)
    except Exception as e:
        print(f"{target:>8}  ERROR {type(e).__name__}: {str(e)[:90]}"); sys.stdout.flush(); continue
    p=r.get("prompt_eval_count",0); c=r.get("eval_count",0)
    pd=r.get("prompt_eval_duration",1)/1e9; ed=r.get("eval_duration",1)/1e9
    print(f"{target:>8} {p:>7} {c:>6} {pd:>7.2f} {p/pd if pd else 0:>8.1f} "
          f"{c/ed if ed else 0:>8.2f} {wall:>7.1f}")
    sys.stdout.flush()
