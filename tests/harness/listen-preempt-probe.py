import json, os, socket, subprocess, sys, time, threading

SC = os.path.dirname(os.path.abspath(__file__))
DAEMON = sys.argv[1]
SOCK = f"{SC}/hf-{os.getpid()}.sock"
MODEL = "/home/sadara/.hipfire/models/Qwen3.6-35B-A3B--oq4.hfq"
env = dict(os.environ)
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1); env[k] = v
env.setdefault("HIPFIRE_KV_MODE", "kvarn")
env["HIPFIRE_DAEMON_EXECUTOR"] = "v2"
env["HIPFIRE_DAEMON_TRACE"] = "1"

if os.path.exists(SOCK): os.unlink(SOCK)
errf = open(f"{SC}/listen.err", "w")
proc = subprocess.Popen([DAEMON, "--listen", SOCK], env=env, stdout=subprocess.DEVNULL, stderr=errf)

def conn():
    for _ in range(600):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(SOCK); return s
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    raise SystemExit("socket never appeared")

def send(s, obj): s.sendall((json.dumps(obj) + "\n").encode())

def reader(s, sink, first_token_evt=None):
    buf = b""
    while True:
        try: d = s.recv(65536)
        except OSError: return
        if not d: return
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip(): continue
            try: o = json.loads(line)
            except Exception: continue
            sink.append((time.monotonic(), o))
            if first_token_evt is not None and o.get("type") == "token":
                first_token_evt.set()

longp = ("Explain, in careful technical detail, how virtual memory paging works in a modern "
         "operating system kernel, covering page tables, the TLB, page faults, demand paging, "
         "copy-on-write, swapping policy, huge pages, and the interaction with DMA and IOMMU. ") * 4

a = conn(); a_frames = []
threading.Thread(target=reader, args=(a, a_frames), daemon=True).start()
send(a, {"type": "load", "model": MODEL, "params": {"max_seq": 4096, "kv_mode": env["HIPFIRE_KV_MODE"]}})
while not any(o.get("type") == "loaded" for _, o in a_frames): time.sleep(0.2)
print("  loaded")

send(a, {"type": "generate", "id": "bulk", "session_id": "lb",
         "prompt": longp, "max_tokens": 8, "temperature": 0.0})
# Let the bulk prefill get genuinely under way before the realtime request lands.
time.sleep(float(env.get("PROBE_DELAY", "2.0")))

b = conn(); b_frames = []; got = threading.Event()
threading.Thread(target=reader, args=(b, b_frames, got), daemon=True).start()
t_send = time.monotonic()
send(b, {"type": "generate", "id": "rt", "session_id": "lr",
         "prompt": "Hello.", "max_tokens": 8, "temperature": 0.0, "priority": 9})
ok = got.wait(timeout=180)
t_first = time.monotonic()
print(f"  realtime CLIENT send -> first token = {(t_first - t_send)*1000:9.1f} ms" if ok else "  realtime: NO TOKEN in 180s")

for _ in range(1800):
    if any(o.get("type") == "done" and o.get("id") == "bulk" for _, o in a_frames): break
    time.sleep(0.2)
send(a, {"type": "executor_trace"}); time.sleep(3.0)
tr = next((o for _, o in a_frames if o.get("type") == "executor_trace"), None)
if tr:
    adm = {}; prio = {}; ft = {}
    for r in tr.get("records") or []:
        s_, e = r.get("stream"), r["event"]
        if e == "admitted" and s_ not in adm: adm[s_] = r["t_ns"]; prio[s_] = r["aux"]
        if e == "token_emitted" and s_ not in ft: ft[s_] = r["t_ns"]
    for s_ in sorted(adm, key=lambda x: -prio[x]):
        v = f"{(ft[s_]-adm[s_])/1e6:9.1f} ms" if s_ in ft else "  no token"
        print(f"      trace p{prio[s_]}: admission -> first token = {v}")
send(a, {"type": "unload"}); time.sleep(2)
proc.terminate()
try: proc.wait(timeout=20)
except subprocess.TimeoutExpired: proc.kill()
if os.path.exists(SOCK): os.unlink(SOCK)
