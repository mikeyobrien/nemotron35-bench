#!/usr/bin/env python3
"""Concurrency probe: fire N parallel completions at the server, report per-request
wall time, tokens, decode t/s, and any failures. Usage: python3 concurrency_probe.py N [tokens]"""
import concurrent.futures, json, sys, time, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
URL = "http://127.0.0.1:8080/v1/chat/completions"

def one(i):
    body = json.dumps({
        "model": "nemotron35-lightning-30b-a3b-nvfp4",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": f"Request {i}: Write a short paragraph explaining what concurrency means in GPU inference, in plain language."},
        ],
        "max_tokens": TOKENS,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        wall = time.time() - t0
        out = d["choices"][0]["message"]["content"]
        nt = d.get("usage", {}).get("completion_tokens", len(out.split()))
        return {"i": i, "ok": True, "wall_s": round(wall, 2), "tokens": nt,
                "tps": round(nt / wall, 2) if wall else 0, "err": None}
    except Exception as e:
        return {"i": i, "ok": False, "wall_s": round(time.time() - t0, 2),
                "tokens": 0, "tps": 0, "err": str(e)[:120]}

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(one, range(N)))
total = time.time() - t0
oks = [r for r in results if r["ok"]]
print(f"=== {N} concurrent requests, {TOKENS} max tokens each ===")
print(f"wall total (all N, serialized view): {total:.1f}s")
for r in sorted(results, key=lambda x: x["i"]):
    st = "OK " if r["ok"] else "FAIL"
    print(f"  req {r['i']}: {st} wall={r['wall_s']}s tokens={r['tokens']} tps={r['tps']} {r['err'] or ''}")
if oks:
    avg_wall = sum(r["wall_s"] for r in oks) / len(oks)
    print(f"per-request avg wall: {avg_wall:.1f}s | aggregate tps: {sum(r['tokens'] for r in oks)/total:.1f}")
    print(f"FAILURES: {N - len(oks)}/{N}")
else:
    print("ALL FAILED")
