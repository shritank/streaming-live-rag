"""Stats of a stream_asr.py run: identical-to-offline, per-decode and
post-speech p50/p95/p99/max (nearest-rank, nothing excluded), slow decodes.

    python handoff/newmachine/stream_stats.py STREAM.json OFFLINE.json [IDS.txt]
"""
import json
import math
import sys


def pct(a, q):
    s = sorted(a)
    return s[max(0, math.ceil(q * len(s)) - 1)]


st = json.load(open(sys.argv[1], encoding="utf-8"))
off = json.load(open(sys.argv[2], encoding="utf-8"))["items"]
items = st["items"]
if len(sys.argv) > 3:
    keep = set(open(sys.argv[3]).read().split())
    items = {k: v for k, v in items.items() if k in keep}
norm = lambda t: " ".join(t.split())
same = sum(norm(v["text"]) == norm(off[k]["text"]) for k, v in items.items())
dec = [d for v in items.values() for d in v["decode_ms"]]
post = [v["asr_post_speech_ms"] for v in items.values()]
out = {"meta": st["meta"], "n_clips": len(items), "identical_to_offline": same,
       "decodes": {"n": len(dec), "p50": pct(dec, .5), "p95": pct(dec, .95), "p99": pct(dec, .99), "max": max(dec),
                   "n_over_1000ms": sum(d > 1000 for d in dec), "n_over_3000ms": sum(d > 3000 for d in dec)},
       "asr_post_speech_ms": {"n": len(post), "p50": pct(post, .5), "p95": pct(post, .95), "p99": pct(post, .99),
                              "max": max(post)},
       "sum_decode_s": round(sum(dec) / 1000, 1)}
print(json.dumps(out, indent=1))
