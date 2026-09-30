"""Item-level identity check between two eval.run_quality outputs:
does every scenario produce the same final answer text + citations?

    python handoff/newmachine/diff_answers.py A.json B.json
"""
import json
import sys


def finals(path):
    raw = json.load(open(path, encoding="utf-8"))["raw"]
    out = {}
    for sid, r in raw.items():
        turns = {}
        for t in r["results"]:
            if t.get("kind") == "turn_result":
                turns[t["utterance_id"]] = (t.get("answer") or "", tuple(t.get("citations") or []),
                                            t.get("uncertainty") or "")
        out[sid] = turns
    return out


a, b = finals(sys.argv[1]), finals(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [s for s in common if a[s] != b[s]]
print(f"scenarios: A={len(a)} B={len(b)} common={len(common)} differing={len(diff)}")
for s in diff[:10]:
    for u in sorted(set(a[s]) | set(b[s])):
        if a[s].get(u) != b[s].get(u):
            print(f"  {s}/{u}\n    A: {a[s].get(u)}\n    B: {b[s].get(u)}")
