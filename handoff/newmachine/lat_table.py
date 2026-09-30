"""Side-by-side p50 / p95 / p99 / max (n) of eval.latency_profile outputs.

    python handoff/newmachine/lat_table.py LABEL=file.json [LABEL=file.json ...]
"""
import json
import sys

STAGES = ["controller.decision_ms", "engine.final_decision_ms", "engine.synthesis_ms", "engine.post_speech_ms",
          "engine.retrieval_wait_ms", "retrieval.subquery_ms", "retrieval.bm25_ms", "retrieval.e5_encode_ms",
          "retrieval.dense_topk_ms", "ce.rerank_ms", "synthesis.claim_select_ms", "synthesis.gate_ms",
          "reader.other_ms", "synthesis.grounding_ms", "synthesis.prefetch_ms"]

runs = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]
print("| Stage | " + " | ".join(f"{lab} p50 / p95 / p99 / max" for lab, _ in runs) + " | n |")
print("|---|" + "---|" * (len(runs) + 1))
for s in STAGES:
    cells, n = [], ""
    for _, r in runs:
        v = r["stages"].get(s)
        cells.append(f"{v['p50']:.1f} / {v['p95']:.1f} / {v['p99']:.1f} / {v['max']:.1f}" if v else "-")
        n = v["n"] if v else n
    print(f"| {s} | " + " | ".join(cells) + f" | {n} |")
print("| cold start (s) | " + " | ".join(str(r.get("cold_start_s")) for _, r in runs) + " | |")
