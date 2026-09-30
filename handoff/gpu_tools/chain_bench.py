"""Where do ~80 ms go in ONE post-speech sub-query (search -> prefetch) on
the GPU? Real production components (build_real_components), real dev
questions, the same call path the engine uses (asyncio.to_thread hops
included). Per stage: e5 encode, BM25, CE rerank, CE claim selection, gate
(CE sentence + reader), whole search, whole prefetch, whole chain.

Modes: warm = back-to-back; idle = sleep GAP s before each question
(the GPU and the worker threads go idle between utterances, as in speech).

    python handoff/gpu_tools/chain_bench.py [GAP_S] [N]
"""
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from streaming_rag.build import build_real_components  # noqa: E402
from streaming_rag.config import load_config  # noqa: E402
from streaming_rag.contracts import SubQuery  # noqa: E402
from streaming_rag.retrieval import bm25 as bm25_mod, neural  # noqa: E402
from streaming_rag.session import synthesis as S  # noqa: E402

T = defaultdict(list)


def timed(name, fn):
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name].append((time.perf_counter() - t) * 1000)
    return w


neural.E5Embedder.encode = timed("e5_encode", neural.E5Embedder.encode)
bm25_mod.BM25Index.top_k = timed("bm25", bm25_mod.BM25Index.top_k)
neural.CrossEncoder.score = timed("ce_score_call", neural.CrossEncoder.score)
neural.SquadReader.answer = timed("reader_call", neural.SquadReader.answer)
S.GroundedSynthesizer._ce_claim = timed("claim_select", S.GroundedSynthesizer._ce_claim)
S.GroundedSynthesizer._answer_probability = timed("gate", S.GroundedSynthesizer._answer_probability)


async def main():
    gap = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    cfg = load_config()
    cfg.corpus_dir = "data/corpus"
    _, retriever, synth, _, _ = build_real_components(cfg)
    await retriever.setup()
    await synth.setup()
    qs = []
    for p in sorted(Path("handoff/newmachine/scen_stream_ada_100").glob("*.json"))[:n]:
        ev = json.loads(p.read_text(encoding="utf-8"))["events"]
        qs.append("".join(e["payload"].get("text", "") for e in ev if e["event_type"] == "transcript_chunk").strip())
    out = {}
    for mode in ("warm", "idle"):
        T.clear()
        chain = []
        for i, q in enumerate(qs):
            if mode == "idle":
                await asyncio.sleep(gap)
            sq = SubQuery(query_id=f"q{i}", text=q, intent_label=q, trigger="final", utterance_id="u")
            t0 = time.perf_counter()
            r = (await retriever.search([sq], k=cfg.retrieval.k))[0]
            t1 = time.perf_counter()
            await asyncio.to_thread(synth.prefetch, sq, r)
            t2 = time.perf_counter()
            T["search_total"].append((t1 - t0) * 1000)
            T["prefetch_total"].append((t2 - t1) * 1000)
            chain.append((t2 - t0) * 1000)
        T["chain_total"] = chain
        out[mode] = {k: {"n": len(v), "p50": round(float(np.percentile(v, 50)), 2),
                         "p95": round(float(np.percentile(v, 95)), 2), "max": round(max(v), 2)}
                     for k, v in T.items()}
    out["providers"] = sorted(set(neural.ACTIVE_PROVIDERS.values()))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
