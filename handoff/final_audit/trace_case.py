"""Trace one question through the production pipeline stage by stage, to root-cause an edge-case failure.

    python handoff/final_audit/trace_case.py CORPUS_DIR "question" "gold answer substring" [more substrings]

Prints: the top evidence (rank, citation, rerank logit, whether the chunk holds the gold answer), the
cross-encoder's best sentences (and where the gold-answer sentence ranks), the chosen claim, and the refusal
gate's six features and probability - recomputed here and cross-checked against the synthesizer's own
number, so the trace cannot silently drift from the code. Uses the same components as the engine; run it
with STREAMING_RAG_ORT_PROVIDER=cuda (GPU only).
"""
import asyncio
import math
import os
import sys

sys.path.insert(0, os.getcwd())
from streaming_rag.build import build_real_components  # noqa: E402
from streaming_rag.config import load_config  # noqa: E402
from streaming_rag.contracts import SubQuery  # noqa: E402
from streaming_rag.retrieval.neural import get_cross_encoder, get_reader  # noqa: E402
from streaming_rag.retrieval.text import content_tokens  # noqa: E402
from streaming_rag.session import synthesis as S  # noqa: E402


def has(text, golds):
    t = text.lower()
    return any(g.lower() in t for g in golds)


async def main(corpus, question, golds):
    cfg = load_config()
    cfg.corpus_dir = corpus
    _, retriever, synth, _, _ = build_real_components(cfg)
    await retriever.setup()
    await synth.setup()
    sq = SubQuery(query_id="q1", text=question, intent_label=question, trigger="final", utterance_id="u1")
    res = (await retriever.search([sq], k=cfg.retrieval.k))[0]
    n3 = cfg.synthesis.ce_evidence_chunks
    print(f"QUESTION: {question!r}   gold answer contains: {golds}")
    print(f"low_confidence={res.low_confidence}  evidence={len(res.evidence)}  (claims use the top {n3} chunks)")
    for i, e in enumerate(res.evidence[:6]):
        print(f"  #{i} {e.chunk.citation:14s} logit {e.score:7.2f}  answer_in_chunk={has(e.chunk.text, golds)}  | {e.chunk.text[:90]!r}")

    cands = []
    for e in res.evidence[:n3]:
        title = e.chunk.metadata.get("doc_title", "")
        for s in synth._evidence_sentences(e):
            if content_tokens(s):
                cands.append((e.chunk.citation, s, f"{title}: {s}" if title else s))
    scores = get_cross_encoder().score(question, [c[2] for c in cands])
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    print(f"\nCE sentence ranking over {len(cands)} candidate sentences (claim threshold {cfg.synthesis.ce_claim_threshold}):")
    for rank, i in enumerate(order[:5]):
        print(f"  {rank + 1}. {scores[i]:7.2f} {cands[i][0]:14s} answer_in_sentence={has(cands[i][1], golds)} | {cands[i][1][:110]!r}")
    gold_rank = next((r + 1 for r, i in enumerate(order) if has(cands[i][1], golds)), None)
    print(f"  -> best sentence holding the gold answer ranks: {gold_rank} of {len(cands)}")

    claim = synth._ce_claim(question, res)
    if claim is None:
        print("\nCLAIM: none (below the sentence threshold)")
        return
    print(f"\nCLAIM {claim.citations}: answer_in_claim={has(claim.text, golds)} | {claim.text[:200]!r}")
    if cfg.synthesis.refusal_gate == "learned":
        model = S._refusal_model()
        top = res.evidence[0]
        title = top.chunk.metadata.get("doc_title", "")
        section = synth._resolve_citation(top.chunk.citation)
        margin, span = get_reader().answer(question, [e.chunk.text for e in res.evidence[:n3]])
        values = {"top": top.score,
                  "sentence": float(get_cross_encoder().score(question, [f"{title}: {claim.text}"])[0]),
                  "overlap_ce": S._overlap(question, claim.text),
                  "overlap_section": S._overlap(question, section.text) if section is not None else 0.0,
                  "margin": top.score - res.evidence[1].score if len(res.evidence) > 1 else 0.0,
                  "reader_margin": -10.0 if margin is None else margin}
        z = model["intercept"]
        print("GATE features (value | standardised | coefficient | contribution to z):")
        for name, mean, scale, w in zip(model["features"], model["scaler_mean"], model["scaler_scale"], model["coef"], strict=True):
            std = (values[name] - mean) / scale
            z += w * std
            print(f"  {name:16s} {values[name]:8.2f} | {std:6.2f} | {w:6.2f} | {w * std:6.2f}")
        p = 1 / (1 + math.exp(-z))
        own = synth._answer_probability(question, res, claim)
        print(f"  P(correct) = {p:.4f}   (synthesizer's own value {own:.4f}; must match)   threshold {model['threshold']}"
              f"  -> {'ASSERT' if p >= model['threshold'] else 'REFUSE'}   reader span: {span!r}")
        assert abs(p - own) < 1e-3, "trace drifted from the synthesizer"


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
