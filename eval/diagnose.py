"""python -m eval.diagnose --scenarios eval/scenarios_real_corpus --corpus-dir data/corpus

Exhaustive failure-case diagnosis. For every gold (query, expected_doc) pair
in every scenario's ground truth, replays the ACTUAL retrieval + relevance +
grounding pipeline the engine uses (not an approximation) and classifies any
failure into exactly one root-cause bucket:

  A - retrieval failure: the gold Doc_ID never appears anywhere in the
      top-k retrieved evidence — a ranking/recall miss upstream of
      synthesis. Nothing downstream could have saved this turn.
  B - synthesizer over-rejection: the gold Doc_ID WAS retrieved, but the
      query-relevance gate (session/synthesis.py::_is_relevant) or the
      post-hoc grounding check (session/grounding.py::claim_support)
      rejected it before/after it became a claim.
  C - late-anchor timing: an eligible turn's first retrieval fired at or
      after utterance_end (a G2 miss).

This directly exercises the same components eval/run_all uses
(streaming_rag.build.build_real_components), so the classification matches
what score_g4/score_g2 actually measure, not an approximation of it. It is
eval-side tooling, never imported by streaming_rag/.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from streaming_rag.config import load_config
from streaming_rag.contracts import SubQuery
from streaming_rag.session import grounding
from streaming_rag.session.synthesis import MIN_QUERY_RELEVANCE
from streaming_rag.telemetry.trace import assemble_turns

from .loader import discover_scenarios, load_ground_truth
from .runner import run_scenario


@dataclass
class Finding:
    scenario: str
    utterance_id: str
    bucket: str
    query: str = ""
    gold_doc: str = ""
    detail: str = ""
    scores: dict = field(default_factory=dict)


def _doc_of_citation(citation: str) -> str:
    return citation.split(" §")[0]


async def _diagnose_grounding(retriever, synth, gold_query: str, gold_doc: str) -> Finding | None:
    """Replays retrieval + the query-relevance gate + post-hoc grounding for
    ONE gold (query, doc) pair, exactly mirroring session/synthesis.py's
    _claims_from logic, and reports why it failed if it did."""
    q = SubQuery(query_id="diag", text=gold_query, intent_label=gold_query[:40],
                 trigger="final", utterance_id="diag")
    results = await retriever.search([q], k=8)
    result = results[0]

    retrieved_docs = [_doc_of_citation(e.chunk.citation) for e in result.evidence]
    if gold_doc not in retrieved_docs:
        return Finding(bucket="A", scenario="", utterance_id="", query=gold_query, gold_doc=gold_doc,
                        detail=f"retrieved_docs={retrieved_docs}", scores={"low_confidence": result.low_confidence})

    rank = retrieved_docs.index(gold_doc)
    evidence = result.evidence[rank]
    sentence = synth._best_sentence(gold_query, evidence.chunk.text)
    relevant = synth._is_relevant(gold_query, sentence, synth._specific_terms())
    from streaming_rag.retrieval.text import content_tokens
    q_tokens = set(content_tokens(gold_query))
    s_tokens = set(content_tokens(sentence))
    overlap = q_tokens & s_tokens
    frac = len(overlap) / len(q_tokens) if q_tokens else 1.0

    if not relevant:
        return Finding(
            bucket="B", scenario="", utterance_id="", query=gold_query, gold_doc=gold_doc,
            detail=f"gold doc retrieved at rank {rank}, but query-relevance gate rejected the "
                   f"extracted sentence: overlap_fraction={frac:.2f} (threshold={MIN_QUERY_RELEVANCE}), "
                   f"overlap_tokens={sorted(overlap)}/{sorted(q_tokens)}, sentence={sentence[:100]!r}",
            scores={"overlap_fraction": frac, "overlap_count": len(overlap), "rank": rank},
        )

    # Passed the relevance gate — would become a claim. Check the post-hoc
    # grounding support too (it always passes for extractive text by
    # construction, but verified here rather than assumed).
    support = grounding.claim_support(sentence, evidence.chunk.text)
    if support < grounding.SUPPORT_THRESHOLD:
        return Finding(
            bucket="B", scenario="", utterance_id="", query=gold_query, gold_doc=gold_doc,
            detail=f"gold doc retrieved at rank {rank} and passed the relevance gate, but "
                   f"post-hoc claim_support={support:.2f} < threshold={grounding.SUPPORT_THRESHOLD}",
            scores={"claim_support": support, "rank": rank},
        )
    return None  # correctly grounded — not a failure


async def diagnose_scenario(path: Path, config) -> list[Finding]:
    from streaming_rag.build import build_real_components
    from streaming_rag.session.store import SessionStore

    gt = load_ground_truth(path)
    _, trace = await run_scenario(str(path), config, impl="real", time_scale=8)
    turns = {t.utterance_id: t for t in assemble_turns(trace)}
    findings: list[Finding] = []

    # Fresh components sharing the same (deterministic) corpus index, purely
    # for replaying the grounding pipeline standalone per gold pair.
    controller, retriever, synth, llm, store = build_real_components(config, session_store=SessionStore())
    await retriever.setup()

    for uid, turn_gt in gt.get("turns", {}).items():
        turn = turns.get(uid)
        if turn is None:
            continue

        # --- Bucket C: late-anchor timing (G2) ---
        if turn_gt.get("eligible_for_early_retrieval"):
            chunk_events = turn.of("chunk_received")
            end_ts = next((e.get("ts_ms") for e in chunk_events if e.get("is_final")), None)
            first_retrieval = turn.first("retrieval_started")
            if end_ts is not None and (first_retrieval is None or first_retrieval.get("ts_ms", 1e18) >= end_ts):
                decisions = [f"{e.get('reason')}(stab={e.get('stability'):.2f})"
                             for e in turn.of("controller_decision")]
                findings.append(Finding(
                    scenario=path.name, utterance_id=uid, bucket="C",
                    detail=f"first_retrieval={'none' if first_retrieval is None else first_retrieval.get('ts_ms')} "
                           f"utterance_end={end_ts}; decisions={decisions}",
                ))

        # --- Bucket A / B: grounding (G4), one gold (query, doc) pair at a time ---
        gold_intents = turn_gt.get("gold_sub_intents", [])
        expected_docs = turn_gt.get("expected_citation_docs", [])
        for i, gold_query in enumerate(gold_intents):
            gold_doc = _doc_of_citation(expected_docs[i]) if i < len(expected_docs) else \
                (_doc_of_citation(expected_docs[0]) if expected_docs else "")
            if not gold_doc:
                continue
            finding = await _diagnose_grounding(retriever, synth, gold_query, gold_doc)
            if finding is not None:
                finding.scenario = path.name
                finding.utterance_id = uid
                findings.append(finding)

    return findings


async def run(scenario_dir: str, corpus_dir: str) -> list[Finding]:
    config = load_config()
    config.corpus_dir = corpus_dir
    scenarios = discover_scenarios(scenario_dir)
    all_findings: list[Finding] = []
    for path in scenarios:
        all_findings.extend(await diagnose_scenario(path, config))
    return all_findings


def render_report(findings: list[Finding]) -> str:
    by_bucket: dict[str, list[Finding]] = {"A": [], "B": [], "C": []}
    for f in findings:
        by_bucket[f.bucket].append(f)

    lines = ["# Exhaustive Failure-Case Diagnosis (real corpus)", ""]
    lines.append(f"Total findings: {len(findings)} "
                  f"(A={len(by_bucket['A'])}, B={len(by_bucket['B'])}, C={len(by_bucket['C'])})")
    lines.append("")
    for bucket, label in [("A", "Retrieval failures (gold doc never retrieved, top-8)"),
                            ("B", "Synthesizer over-rejection (gold doc retrieved, not asserted)"),
                            ("C", "Late-anchor timing (G2 miss)")]:
        lines.append(f"## Bucket {bucket}: {label} ({len(by_bucket[bucket])})")
        lines.append("")
        if not by_bucket[bucket]:
            lines.append("_none_")
        for f in by_bucket[bucket]:
            header = f"- **{f.scenario}** / {f.utterance_id}"
            if f.gold_doc:
                header += f" (gold_doc={f.gold_doc})"
            lines.append(header)
            if f.query:
                lines.append(f"  - query: {f.query!r}")
            lines.append(f"  - {f.detail}")
            if f.scores:
                lines.append(f"  - scores: {f.scores}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios_real_corpus")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--out", default="reports/failure_diagnosis.md")
    args = parser.parse_args(argv)

    findings = asyncio.run(run(args.scenarios, args.corpus_dir))
    report = render_report(findings)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n(written to {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
