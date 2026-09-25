"""python -m eval.error_analysis --scenarios DIR --corpus-dir DIR [--set k=v ...] [--out FILE]

Traces every gold sub-question through the pipeline and assigns each miss
exactly one cause, in pipeline order (the first stage that lost it):

  decomposition       no sub-query reaching synthesis corresponds to the
                      question (content-token Jaccard < 0.5 with every one)
  query_formulation   gold section not retrieved for the issued sub-query,
                      but retrieved when the clean question is searched
  rerank_depth        not retrieved; the clean question's gold section is in
                      the fused top-20 but not the final top-k
  first_stage         not retrieved, not even in the fused top-20
  refusal             gold section retrieved but the result was flagged
                      low-confidence, so the question was declined
  relevance_gate      gold section retrieved, confident, but no evidence
                      sentence passed the lexical relevance gate
  context_selection   a claim was made from a different, non-gold section
                      ranked above the gold one
  chunking            claim from the gold section, but the answer sits in a
                      different chunk of that section
  sentence_selection  claim from the gold chunk, but a different sentence
                      of the same chunk holds the answer
  merge_lost          a correct claim was produced but did not survive into
                      the final answer (refinement merge)
  answer_not_in_text  gold section cited, answer string absent from its text
                      (label/normalisation artefact)

Also reports ceilings (question isolated; gold section in the evidence;
answer present in some sentence of the top-3 evidence chunks) and assigns
every WRONG asserted claim a cause (spurious sub-query vs wrong passage).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from streaming_rag.config import load_config
from streaming_rag.contracts import SubQuery
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.retrieval.text import content_tokens, jaccard, split_sentences
from streaming_rag.session.store import SessionStore
from streaming_rag.session.synthesis import GroundedSynthesizer

from .loader import discover_scenarios, load_ground_truth
from .quality import contains_answer, informative_answers, load_doc_titles, load_qrels_index
from .run_quality import apply_overrides
from .runner import run_scenario_detailed

MATCH_THRESHOLD = 0.5


def _best_match(question: str, sub_queries) -> tuple[object, float]:
    q = set(content_tokens(question))
    best, score = None, 0.0
    for sq in sub_queries:
        s = jaccard(q, set(content_tokens(sq.text)))
        if s > score:
            best, score = sq, s
    return best, score


def _match(question: str, sub_queries, n_gold: int) -> tuple[object, float]:
    # One gold question and one sub-query: that sub-query IS the question,
    # however garbled (ASR input shares few exact words with the typed gold
    # question, which Jaccard matching would misread as a decomposition miss).
    if n_gold == 1 and len(sub_queries) == 1:
        return sub_queries[0], 1.0
    return _best_match(question, sub_queries)


class Analyzer:
    def __init__(self, corpus_dir: str, overrides: list[str]):
        self.config = apply_overrides(load_config(), overrides)
        self.config.corpus_dir = corpus_dir
        self.qrels = load_qrels_index(corpus_dir)
        self.titles = load_doc_titles(corpus_dir)
        self.retriever = HybridRetriever(self.config)
        fused_cfg = apply_overrides(load_config(), overrides + ["retrieval.reranker=none"])
        fused_cfg.corpus_dir = corpus_dir
        self.fused = HybridRetriever(fused_cfg)
        asyncio.run(self.retriever.setup())
        asyncio.run(self.fused.setup())
        self.synth = GroundedSynthesizer(self.retriever, SessionStore(), self.config)

    def _usable(self, row: dict) -> list[str]:
        doc = row["relevant"][0].split(" §")[0] if row["relevant"] else ""
        return informative_answers(row["answers"], self.titles.get(doc, ""))

    def _search(self, retriever, text: str, k: int):
        return retriever._search_sync(SubQuery(query_id="probe", text=text, intent_label="",
                                               trigger="final", utterance_id="probe"), k)

    def classify_miss(self, row: dict, question: str, call, final_answer, n_gold: int = 0) -> tuple[str, dict]:
        gold = set(row["relevant"])
        good = self._usable(row)
        info: dict = {}
        if call is None:
            return "decomposition", {"reason": "no synthesis call for the turn"}
        match, score = _match(question, call["sub_queries"], n_gold)
        info["best_subquery"] = match.text if match else None
        info["match"] = round(score, 2)
        if match is None or score < MATCH_THRESHOLD:
            info["sub_queries"] = [sq.text for sq in call["sub_queries"]]
            return "decomposition", info
        result = next((r for r in call["results"] if r.query_id == match.query_id), None)
        cits = [e.chunk.citation for e in result.evidence] if result else []
        info["evidence"] = cits[:5]
        if not gold & set(cits):
            k = self.config.retrieval.k
            clean = [e.chunk.citation for e in self._search(self.retriever, question, k)]
            if gold & set(clean):
                return "query_formulation", info
            fused = [e.chunk.citation for e in self._search(self.fused, question, 20)]
            return ("rerank_depth" if gold & set(fused) else "first_stage"), info
        if result.low_confidence:
            info["top_score"] = round(result.evidence[0].score, 2)
            return "refusal", info
        claims, _ = self.synth._claims_from([match], [result])
        if not claims:
            return "relevance_gate", info
        claim = claims[0]
        info["claim"] = claim.text[:200]
        info["claim_citation"] = claim.citations[0]
        if claim.citations[0] not in gold:
            return "context_selection", info
        if good and contains_answer(claim.text, good) or (not good):
            return "merge_lost", info
        # claim came from the gold section but lacks the answer: where is it?
        claim_chunk = next(e.chunk for e in result.evidence if e.chunk.citation == claim.citations[0])
        in_chunk = [s for s in split_sentences(claim_chunk.text) if contains_answer(s, good)]
        if in_chunk:
            info["answer_sentence"] = in_chunk[0][:200]
            return "sentence_selection", info
        section = self.retriever.get_chunk_by_citation(claim.citations[0])
        if section and contains_answer(section.text, good):
            return "chunking", info
        return "answer_not_in_text", info

    def ceilings(self, row: dict, question: str, call, n_gold: int = 0) -> dict:
        gold = set(row["relevant"])
        good = self._usable(row)
        out = {"isolated": False, "gold_in_evidence": False, "answer_in_top3": False}
        if call is None:
            return out
        match, score = _match(question, call["sub_queries"], n_gold)
        if match is None or score < MATCH_THRESHOLD:
            return out
        out["isolated"] = True
        result = next((r for r in call["results"] if r.query_id == match.query_id), None)
        if result is None:
            return out
        out["gold_in_evidence"] = bool(gold & {e.chunk.citation for e in result.evidence})
        if good:
            out["answer_in_top3"] = any(contains_answer(s, good) for e in result.evidence[:3]
                                        for s in split_sentences(e.chunk.text))
        else:
            out["answer_in_top3"] = bool(gold & {e.chunk.citation for e in result.evidence[:3]})
        return out

    async def analyze_scenario(self, path: Path, time_scale: float) -> dict:
        gt = load_ground_truth(path)
        calls: list = []
        results, trace, answers = await run_scenario_detailed(str(path), self.config,
                                                              time_scale=time_scale, calls=calls)
        by_version = {(c["session_id"], c["version"]): c for c in calls}
        by_utt = {r["utterance_id"]: r for r in results}
        records, wrong = [], []
        for uid, turn in gt.get("turns", {}).items():
            if turn.get("kind") not in ("new_request", "refinement"):
                continue
            result = by_utt.get(uid)
            call = by_version.get((result["session_id"], result["answer_version"])) if result else None
            answer_text = result["answer"] if result else ""
            cited = set(result["citations"]) if result else set()
            n_gold = len(turn.get("gold_sub_intents", []))
            for question in turn.get("gold_sub_intents", []):
                row = self.qrels.get(question)
                if not row or not row.get("answers"):
                    continue
                good = self._usable(row)
                ok = contains_answer(answer_text, good) if good else bool(cited & set(row["relevant"]))
                rec = {"scenario": path.stem, "utterance": uid, "kind": turn["kind"],
                       "template": path.stem.rsplit("_", 1)[0], "question": question,
                       "gold": row["relevant"], "answers": row["answers"][:3],
                       "correct": ok, **self.ceilings(row, question, call, n_gold)}
                if not ok:
                    rec["cause"], rec["detail"] = self.classify_miss(row, question, call, answer_text, n_gold)
                records.append(rec)
            # wrong asserted claims
            if call is None:
                continue
            version = answers.get((result["session_id"], result["answer_version"]))
            if version is None:
                continue
            parent = answers.get((result["session_id"], result["parent_version"])) if result["parent_version"] else None
            prior = {(c.text, tuple(c.citations)) for c in parent.claims} if parent else set()
            labelled = [self.qrels[q] for q in turn.get("gold_sub_intents", []) if q in self.qrels]
            gold_secs = {s for r in labelled for s in r["relevant"]}
            good_all = [a for r in labelled for a in self._usable(r)]
            for claim in version.claims:
                if (claim.text, tuple(claim.citations)) in prior:
                    continue
                correct = bool(set(claim.citations) & gold_secs) or (bool(good_all) and contains_answer(claim.text, good_all))
                if correct:
                    continue
                source = None
                for sq, res in ((sq, next((r for r in call["results"] if r.query_id == sq.query_id), None))
                                for sq in call["sub_queries"]):
                    if res is None:
                        continue
                    cl, _ = self.synth._claims_from([sq], [res])
                    if cl and cl[0].text == claim.text:
                        source = sq
                        break
                matched = max((jaccard(set(content_tokens(q)), set(content_tokens(source.text)))
                               for q in turn.get("gold_sub_intents", [])), default=0.0) if source else 0.0
                if turn.get("expect_uncertainty"):
                    cause = "answered_unanswerable"
                elif source is None or matched < MATCH_THRESHOLD:
                    cause = "spurious_subquery"
                else:
                    cause = "wrong_passage"
                wrong.append({"scenario": path.stem, "utterance": uid, "cause": cause,
                              "sub_query": source.text if source else None,
                              "claim": claim.text[:200], "citation": claim.citations})
        return {"records": records, "wrong": wrong}


def summarize(records: list[dict], wrong: list[dict]) -> dict:
    n = len(records)
    misses = [r for r in records if not r["correct"]]
    causes = Counter(r["cause"] for r in misses)
    return {
        "n_questions": n, "correct": n - len(misses), "misses": len(misses),
        "miss_causes": dict(causes.most_common()),
        "ceilings": {k: sum(1 for r in records if r[k]) for k in ("isolated", "gold_in_evidence", "answer_in_top3")},
        "by_template": {t: {"n": sum(1 for r in records if r["template"] == t),
                            "correct": sum(1 for r in records if r["template"] == t and r["correct"])}
                        for t in sorted({r["template"] for r in records})},
        "wrong_claims": dict(Counter(w["cause"] for w in wrong).most_common()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--time-scale", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    analyzer = Analyzer(args.corpus_dir, args.overrides)
    records, wrong = [], []
    for path in discover_scenarios(args.scenarios):
        out = asyncio.run(analyzer.analyze_scenario(path, args.time_scale))
        records += out["records"]
        wrong += out["wrong"]
    summary = summarize(records, wrong)
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "records": records, "wrong": wrong},
                                             indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
