"""Answer-level quality metrics, scored against gold data the engine never sees.

G4 (eval/gates/g4_grounding.py) aggregates the synthesizer's OWN grounding
verdicts from telemetry. For extractive claims that verdict is close to
tautological (a sentence copied from a chunk always "supports" itself), so G4
cannot tell a right answer from a confidently wrong one. These metrics can,
because they compare the output with the corpus's gold labels:

  answer_recall    per gold sub-question: does the answer text contain one
                   of its gold answer strings (SQuAD normalisation)?
  citation_recall  per gold sub-question: is its gold `Doc_ID §Section` cited?
  claim_precision  per NEW asserted claim: is it backed by a gold section, or
                   does it contain a gold answer? The complement is the
                   "confidently wrong" rate — an asserted, cited, self-
                   supported claim that answers nothing that was asked.
  abstention       per unanswerable turn: no claim asserted, uncertainty set
  split_exact      per compound turn: final sub-query count == gold count
  oversplit        per single-question turn: more than one final sub-query

A claim that cites a non-gold section can still be right (SQuAD labels one
paragraph per question, but facts recur); counting a gold-answer string match
as correct keeps that case from being scored as wrong, so claim_precision is
a lower bound only where an answer is phrased differently from the gold span.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from streaming_rag.telemetry.trace import assemble_turns

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"–", "—", "‘", "’", "“", "”"}


def normalize(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_answer(text: str, answers: list[str]) -> bool:
    haystack = f" {normalize(text)} "
    return any(f" {normalize(a)} " in haystack for a in answers if normalize(a))


def informative_answers(answers: list[str], title: str) -> list[str]:
    """Gold answers that occur in the article's own title ("Tesla" in an
    article titled "Nikola Tesla") appear in almost every sentence of that
    article, so finding them in an answer is no evidence the question was
    answered. Such answers are excluded from string matching; the question is
    then judged by whether its gold section was cited instead."""
    t = f" {normalize(title)} "
    return [a for a in answers if normalize(a) and f" {normalize(a)} " not in t]


def load_doc_titles(corpus_dir: str | Path) -> dict[str, str]:
    titles = {}
    for path in Path(corpus_dir).glob("*.md"):
        head = path.read_text(encoding="utf-8")[:500]
        doc = re.search(r"^doc_id:\s*(\S+)", head, re.M)
        title = re.search(r"^title:\s*(.+)$", head, re.M)
        if doc and title:
            titles[doc.group(1)] = title.group(1).strip()
    return titles


def load_qrels_index(corpus_dir: str | Path) -> dict[str, dict]:
    path = Path(corpus_dir) / "qrels.jsonl"
    index: dict[str, dict] = {}
    if not path.exists():
        return index
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            index.setdefault(row["query"], row)
    return index


@dataclass
class QualityResult:
    # (scenario, item_id) -> 0/1 outcome, kept per item for paired comparison
    answer: dict = field(default_factory=dict)
    citation: dict = field(default_factory=dict)
    claim_correct: dict = field(default_factory=dict)
    abstention: dict = field(default_factory=dict)
    split_exact: dict = field(default_factory=dict)
    oversplit: dict = field(default_factory=dict)
    wrong_claims: list = field(default_factory=list)
    # words per newly asserted claim: longer claims contain more gold strings
    # by chance, so a recall gain that only comes with longer claims is suspect
    claim_words: list = field(default_factory=list)

    def merge(self, other: "QualityResult") -> None:
        for name in ("answer", "citation", "claim_correct", "abstention", "split_exact", "oversplit"):
            getattr(self, name).update(getattr(other, name))
        self.wrong_claims.extend(other.wrong_claims)
        self.claim_words.extend(other.claim_words)

    @staticmethod
    def rate(items: dict) -> tuple[int, int]:
        return int(sum(items.values())), len(items)


def final_sub_query_counts(trace: list[dict]) -> dict[str, int]:
    """utterance_id -> sub-queries emitted and never cancelled/superseded."""
    counts = {}
    for turn in assemble_turns(trace):
        emitted = {qid for e in turn.of("sub_queries_emitted") for qid in e.get("query_ids", [])}
        cancelled = {qid for e in turn.of("retrieval_cancelled") for qid in e.get("query_ids", [])}
        counts[turn.utterance_id] = len(emitted - cancelled)
    return counts


def score_quality(scenario: str, ground_truth: dict, turn_results: list[dict],
                  answers: dict, subq_counts: dict[str, int], qrels: dict[str, dict],
                  titles: dict[str, str] | None = None) -> QualityResult:
    """`answers` maps (session_id, version) -> an AnswerVersion-like object
    (.claims[.text/.citations], .change_kind, .uncertainty), captured by the
    eval runner; `turn_results` are the engine's output records;
    `subq_counts` comes from final_sub_query_counts(trace); `titles` maps
    Doc_ID -> article title (see informative_answers)."""
    titles = titles or {}

    def usable(row: dict) -> list[str]:
        doc = row["relevant"][0].split(" §")[0] if row["relevant"] else ""
        return informative_answers(row["answers"], titles.get(doc, ""))

    out = QualityResult()
    turns_gt = ground_truth.get("turns", {})
    result_by_utt = {r["utterance_id"]: r for r in turn_results}

    for uid, gt in turns_gt.items():
        result = result_by_utt.get(uid)
        kind = gt.get("kind")
        gold_questions = gt.get("gold_sub_intents", [])
        labelled = [qrels[q] for q in gold_questions if q in qrels and qrels[q].get("answers")]
        turn_gold_sections = {s for row in labelled for s in row["relevant"]}

        if kind in ("new_request", "refinement") and labelled:
            answer_text = result["answer"] if result else ""
            citations = set(result["citations"]) if result else set()
            for i, row in enumerate(labelled):
                key = (scenario, f"{uid}:{i}")
                cited = bool(citations & set(row["relevant"]))
                good = usable(row)
                out.answer[key] = int(contains_answer(answer_text, good) if good else cited)
                out.citation[key] = int(cited)

        if kind == "new_request" and uid in subq_counts and gold_questions:
            n_final = subq_counts[uid]
            if len(gold_questions) >= 2:
                out.split_exact[(scenario, uid)] = int(n_final == len(gold_questions))
            else:
                out.oversplit[(scenario, uid)] = int(n_final > 1)

        if result is None or result.get("answer_version") is None:
            if gt.get("expect_uncertainty"):
                out.abstention[(scenario, uid)] = 0
            continue
        session_id = result["session_id"]
        version = answers.get((session_id, result["answer_version"]))
        if version is None or version.change_kind == "restructure":
            continue
        parent = answers.get((session_id, result["parent_version"])) if result["parent_version"] else None
        prior_claims = {(c.text, tuple(c.citations)) for c in parent.claims} if parent else set()
        new_claims = [c for c in version.claims if (c.text, tuple(c.citations)) not in prior_claims]

        if gt.get("expect_uncertainty"):
            out.abstention[(scenario, uid)] = int(not new_claims and bool(version.uncertainty))
        all_answers = [a for row in labelled for a in usable(row)]
        for j, claim in enumerate(new_claims):
            out.claim_words.append(len(claim.text.split()))
            correct = bool(set(claim.citations) & turn_gold_sections) or \
                (bool(all_answers) and contains_answer(claim.text, all_answers))
            out.claim_correct[(scenario, f"{uid}:c{j}")] = int(correct)
            if not correct:
                out.wrong_claims.append({"scenario": scenario, "utterance": uid,
                                         "questions": gold_questions, "claim": claim.text[:160],
                                         "citations": claim.citations})
    return out


def summarize(q: QualityResult) -> dict:
    from .stats import wilson
    rows = {}
    for name in ("answer", "citation", "claim_correct", "abstention", "split_exact", "oversplit"):
        k, n = QualityResult.rate(getattr(q, name))
        lo, hi = wilson(k, n)
        rows[name] = {"k": k, "n": n, "rate": (k / n) if n else None, "ci": [lo, hi]}
    words = q.claim_words
    rows["claim_words"] = {"mean": sum(words) / len(words) if words else None, "n": len(words)}
    return rows
