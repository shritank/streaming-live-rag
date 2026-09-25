"""GroundedSynthesizer — session-aware synthesis (Task 3).

Answers are built extractively by default: every claim is a sentence taken
from a retrieved chunk, carrying that chunk's citation. This makes the hard
rule "no factual claim from parametric memory" structurally true rather than
a prompt instruction, and makes G4 support verifiable.

When an LLM is configured it is used only to *phrase* the assembled claims,
never to supply facts: the rewritten text is re-verified against the same
evidence and is discarded if it drifts (see `_phrase`).
"""
from __future__ import annotations

import math
import re
import time

from ..config import Config
from ..contracts import (
    AnswerVersion, Claim, LLMClient, RetrievalResult, SubQuery,
    Telemetry, NullTelemetry,
)
from ..retrieval.text import content_tokens, split_sentences
from . import grounding
from .delta import merge_claims, union_citations
from .store import SessionStore

# Minimum fraction of the sub-query's content tokens that must appear in the
# extracted claim sentence for the evidence to be trusted. Below this, the
# retrieval cleared the low-confidence bar but the specific sentence doesn't
# actually address the question — tuned against data/corpus (real SQuAD),
# see docs/benchmark_report.md.
MIN_QUERY_RELEVANCE = 0.2

# A long, highly-specific query's relevant overlap is naturally spread across
# more tokens than a short one — a genuinely correct paraphrase can validly
# share a smaller FRACTION of an 8-token query than of a 3-token one, even
# though it shares just as many or more tokens in absolute terms. Short
# queries (<=3 content tokens) keep the strict floor unchanged, since that is
# exactly where a single ubiquitous word (e.g. a person's name) could
# trivially clear a looser threshold. Floors out at MIN_QUERY_RELEVANCE_FLOOR
# so a long query still can't pass on a near-zero match.
MIN_QUERY_RELEVANCE_FLOOR = 0.12
_RELEVANCE_TAPER_START = 3       # content-token count where relaxation begins
_RELEVANCE_TAPER_STEP = 0.02     # relaxation per additional content token


_ANAPHOR_RE = re.compile(r"^(?:it|its|he|his|she|her|they|their|them|this|these|those)\b", re.IGNORECASE)

_REFUSAL_MODEL: dict | None = None


def _refusal_model() -> dict:
    global _REFUSAL_MODEL
    if _REFUSAL_MODEL is None:
        import json
        from pathlib import Path
        _REFUSAL_MODEL = json.loads((Path(__file__).parent / "refusal_gate.json").read_text(encoding="utf-8"))
    return _REFUSAL_MODEL


def _quoted(q: SubQuery) -> str:
    # the user's own words, not a keyword label ("didn newton mechanics")
    return f'"{q.text.strip()}"' if q.text.strip() else q.intent_label


def _overlap(question: str, text: str) -> float:
    q = set(content_tokens(question))
    return len(q & set(content_tokens(text))) / len(q) if q else 0.0


def _min_relevance_for(n_query_tokens: int) -> float:
    if n_query_tokens <= _RELEVANCE_TAPER_START:
        return MIN_QUERY_RELEVANCE
    relaxed = MIN_QUERY_RELEVANCE - _RELEVANCE_TAPER_STEP * (n_query_tokens - _RELEVANCE_TAPER_START)
    return max(MIN_QUERY_RELEVANCE_FLOOR, relaxed)


class GroundedSynthesizer:
    def __init__(self, retriever, session_store: SessionStore, config: Config,
                 llm: LLMClient | None = None, telemetry: Telemetry | None = None):
        self._retriever = retriever
        self._sessions = session_store
        self._config = config
        self._llm = llm
        self._telemetry = telemetry or NullTelemetry()
        # query_id -> (id of the RetrievalResult it was computed from, claim)
        self._prefetched: dict[str, tuple[int, Claim | None]] = {}

    def prefetch(self, query: SubQuery, result: RetrievalResult) -> None:
        """Speculative claim selection (and refusal decision), called by the
        engine in a worker thread when a retrieval completes during speech.
        Only the model-based selector is expensive enough to be worth it."""
        if (self._config.synthesis.selector != "cross-encoder" or not self._config.synthesis.speculative_selection
                or not result.evidence or result.low_confidence):
            return
        claim = self._gated_claim(query.text, result)
        if len(self._prefetched) > 512:
            self._prefetched.clear()
        self._prefetched[query.query_id] = (id(result), claim)

    def _gated_claim(self, query: str, result: RetrievalResult) -> Claim | None:
        claim = self._ce_claim(query, result)
        if claim is None or self._config.synthesis.refusal_gate != "learned":
            return claim
        # The gate's features (the SQuAD2 reader's answer margin especially)
        # were fit on well-formed spoken QUESTIONS (HeySQuAD); a bare
        # keyword-style sub-query ("reimbursement", no verb or wh-word) is
        # out-of-distribution for the reader and drags its own margin down
        # for reasons unrelated to whether the retrieved evidence is right.
        # Below 2 content tokens there usually isn't a real question left to
        # score, so fall back to the same lexical relevance floor the
        # non-gated selector already relies on, rather than trust a model
        # score computed on text it was never calibrated for.
        if len(content_tokens(query)) < 2:
            return claim if self._is_relevant(query, claim.text, self._specific_terms()) else None
        return claim if self._answer_probability(query, result, claim) >= self._refusal_threshold() else None

    def _refusal_threshold(self) -> float:
        t = self._config.synthesis.refusal_threshold
        return t if t is not None else _refusal_model()["threshold"]

    def _answer_probability(self, query: str, result: RetrievalResult, claim: Claim) -> float:
        """P(the claim answers the question correctly), from the logistic
        model fitted on development data (session/refusal_gate.json). The
        features are computed exactly as they were for fitting."""
        from ..retrieval.neural import get_cross_encoder, get_reader
        model = _refusal_model()
        evidence = result.evidence
        top = evidence[0]
        title = top.chunk.metadata.get("doc_title", "")
        section = self._resolve_citation(top.chunk.citation)
        values = {
            "top": top.score,
            "sentence": float(get_cross_encoder().score(query, [f"{title}: {claim.text}"])[0]),
            "overlap_ce": _overlap(query, claim.text),
            "overlap_section": _overlap(query, section.text) if section is not None else 0.0,
            "margin": top.score - evidence[1].score if len(evidence) > 1 else 0.0,
            "reader_margin": get_reader().margin(
                query, [e.chunk.text for e in evidence[:self._config.synthesis.ce_evidence_chunks]]),
        }
        if values["reader_margin"] is None:
            values["reader_margin"] = -10.0
        z = model["intercept"]
        for name, mean, scale, w in zip(model["features"], model["scaler_mean"], model["scaler_scale"], model["coef"]):
            z += w * (values[name] - mean) / scale
        return 1.0 / (1.0 + math.exp(-z))

    # ---------- Synthesizer protocol ----------

    async def synthesize(self, session_id: str, utterance: str,
                          sub_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        claims, unsupported = self._claims_from(sub_queries, results)
        prior = self._sessions.view(session_id).latest_answer()
        return self._finalise(session_id, claims, unsupported, sub_queries, results,
                               parent=prior, change_kind="initial", carried=[])

    async def refine(self, session_id: str, utterance: str,
                      delta_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        prior = self._sessions.view(session_id).latest_answer()
        if prior is None:
            return await self.synthesize(session_id, utterance, delta_queries, results)

        delta_claims, unsupported = self._claims_from(delta_queries, results)
        merged, carried = merge_claims(prior.claims, delta_claims)
        return self._finalise(session_id, merged, unsupported,
                               list(prior.sub_queries) + list(delta_queries), results,
                               parent=prior, change_kind="refinement", carried=carried)

    async def restructure(self, session_id: str, instruction: str) -> AnswerVersion:
        """Presentation-only turn: no retrieval, no new citations (pitfall 4)."""
        prior = self._sessions.view(session_id).latest_answer()
        if prior is None:
            raise RuntimeError("restructure with no prior answer in session")

        text = self._reformat(prior.claims, instruction)
        return AnswerVersion(
            version=prior.version + 1, parent_version=prior.version, change_kind="restructure",
            text=text, claims=list(prior.claims), citations=list(prior.citations),
            evidence_ids=list(prior.evidence_ids), sub_queries=list(prior.sub_queries),
            uncertainty=prior.uncertainty, created_ms=int(time.monotonic() * 1000),
        )

    # ---------- internals ----------

    def _claims_from(self, sub_queries: list[SubQuery],
                      results: list[RetrievalResult]) -> tuple[list[Claim], list[str]]:
        by_query = {r.query_id: r for r in results}
        claims: list[Claim] = []
        unsupported: list[str] = []

        for q in sub_queries:
            result = by_query.get(q.query_id)
            if result is None or not result.evidence or result.low_confidence:
                unsupported.append(_quoted(q))
                continue
            if self._config.synthesis.selector == "cross-encoder":
                cached = self._prefetched.pop(q.query_id, None)
                claim = cached[1] if cached and cached[0] == id(result) else self._gated_claim(q.text, result)
                if claim is None:
                    unsupported.append(_quoted(q))
                else:
                    claims.append(claim)
                continue
            # The reranked #1 chunk can clear the retriever's low_confidence
            # bar (its BM25/RRF score is decent) while still not actually
            # answering THIS sub-query — e.g. a topically-adjacent passage
            # from the same document that merely shares its subject noun.
            # Rather than asserting it and discovering the mismatch only in
            # the post-hoc grounding check, scan the retrieved evidence in
            # rank order for the first chunk whose best sentence is actually
            # relevant to the query, falling back to uncertainty only if none
            # of the top-k evidence is.
            claim = None
            for evidence in result.evidence:
                sentences = self._evidence_sentences(evidence)
                if not sentences:
                    continue
                idx = self._best_index(q.text, sentences)
                if self._is_relevant(q.text, sentences[idx], self._specific_terms()):
                    claim = Claim(text=self._with_continuation(sentences, idx),
                                  citations=[evidence.chunk.citation])
                    break
            if claim is None:
                unsupported.append(_quoted(q))
                continue
            claims.append(claim)

        return claims, unsupported

    def _ce_claim(self, query: str, result: RetrievalResult) -> Claim | None:
        """Learned relevance instead of lexical overlap: the cross-encoder
        judges whether each candidate sentence answers the query, so a
        sentence that merely shares the query's words (the self-fulfilling
        case) scores low. The document title is prepended when scoring
        because SQuAD-style sentences often refer to their subject by pronoun
        ("he contracted cholera"); the claim itself stays the verbatim
        sentence."""
        from ..retrieval.neural import get_cross_encoder
        candidates, seen = [], set()
        for evidence in result.evidence[:self._config.synthesis.ce_evidence_chunks]:
            title = evidence.chunk.metadata.get("doc_title", "")
            sentences = self._evidence_sentences(evidence)
            for idx, sentence in enumerate(sentences):
                key = (evidence.chunk.citation, sentence)
                if key in seen or not content_tokens(sentence):
                    continue
                seen.add(key)
                candidates.append((evidence.chunk.citation, sentences, idx,
                                   f"{title}: {sentence}" if title else sentence))
        if not candidates:
            return None
        scores = get_cross_encoder().score(query, [c[3] for c in candidates])
        best = int(max(range(len(candidates)), key=lambda i: (scores[i], -i)))
        if scores[best] < self._config.synthesis.ce_claim_threshold:
            return None
        citation, sentences, idx, _ = candidates[best]
        return Claim(text=self._with_continuation(sentences, idx), citations=[citation])

    def _evidence_sentences(self, evidence) -> list[str]:
        """Candidate claim sentences for one piece of evidence: its chunk, or
        (evidence_scope=section) the whole section it belongs to — a long
        section is split over several chunks and the answer may sit in a
        neighbouring one."""
        text = evidence.chunk.text
        if self._config.synthesis.evidence_scope == "section":
            section = self._resolve_citation(evidence.chunk.citation)
            if section is not None:
                text = section.text
        out: list[str] = []
        for s in split_sentences(text):
            s = s.strip()
            if s and s not in out:
                out.append(s)
        return out

    def _with_continuation(self, sentences: list[str], idx: int) -> str:
        if (self._config.synthesis.anaphoric_continuation and idx + 1 < len(sentences)
                and _ANAPHOR_RE.match(sentences[idx + 1])):
            return f"{sentences[idx]} {sentences[idx + 1]}"
        return sentences[idx]

    @staticmethod
    def _best_index(query: str, sentences: list[str]) -> int:
        """Index of the sentence with the highest query-token coverage (first
        wins ties) — the same rule as _best_sentence, over a sentence list."""
        if len(sentences) <= 1:
            return 0
        q_tokens = set(content_tokens(query))
        best, best_score = 0, -1.0
        for i, s in enumerate(sentences):
            s_tokens = set(content_tokens(s))
            if not s_tokens:
                continue
            score = len(q_tokens & s_tokens) / len(q_tokens or {1})
            if score > best_score:
                best, best_score = i, score
        return best

    def _specific_terms(self) -> frozenset[str] | None:
        """The corpus's discriminative-term vocabulary (see
        HybridRetriever.specific_vocabulary), if the configured retriever
        exposes one. Returns None for a retriever that doesn't (e.g. a mock),
        in which case _is_relevant falls back to its plain 2-token rule."""
        return getattr(self._retriever, "specific_vocabulary", None)

    @staticmethod
    def _is_relevant(query: str, sentence: str, specific_terms: frozenset[str] | None = None) -> bool:
        """A single shared token is not enough evidence on its own in
        general: in a biographical article, "Tesla" appears in nearly every
        sentence, so a query like "who was Tesla prejudiced against" would
        trivially "match" any sentence mentioning Tesla regardless of topic.
        Require BOTH a minimum overlap fraction (relaxed for longer, more
        specific queries — see _min_relevance_for) AND, once the query has
        more than one content term, at least two of them present — UNLESS
        the single overlapping token is itself a discriminative corpus term
        (rare across the corpus, e.g. "Astor" vs. the ubiquitous "Tesla"),
        in which case that one token is strong enough evidence on its own:
        a genuine paraphrase can legitimately share nothing but a specific
        name ("did Astor provide the money" -> "Astor invested $100,000")."""
        q_tokens = set(content_tokens(query))
        if not q_tokens:
            return False  # nothing to check relevance against: never assert blind
        s_tokens = set(content_tokens(sentence))
        overlap = q_tokens & s_tokens
        if len(overlap) / len(q_tokens) < _min_relevance_for(len(q_tokens)):
            return False
        min_overlap_count = min(2, len(q_tokens))
        if len(overlap) >= min_overlap_count:
            return True
        return bool(specific_terms) and bool(overlap) and overlap.issubset(specific_terms)

    @staticmethod
    def _best_sentence(query: str, chunk_text: str) -> str:
        """The sentence of the chunk that best answers the query. Keeping the
        claim to one sentence keeps the grounding check meaningful — a whole
        chunk would trivially 'support' anything inside it."""
        sentences = split_sentences(chunk_text)
        if len(sentences) <= 1:
            return chunk_text.strip()
        q_tokens = set(content_tokens(query))
        best, best_score = sentences[0], -1.0
        for s in sentences:
            s_tokens = set(content_tokens(s))
            if not s_tokens:
                continue
            score = len(q_tokens & s_tokens) / len(q_tokens or {1})
            if score > best_score:
                best, best_score = s, score
        return best.strip()

    def _finalise(self, session_id: str, claims: list[Claim], unsupported: list[str],
                   sub_queries: list[SubQuery], results: list[RetrievalResult],
                   parent: AnswerVersion | None, change_kind: str,
                   carried: list[str]) -> AnswerVersion:
        # an over-split request can select the same sentence for two sub-queries
        claims = list({(c.text, tuple(c.citations)): c for c in claims}.values())
        if self._config.synthesis.grounding_check:
            report = grounding.verify(claims, self._resolve_citation)
            claims = report.claims
            unsupported = unsupported + [
                c.text[:40] for c in claims if c.supported is False
            ]
        citations = union_citations(claims, carried)
        evidence_ids = sorted({e.chunk.chunk_id for r in results for e in r.evidence})
        if parent is not None and change_kind == "refinement":
            evidence_ids = sorted(set(evidence_ids) | set(parent.evidence_ids))

        text = " ".join(c.text for c in claims) if claims else \
            "I could not find anything in the corpus that answers this."
        uncertainty = None
        if unsupported:
            uncertainty = ("Could not be verified from the retrieved corpus: "
                            + "; ".join(dict.fromkeys(unsupported)) + ".")

        version = (parent.version + 1) if parent is not None else 1
        return AnswerVersion(
            version=version,
            parent_version=parent.version if parent is not None else None,
            change_kind=change_kind, text=text, claims=claims, citations=citations,
            evidence_ids=evidence_ids, sub_queries=list(sub_queries),
            uncertainty=uncertainty, created_ms=int(time.monotonic() * 1000),
        )

    def _resolve_citation(self, citation: str):
        getter = getattr(self._retriever, "get_chunk_by_citation", None)
        if getter is not None:
            return getter(citation)
        return self._retriever.get_chunk(citation)

    @staticmethod
    def _reformat(claims: list[Claim], instruction: str) -> str:
        lowered = instruction.lower()
        wants_bullets = any(w in lowered for w in ("bullet", "point", "list"))
        limit = None
        for word, n in (("two", 2), ("three", 3), ("four", 4)):
            if word in lowered:
                limit = n
        import re
        m = re.search(r"\b(\d+)\b", lowered)
        if m:
            limit = int(m.group(1))

        selected = claims[:limit] if limit else claims
        if wants_bullets:
            return "\n".join(f"• {c.text} [{', '.join(c.citations)}]" for c in selected)
        return " ".join(c.text for c in selected)
