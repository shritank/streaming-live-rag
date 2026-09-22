"""BM25 sparse index (Okapi BM25). Pure Python, deterministic, no deps."""
from __future__ import annotations

import math
from collections import Counter

from .text import content_tokens


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_tokens = [content_tokens(d) for d in documents]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._n = len(documents)

        self._tf: list[Counter] = [Counter(t) for t in self._doc_tokens]
        df: Counter = Counter()
        bigram_df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
            bigram_df.update({f"{a} {b}" for a, b in zip(tokens, tokens[1:])})
        # BM25 idf with the +0.5 smoothing, floored at a small positive value so
        # very common terms never contribute negative score.
        self._idf = {
            term: max(1e-6, math.log((self._n - count + 0.5) / (count + 0.5) + 1.0))
            for term, count in df.items()
        }
        self._bigram_df = dict(bigram_df)

    def score(self, query: str) -> list[float]:
        q_tokens = content_tokens(query)
        scores = [0.0] * self._n
        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in range(self._n):
                freq = self._tf[i].get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._doc_len[i] / (self._avg_len or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def specific_terms(self, min_idf: float = 2.5) -> frozenset[str]:
        """Discriminative vocabulary: terms that appear in relatively few
        documents (high idf), i.e. topic-specific nouns rather than words
        common across most of the corpus. Used by the controller's weak-anchor
        heuristic to recognise a real, indexed topic term even before it is
        capitalised or paired with a number."""
        return frozenset(term for term, idf in self._idf.items() if idf >= min_idf)

    def low_df_bigrams(self, max_df: int = 5) -> frozenset[str]:
        """Content-token bigrams that co-occur in fewer than `max_df`
        documents — a strong topical anchor: a bigram like "military
        campaign" pins down a specific passage far more precisely than either
        word alone, so it's trusted as an anchor at a much lower document
        frequency than a single-token specific term needs."""
        return frozenset(bg for bg, count in self._bigram_df.items() if count < max_df)

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(range(self._n), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]
