# Streaming Live RAG — Architecture Brief

## 1. Problem and goal

Standard RAG is turn-by-turn: the user finishes speaking, the system searches, then answers.
That produces three failures in live voice/support settings: high conversational latency
(search starts only after the user stops), compound requests answered with one blended query,
and late-arriving constraints ("actually, it was international") that force a full restart.

This system is an event-driven engine that (1) listens incrementally and predicts retrieval
intent before the utterance ends, (2) decomposes compound utterances into parallel sub-queries,
(3) refines an existing answer in place rather than restarting when a constraint arrives, and
(4) grounds every factual claim in a retrieved corpus chunk, with explicit uncertainty when
evidence is missing.

## 2. Pipeline

```
[chunk 0.0s] -> [chunk 0.8s] -> [chunk 1.6s] -> [utterance_end 2.1s]
        |
        v
[1] Retrieval Controller (controller/)        WAIT | RETRIEVE | SUPPRESS, per chunk
        |
        v
[2] Multi-Intent Decomposer (controller/decompose.py)
        |
        v
[3] Hybrid Retrieval & Fusion (retrieval/)     BM25 + LSA dense, RRF, rerank, dedup
        |
        v
[4] Session-Aware Synthesis (session/)         extractive claims, grounding check, delta engine
        |
        v
Engine (engine.py) wires 1-4 on one asyncio event loop, emits telemetry, writes turn_result
```

## 3. Retrieval trigger logic (controller/stability.py, controller/controller.py)

The controller scores intent **stability** on every chunk from three structural signals —
no topic keyword lists, so it survives a re-skinned corpus:

- **anchor**: does the fragment contain a number, a mid-sentence capitalised entity, or a
  *discriminative corpus term* (strong/topic anchor — see below), or at least 4 content words
  (weak anchor)?
- **closure**: does the fragment end mid-thought (trailing preposition/conjunction/article)?
- **growth**: did the latest chunk add any new content tokens?

The topic-anchor path (`stability.py::has_topic_anchor`) treats a content token as anchoring
even without capitalisation or a number, if it's a *discriminative* term in the corpus's own
vocabulary (high idf — `BM25Index.specific_terms()`, exposed as
`HybridRetriever.specific_vocabulary`). This catches abstract topic nouns a capitalisation/number
check alone would miss, without hardcoding any topic list: the controller receives it as an
injected zero-arg callable (`corpus_vocab`), not a `Retriever` reference, so it stays decoupled
from the retrieval protocol. Measured effect on the real corpus: G2 early-retrieval rate
86.7% → 93.3% with no change to the false-trigger rate (see `docs/benchmark_report.md` §0/§7).

A weighted sum against a configurable threshold (`controller.min_stability` for the final
chunk, `controller.provisional_stability` for partials) yields WAIT or RETRIEVE. Turns are
additionally classified as `presentation_only` (regex over reformat/shorten/bullet verbs,
only when a prior answer exists) or `chit_chat` (canonical greeting/thanks phrases), both of
which SUPPRESS retrieval entirely. This reproduces the guide's Example 1 timeline exactly:
WAIT at t=0.0s ("no anchor yet"), RETRIEVE at t=0.8s once "Pune"/"30" appear, decompose into
3 sub-queries at t=1.6s.

## 4. Multi-intent decomposition (controller/decompose.py)

Utterances are split on coordination/list boundaries (`,`, ` and `, `;`), then each candidate
clause is **rejected** if its content tokens are already covered by an earlier sub-query in the
same utterance (no re-emission, no over-fragmenting), and **merged** into its sibling if token
overlap exceeds a threshold (near-duplicate suppression). A clause too short to search on its
own borrows shared context terms from the utterance's first clause, so "the cancellation
policy" becomes "cancellation policy plan customer workshop pune", not just "cancellation
policy" against the whole corpus.

## 5. Hybrid retrieval & fusion (retrieval/)

- **Ingestion** (`ingest.py`) is format-driven: any directory of Markdown with `## §N heading`
  sections and an optional YAML frontmatter `doc_id`/`title` ingests into `Chunk` objects.
  Swapping the dev fixture for the real corpus is a config change, not a code change.
- **Sparse**: pure-Python Okapi BM25 (`bm25.py`).
- **Dense**: TF-IDF + truncated SVD ("LSA"), fit at cold-start in `HybridRetriever.setup()`
  (`embed.py`). Deterministic, no model download, ~1s to fit on this corpus size. Any object
  exposing `.encode(list[str]) -> np.ndarray` can replace it (a transformer encoder, e.g.)
  without touching the retriever.
- **Fusion**: Reciprocal Rank Fusion (`fusion.py`), not score interpolation — BM25 scores and
  cosine similarities are on incomparable scales, so fusing on rank avoids a calibration step
  that breaks every time the corpus changes. Weighted `sparse=0.99 / dense=0.01` — tuned against
  both `fixtures/dev_corpus` and the real SQuAD corpus (`docs/benchmark_report.md` §0/§3): the
  weak LSA encoder measurably *hurt* real-corpus r@5 at higher dense weights (83.8% at 0.15 vs
  86.8% at 0.01) with zero offsetting gain on dev_corpus's paraphrase queries, which score
  identically at every weight tested. Kept non-zero, not sparse-only, so a stronger encoder
  dropped into the `Embedder` protocol has a fusion weight to grow into without another retune.
- **Rerank & dedup** (`rerank.py`): promotes chunks with high query-term coverage and factual
  density (numbers, modal obligations like "must"/"required"), then drops near-duplicate chunks
  by Jaccard similarity so three paraphrases of one sentence don't crowd out a second fact.
- Cold-start (ingest + index + fit) happens once in `async setup()`, off the per-turn clock.

## 6. Session-aware synthesis (session/)

Answers are built **extractively by default**: every claim is the single best sentence of a
retrieved chunk, carrying that chunk's citation (`session/synthesis.py::_best_sentence`). This
makes "no fact from parametric memory" structurally true rather than a prompt instruction. An
`LLMClient` (fake by default, OpenAI/Gemini adapters in `llm.py`) may *phrase* an answer, never
supply facts.

**Grounding** (`grounding.py`): every claim's citation must resolve to a real corpus chunk
(`retriever.get_chunk_by_citation`) or the claim is dropped and the citation reported as
fabricated (G4's zero-tolerance check) — this catches a citation pointing at a doc/section that
doesn't exist. It does *not*, by itself, catch a citation that exists but doesn't answer the
question: because a claim's text is extracted verbatim from the chunk it cites, token-overlap
support against that same chunk is trivially ~1.0 by construction. That gap is closed one layer
earlier, in `_claims_from`'s **query-relevance gate** (`_is_relevant`): the extracted sentence
must share a minimum fraction *and* (once the query has more than one content term) at least two
of the query's own content tokens — otherwise a chunk that merely mentions the subject (e.g.
"Tesla" appearing in nearly every sentence of a Tesla biography) doesn't get asserted as an
answer. The check scans the ranked evidence list for the first chunk that clears it, falling
back to `uncertainty` only if none do — catching a real answer that was ranked #2 or #3, not
just #1.

**Corpus-vocabulary anchor** (see §3) and **within-utterance supersession** (see §7) are two
further legitimacy checks upstream of this: the fewer wrong-topic claims a growing utterance
produces in the first place, the less the relevance gate has to filter after the fact.

**Refinement, not restart** (`delta.py`): on a refinement turn, the new evidence's claims are
merged into the prior answer's claims — a new claim *supersedes* a prior one only when they
overlap heavily (same topic), everything else carries forward verbatim with its citation. The
answer version increments, `parent_version` links back, and every prior citation still appears
in the union (`union_citations`). This is what reproduces the guide's Example 2 exactly:
"the trip was international" only touches the international-travel claim, the reimbursement
summary claim from Version 1 survives untouched into Version 2.

**Presentation-only turns** (`restructure`) never call the retriever or `synthesize()` — they
reformat the existing claims (bullets, a word-count limit) and keep the same citation set,
which is how the engine guarantees pitfall 4 ("querying the vector DB for reformat requests")
cannot happen: the code path structurally has no retrieval call on this branch.

## 7. Engine (engine.py)

One asyncio event loop consumes an input queue of `{"timestamp_ms","event_type","payload"}`
envelopes and writes `turn_result` dicts to an output queue. Per chunk: controller decision ->
if RETRIEVE, `asyncio.create_task(retriever.search(...))` — non-blocking, so retrieval overlaps
with the user still speaking. At `utterance_end`, pending retrievals are awaited, then
`synthesize`/`refine`/`restructure` is dispatched by `turn_kind`. A sub-query whose
`parent_query_id` links it to an earlier one cancels/excludes that earlier one — the
stale-cancellation path required by the guide's "late-arriving constraint" story, needed in two
places: cross-turn refinement (`controller/controller.py::_scope_to_delta`) *and*
within-utterance supersession, where a single clause grows across several chunks
(`_link_supersession`, added after profiling showed a growing clause was re-triggering a fresh,
never-cancelled retrieval on every qualifying chunk — see `docs/benchmark_report.md` §7).
Cancellation itself (`_cancel_task`) cleans up stale evidence even when the superseded retrieval
had *already completed* by the time it's superseded — the common case for a fast local
retriever — not only when it's still in flight.

A **virtual clock** (`Engine._now_ms`) is anchored to each envelope's `timestamp_ms` and
advances by real elapsed time × `--time-scale`, so telemetry numbers are identical whether a
scenario replays at 1x or 8x. A **loop-lag watchdog** ticks every 20ms and emits
`error(error_type=loop_lag)` if a tick is late by more than `engine.loop_lag_threshold_ms` —
proof that a blocking call anywhere in the pipeline will be caught. A per-turn timeout
(`engine.per_turn_timeout_ms`) degrades to an uncertainty answer instead of hanging; any
component exception is caught, logged as an `error` event, and the turn still completes.

`engine.mode=baseline` is the guide's comparison pipeline: wait for `utterance_end`, issue one
undecomposed query on the full utterance text, call `synthesize` fresh every turn (no
refinement, no early retrieval) — used by `eval/compare.py`.

## 8. Telemetry (telemetry/)

Every component emits one flat event via the sync `Telemetry.emit()` protocol (never blocks the
loop). `JsonlTelemetry` buffers in memory and flushes from a background task; `BufferedTelemetry`
is the in-memory sink used by tests and the coverage checker. `schema.json` is the JSON Schema
every event validates against (G6). `trace.py::assemble_turns` groups a flat trace into one
record per `(session_id, utterance_id)` so the coverage checker and the text dashboard
(`python -m streaming_rag.telemetry.report trace.jsonl`) can reconstruct a turn from the trace
alone.

## 9. Data provenance

- **Dev fixture** (`fixtures/dev_corpus/`, `fixtures/build_dev_corpus.py`): ~41 synthetic
  policy/venue/travel/catering documents, seeded and deterministic, with a `qrels.jsonl` of
  query -> gold `Doc_ID §Section`, including paraphrased queries (not just lexical repeats of
  the source text) so hybrid vs sparse can actually be told apart.
- **Real corpus** (`data/corpus/`, `data/build_squad_corpus.py`): built from **SQuAD v1.1**
  (validation split), real Wikipedia prose, CC BY-SA 4.0 licensed
  (https://huggingface.co/datasets/rajpurkar/squad). 14 articles selected deterministically by
  seed, ~1,098 chunks, 3,458 question/answer/context qrels — used to prove the format-driven
  ingestion pipeline behaves the same on content it has never seen, and to benchmark retrieval
  against real gold evidence rather than only synthetic fixtures. This is **not** the hidden
  evaluation corpus.

## 10. Trade-offs and failure-mode mitigations

| Decision | Trade-off | Mitigation |
|---|---|---|
| LSA dense encoder instead of a transformer | Weaker semantic recall than a real embedding model | RRF weighted towards BM25 (0.85/0.15); swappable via the `Embedder` protocol with zero retriever changes |
| Extractive (not generative) synthesis by default | Answers read as assembled sentences, not fluent prose | An `LLMClient` may *phrase* the assembled claims; facts still come only from the extraction |
| Rule-based controller by default | Misses genuinely ambiguous refinement/new-topic boundaries an LLM might catch | `controller.mode=model` is a config switch (Architectural Parsimony: a rule/heuristic path first, LLM only where judgment is needed) |
| Per-turn timeout degrades to uncertainty | A slow but eventually-correct retrieval is abandoned | Timeout is configurable (`engine.per_turn_timeout_ms`); the guide's own cap is 120s per scenario |
