# Landscape audit and benchmark specification — Streaming Live RAG

**Scope note, read first.** This document was produced with a general web-search tool, not
authenticated access to arXiv's, ACL Anthology's, or IEEE/ACM's search APIs, and not a citation
graph tool. It is a **targeted, evidence-checked scan of the highest-value questions**, not the
exhaustive 20-venue systematic review the brief asked for — that scale of review (hundreds of papers,
full-text extraction, citation verification) is not something a handful of web searches can honestly
deliver, and claiming otherwise would violate the brief's own rule against fabrication. Every claim
below is labelled **FACT**, **PUBLISHED RESULT**, **OUR RESULT**, **HYPOTHESIS**, or **UNVERIFIED
CLAIM**; anything I could not independently check is marked as such rather than presented as
established. Where the brief's requested table cell (e.g. "exact Whisper version" for a paper I did
not fetch in full) is unknown, the cell says `not verified` rather than a guess.

**Our own system's numbers**, cited throughout as **OUR RESULT**, are drawn entirely from
[`final_report.docx`](../final_report.docx) — the sealed `test2` evaluation completed just before this
document. That report is the source of truth for anything about our own system stated here.

---

## A. State-of-the-art comparison table

| System | Year | Text | Speech | Real ASR | Streaming | Multi-intent | Retrieval R@1 | Answer metric | Citation/grounding | Latency | Dataset | Code | Directly comparable to us? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Our system (sealed test2)** | 2026 | ✓ | — (ASR text only) | ✓ | ✓ | ✓ | **90.1%** [89.0,91.0] (n=3,282 qrels) | Answer recall 76.5% (clean) / 56.0% (ASR) | Claim precision 78.2–97.7%, citation recall 57.0–88.9% | p95 47 ms post-speech (streaming) | Custom, SQuAD-derived + HeySQuAD | Not published | — |
| HeySQuAD baseline reader (roberta-base, fine-tuned ON HeySQuAD) | 2023 | ✓ | — | ✓ | ✗ | ✗ | not applicable (no retrieval stage; oracle-context QA) | **F1 87.34%** (span-level token F1) | none | not reported | HeySQuAD (76k human-spoken Q) | Public (HF dataset) | **No** — different metric (F1 vs. our answer-recall), fine-tuned on the target distribution vs. our zero-shot use of the same dataset, no retrieval stage |
| Stream RAG (speech-in speech-out, speculative tool calls) | Oct 2025 | ✗ | ✓ | ✓ (via ASR inside an E2E model) | ✓ | not reported | not reported | **11.1% → 34.2% absolute** QA accuracy (their metric) | not reported | "20% latency reduction" (relative, not absolute) | AudioCRAG (CRAG converted to speech) | Not verified public | **No** — open-domain web-QA task (CRAG), E2E speech-to-speech generation, different accuracy definition |
| VoiceAgentRAG (Salesforce AI Research) | Mar 2026 | ✓ (text queries, voice-agent context) | partial | not verified | ✓ (dual-agent pre-fetch) | not reported | not applicable — measures **cache hit rate**, not retrieval recall | not their focus | not their focus | 110 ms → 0.35 ms on cache hit (316×) | 200 queries / 10 scenarios (small, internal) | **Public** — `github.com/SalesforceAIResearch/VoiceAgentRAG` | **No** — different metric family entirely (cache-hit latency, not answer/retrieval quality); small internal eval set |
| ALCE (citation-grounded generation) | 2023 | ✓ | ✗ | ✗ | ✗ | ✗ | not applicable | Correctness / claim recall | Citation precision & recall (entailment-checked) | not applicable | ASQA, QAMPARI, ELI5 | Public | **No** — long-form generative answers (not extractive), open-domain multi-document, different question style |
| Industry citation-faithfulness studies (aggregate, various) | 2024–2026 | ✓ | ✗ | ✗ | ✗ | ✗ | — | — | **50–90% of LLM citations do not fully support their claims** (reported range across studies) | — | various | mixed | **No** — aggregate figure across generative systems; cited here only as context for why our claim precision (extractive-by-construction) is not the same *kind* of number |

**Reading this table honestly**: the row that looks best in isolation (our own) is also the row using
the metric we defined for the system we built. No row above is on the same dataset, same qrels, and
same metric definition as another — which is exactly why every comparison here is marked "No" and not
a leaderboard position. This is not evasion; it is the actual state of the field for this specific
combination of properties (§B).

---

## B. Research gap map

Based on this scan (not an exhaustive one — see scope note), nobody appears to have published a
benchmark that simultaneously tests, on the **same paired items**:

1. **Typed text, real human speech, and its real ASR transcript** for the *same* underlying question
   (HeySQuAD comes closest — it has human-spoken audio and machine-transcribed text for the same SQuAD
   questions — but has no streaming/partial-transcript dimension and no multi-intent/compound
   questions).
2. **Streaming partial transcripts** paired with a **closed, citation-checkable corpus** (Stream RAG
   and VoiceAgentRAG both do streaming/speculative retrieval, but Stream RAG evaluates open-domain
   web-QA accuracy with no citation metric, and VoiceAgentRAG evaluates cache-hit latency, not answer
   or citation quality).
3. **Multi-intent spoken questions** with **per-intent gold decomposition** as a scored metric
   (Compound-QA benchmarks LLM behavior on compound *text* questions, not spoken, and does not score
   decomposition exactness — it scores final-answer correctness only).
4. **Citation/claim-precision metrics on a system that is extractive by construction**, compared
   against systems that are generative (ALCE, RAGTruth, RAGBench) — extractive claim precision and
   generative citation faithfulness are measuring different failure modes and are not on a shared
   scale.

**HYPOTHESIS** (not verified by this scan, stated as a hypothesis per the brief's own rule): if this
combination has been benchmarked somewhere, it is most likely inside an industrial/internal voice-
assistant evaluation that is not published, rather than in the academic venues searched here — several
of the GitHub repos found (§G) are exactly this kind of unpublished internal-tool pattern.

---

## C. Record opportunity — stated honestly

**No directly comparable published result was located for our specific combination**: a closed,
citation-checkable corpus + streaming speculative retrieval + rule-based multi-intent decomposition +
real (not synthetic) ASR transcripts + a learned refusal gate, scored on retrieval R@1, answer recall,
citation recall, claim precision, correct abstention, and post-speech p95 latency, all at once, with a
sealed test split.

Per the brief's explicit instruction: **this is not a "world record" claim.** It is a statement that
this scan did not find a matching benchmark to be a record *against*. Two things would need to be true
before "record" became a defensible word: (1) a systematic, database-backed review (not a web-search
scan) confirming no one has published this combination, and (2) the same evaluation protocol run
against at least one other real system on the same data — neither has been done here.

What **can** be said with the evidence in hand: our system's retrieval R@1 (90.1%, sealed, 3,282
qrels), extractive claim precision (78.2–97.7%), and streaming-vs-deferred accuracy delta (0.0%,
bit-identical, §5.5 of the final report) are strong, reproducible, sealed-test-verified numbers *for
the benchmark we ourselves defined*. Whether that benchmark is harder or easier than any specific
published one is not established by this scan.

---

## D. Proposed benchmark specification (if pursuing a defensible claim later)

This is a specification, not a claim that it has been built beyond what already exists in this repo's
`eval/` harness.

* **Corpus**: closed, disjoint article sets (already exists: `data/corpus`, `data/corpus_test`,
  `data/corpus_test2`), each with qrels mapping questions to gold sections.
* **Splits**: `train` (n/a — this system is not trained) / **development** (`data/corpus`,
  architecture and threshold decisions) / **diagnostic** (`data/corpus_test`, confirmatory checks
  during development) / **sealed test** (`data/corpus_test2`, frozen before first use, run exactly
  once per component — already implemented and executed this audit, §5.8 of the final report).
* **Question types**: typed text (SQuAD-derived), single-intent spoken (HeySQuAD clean + ASR), and
  multi-intent compound (synthetically generated from SQuAD by concatenation — labelled as synthetic,
  not real spoken compound questions, which do not exist in HeySQuAD).
* **Speech collection**: HeySQuAD's existing human-recorded audio + its official ASR transcriptions
  (`yijingwu/HeySQuAD_human`, CC BY 4.0) — this project does not record its own speech and should not
  claim to.
* **Streaming protocol**: transcript chunks fed at their real HeySQuAD/SQuAD-scenario timestamps,
  `time-scale=1` for latency claims (already implemented, `eval/run_quality.py --time-scale 1`).
* **Gold annotations**: qrels (section-level relevance) + SQuAD answer spans (string-level) +
  synthetic unanswerable questions (SQuAD 2.0-style, adversarially written against the actual corpus
  passages — already implemented in `eval/heysquad.py`).
* **Metrics**: exactly the scorecard in §12 of the brief, already implemented in
  `eval/quality.py`/`eval/stats.py`/`eval/retrieval_eval.py`.
* **Statistical protocol**: Wilson intervals for single rates, paired cluster-bootstrap (4,000
  resamples) for A/B comparisons — already implemented, used throughout `final_report.docx`.
* **Reproducibility requirements**: pinned model revisions + SHA-256 (already implemented,
  `streaming_rag/retrieval/neural.py`), documented dataset revisions (already implemented, `.meta.json`
  files), frozen git state before the sealed run (done this audit, commit `6fb6305` + audit changes).

**What this specification is missing to be a genuinely new, publishable benchmark** rather than an
internal eval harness: (1) real multi-intent *spoken* recordings, not synthetic text concatenation —
no such data was found to exist publicly; (2) independent human verification of the qrels/gold
answers by someone other than the system's own author; (3) at least one other system's numbers on the
exact same sealed split, which requires either running a competitor's open-source code (§G) or
releasing the benchmark publicly for others to run against.

---

## E. Experimental matrix (minimal, information-ordered)

| # | Experiment | Hardware | Expected information value | Status |
|---|---|---|---|---|
| 1 | Vocabulary/typo correction (SymSpell) → full end-to-end HeySQuAD-ASR re-score | CPU-only, ~30 min | High — §5.4 of the final report shows a preliminary +1pt retrieval signal never carried through end-to-end | Not done |
| 2 | Multi-dataset extractive reader (not SQuAD2-only) as a refusal-gate feature | CPU-only, needs a new ~130 MB model download | Medium — could reduce the in-domain-reader optimism flagged in §9/§10.4 of the final report | Not done |
| 3 | Real independent human check of 50 sealed-test qrels/gold answers | No hardware — human labor | High for credibility, zero for score | Not done |
| 4 | Run one open-source comparable system (VoiceAgentRAG or similar) on our sealed corpus | CPU, needs their dependencies installed | High — this is the only way to get a genuine same-data comparison point | Not done — see §G for cost/risk |
| 5 | int8 quantisation of the cross-encoder | CPU-only | Low — §5.7 of the final report already shows latency is not the bottleneck | Deprioritised, not done |
| 6 | GPU inference as default | MX250, isolated toolchain | Already done — §5.7 of final report, rejected with evidence | Done |

---

## F. Hardware-aware plan

Given **8-core CPU / ~8 GB RAM / MX250 2 GB VRAM**:

* **Realistically executable here**: everything in §E except #4 unless the comparator system is
  lightweight (most of the GitHub repos in §G assume a cloud LLM/vector-DB and would need substantial
  adaptation to run locally and offline, which is this project's actual constraint).
* **Needs external/cloud hardware**: fine-tuning any reader/reranker (this project has never
  fine-tuned anything — all models are used zero-shot/off-the-shelf); running a competitor system that
  assumes a hosted vector database (Qdrant Cloud, as VoiceAgentRAG's own benchmark used) or a paid LLM
  API (every generative competitor found in §A/§G); large-reader experiments (roberta-base-squad2 is
  feasible on this hardware, larger models likely are not given the 2 GB VRAM / ~8 GB RAM ceiling
  already tight on the old 2 GB MX250 laptop this audit ran on; superseded - the final system runs on a 48 GB RTX 6000 Ada
  and was re-verified on a 6 GB RTX 4050 laptop, `final_report.docx` sections 0 and 0.9).
* **2 GB VRAM is the binding constraint** for any GPU experiment — the current reader/cross-encoder
  models (33–66M params) already fit; nothing larger realistically will, confirmed by this session's
  own experience running the GPU benchmark on that old machine (`final_report.docx` §5.7; superseded by the machines above).

---

## G. GitHub implementation shortlist

| Repository | What it implements | Integration potential | Expected benefit | Cost | License | Reproducibility risk |
|---|---|---|---|---|---|---|
| `SalesforceAIResearch/VoiceAgentRAG` | Dual-agent (slow pre-fetch / fast cache-read) speculative retrieval for voice agents | Conceptually close to our speculative sub-query prefetch; architecture, not a drop-in library | Low direct benefit — our supersession mechanism already solves the same problem for our task shape (§5.5, bit-identical streaming/deferred) | Medium — needs adaptation from their cache-router design to our per-sub-query prefetch | not verified from this scan | Small internal eval (200 queries); not independently reproduced |
| `Chia-Hsuan-Lee/Spoken-SQuAD` | The original Spoken-SQuAD dataset (predecessor to HeySQuAD, synthetic TTS speech + ASR, not real human speech) | Could extend our test corpus, but **synthetic TTS**, which this project's own report already flags as a weaker proxy than HeySQuAD's real human speech | Low — HeySQuAD is already the stronger, real-speech option we use | Low | not verified from this scan | Synthetic-speech proxy, documented limitation in the paper itself |
| Lattice/confusion-network retrieval prior art (Saraclar et al., ACL Anthology N04-1017; word confusion networks, E09-1028) | ASR lattice-based retrieval, not a maintained repo | **Not applicable to this project** — HeySQuAD provides only final 1-best transcripts, not lattices/confusion networks or raw audio, so this technique has no data to operate on here | None, given current data | N/A | N/A (research technique, decades-old) | N/A |
| Various voice-RAG chat repos (`Warishayat/Voice-Based-Rag`, `sidhyaashu/multimodal-live-rag-voice`, etc.) | End-to-end demo apps: STT → cloud LLM RAG → TTS | Low — these are demo-quality integrations of hosted APIs (OpenAI/Groq/ElevenLabs), not benchmarked, not offline-capable, incompatible with this project's offline/CPU constraint | None for benchmark purposes | N/A | not verified from this scan | No benchmark numbers at all; README-only claims (explicitly what the brief warns against treating as evidence) |

**Explicitly**: none of the repositories found in this scan are drop-in, benchmarked, license-clear,
offline-capable systems that could be run against our sealed corpus without substantial adaptation
work first. Experiment #4 in §E (run one comparator on our data) is real but non-trivial effort, not a
quick addition.

---

## H. "Do not waste time" list

Based on evidence gathered this audit (the completed experiments in `final_report.docx`) and this scan:

* **Bigger reranker/embedder models** — §5.2 of the final report already shows MiniLM-L12 (larger) and
  BGE-small (swapped) both *lose* to the current config with significant paired CIs. No reason from
  this scan to expect a different larger model to reverse that on this corpus scale.
* **GPU as the default execution path** — settled with an end-to-end measurement (§5.7): no benefit,
  because speculative prefetch already removes the bottleneck GPU would speed up.
* **Lattice/confusion-network ASR retrieval** — genuine, well-established technique (§G), but this
  project has no access to ASR lattices, only final transcripts. Not actionable without a different
  data source.
* **Chasing "record" framing** — no evidence base exists in this scan to support that word; time is
  better spent on §E #1–3 than on positioning language.
* **Adopting a demo voice-RAG repo wholesale** — all found repos in this category are cloud-API demo
  apps, not benchmarked systems; they would not move any number in our scorecard and would violate the
  offline/CPU constraint.

---

## I. Final recommendation — roadmap ordered by expected information value

1. **Independent spot-check of 50 sealed-test qrels/gold answers by a second reviewer** (§E #3). Zero
   hardware cost, directly addresses the credibility gap that stops §C from being a stronger claim.
2. **Finish the vocabulary-correction experiment end-to-end** (§E #1) — it is the one lever this
   audit's own report (§10.1) flagged as promising but unfinished, and directly targets the largest
   remaining gap (the 22–33-point clean-vs-ASR answer-recall drop).
3. **Attempt one real same-data comparison** (§E #4) — without this, §C's "no comparable result
   found" stays a statement about search coverage, not a statement about the field. This is the
   highest-effort, highest-payoff item.
4. **Multi-dataset reader for the refusal gate** (§E #2) — addresses a named, specific limitation
   (§10.4 of the final report) rather than a general "try a bigger model" instinct.
5. Everything in §H, explicitly, is **not** recommended next.

---

## Fact/claim classification key, applied throughout

- **FACT** — directly observable in this repo's code or this scan's search results, no interpretation.
- **PUBLISHED RESULT** — a number reported in a paper/repo found by this scan, not independently
  reproduced here.
- **OUR RESULT** — from `final_report.docx`, this project's own sealed/verified evaluation.
- **HYPOTHESIS** — a reasoned guess, explicitly labelled, not evidenced by this scan.
- **UNVERIFIED CLAIM** — a claim this scan encountered (e.g. in a README) but could not independently
  check.

No number in this document was fabricated; where a requested field (Whisper version, exact hardware,
etc.) could not be confirmed for a third-party system, it is marked `not verified` rather than
estimated.
