# Streaming Live RAG

Real-time incremental retrieval, multi-intent decomposition, and state-preserving answer
refinement — an event-driven RAG engine that starts searching *while the user is still
speaking*, splits compound requests into parallel sub-queries, and patches an existing answer
in place when a late constraint arrives, instead of restarting from scratch.

Built against the Theme 4 guide (`../Theme 4 Guide_RAG.pdf`) and the four-task project context
in `../Task 1`–`../Task 4`.

## Quickstart

```bash
pip install -r requirements.lock

# build the fixture and real corpora (idempotent, seeded/deterministic)
python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus
python -m data.build_squad_corpus --n-articles 14 --out data/corpus

# watch the reference agent handle the guide's 3 example scenarios
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --time-scale 8
python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json --time-scale 8

# run the full gate report (offline, zero API cost — LLM_PROVIDER=fake by default)
python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1

# same system, on a real corpus it has never been tuned against
python -m eval.run_all --scenarios eval/scenarios_real_corpus --corpus-dir data/corpus --time-scale 8

# baseline vs streaming latency comparison, and the two required ablations
python -m eval.compare --modes streaming,baseline --simulated-latency-ms 300
python -m eval.ablate

pytest tests/ -q
```

Or, one command on a clean machine:

```bash
cp .env.example .env
docker compose up --build
```

## Repository layout

```
streaming_rag/
  contracts.py         shared dataclasses/protocols — the only coupling surface (§6.1 of project context)
  config.py             every tunable + ablation switch
  llm.py                 async LLMClient: fake (offline, default) | openai | gemini
  mocks.py               deterministic mock Controller/Retriever/Synthesizer, with fault injection
  engine.py              wires Controller -> Retriever -> Synthesizer on one asyncio event loop
  build.py               component factory (real vs mock), used by the CLI and eval harness alike
  cli.py                 `replay` / `live`
  retrieval/             ingest (format-driven), BM25, LSA dense encoder, RRF fusion, rerank/dedup
  controller/            intent-stability scoring, multi-intent decomposition, WAIT/RETRIEVE/SUPPRESS policy
  session/               ephemeral session store, extractive synthesis, grounding verifier, refinement delta engine
  telemetry/              JSONL sink, schema, cost model, trace assembler, text dashboard
fixtures/dev_corpus/      seeded synthetic corpus (fixtures/build_dev_corpus.py)
data/corpus/              real corpus from SQuAD v1.1 (data/build_squad_corpus.py) — CC BY-SA 4.0
eval/
  scenarios/               hand-written + generated scenarios (dev_corpus)
  scenarios_real_corpus/   generated scenarios (data/corpus)
  gates/                   G2-G6 scorers
  run_all.py, compare.py, ablate.py, edge_cases.py, scenario_gen.py
tests/                     contracts, engine, retrieval, controller, session, eval — 77 tests
docs/                      architecture brief, telemetry schema, benchmark report, demo script
```

## Data

- **`fixtures/dev_corpus/`** — synthetic, seeded, format-driven fixture (policy/venue/travel/
  catering documents), used for fast local iteration and as the primary gate-report corpus.
- **`data/corpus/`** — real, publicly-licensed corpus built from **SQuAD v1.1** (Wikipedia
  prose, CC BY-SA 4.0). Used to prove the ingestion pipeline and every downstream gate behave
  the same on content the system has never seen. Not the hidden evaluation corpus.

Both ship with a `qrels.jsonl` (query → gold `Doc_ID §Section`) so retrieval quality and gate
scores are measured against real ground truth, not eyeballed.

## Results at a glance

See `docs/benchmark_report.md` for full numbers, including three diagnostic-and-tune passes on
the real SQuAD corpus (§0/§0b/§0c: sentence-segmentation fix, within-utterance retrieval
supersession fix, a corpus-vocabulary + bigram-DF weak-anchor heuristic, fusion-weight retune, a
dynamic query-relevance gate, a real correctness bug fix — short legitimate utterances like "When
are the ashes now?" were silently dropped with zero retrieval — and a general definitional-query
reranking boost). All 6 gates (G1–G6) pass on both corpora under the official `--reps 3`
procedure, confirmed byte-identical to `--reps 1`; streaming beats the baseline pipeline on
latency; 77/77 tests pass; **zero fabricated citations across every run.** On the real corpus:
G2 early retrieval 86.7%→**100.0%**, G4 grounding 88.9%→**97.5%** (both originally-requested
>95% targets now met), hybrid retrieval r@1 62.0%→63.2% and r@5 84.0%→86.8% — every post-tuning
number measured, not projected. `eval/diagnose.py` root-causes any remaining failure into exactly
one bucket (retrieval miss / synthesizer over-rejection / late-anchor timing) rather than
treating gate scores as a black box.

## What each task owns

| Task | Package |
|---|---|
| 1 — Corpus ingestion & hybrid retrieval | `streaming_rag/retrieval/`, `fixtures/`, `data/` |
| 2 — Stream controller & decomposition | `streaming_rag/controller/` |
| 3 — Session state, synthesis, grounding | `streaming_rag/session/` |
| 4 — Engine, telemetry, eval, packaging | `streaming_rag/engine.py`, `llm.py`, `telemetry/`, `eval/`, Docker, `docs/` |
