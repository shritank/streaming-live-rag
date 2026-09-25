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
python -m streaming_rag.retrieval.neural --fetch   # one-time download of the pinned ONNX models

# build the fixture and real corpora (idempotent, seeded/deterministic)
python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus
python -m data.build_squad_corpus --n-articles 14 --seed 1 --out data/corpus
python -m data.build_squad_corpus --n-articles 14 --seed 0 --out data/corpus_test

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
  retrieval/             ingest (format-driven), BM25, e5-small-v2 dense (ONNX), RRF fusion,
                         cross-encoder rerank (ONNX); LSA + heuristic rerank kept as options
  controller/            intent-stability scoring, multi-intent decomposition, WAIT/RETRIEVE/SUPPRESS policy
  session/               ephemeral session store, extractive synthesis, grounding verifier, refinement delta engine
  telemetry/              JSONL sink, schema, cost model, trace assembler, text dashboard
fixtures/dev_corpus/      seeded synthetic corpus (fixtures/build_dev_corpus.py)
data/corpus/              real corpus from SQuAD v1.1 (data/build_squad_corpus.py) — CC BY-SA 4.0
eval/
  scenarios/               hand-written + generated scenarios (dev_corpus)
  scenarios_real_corpus/   legacy 15 scenarios (data/corpus) — earlier tuning set
  scenarios_real_dev/      60 dev scenarios (data/corpus) — all decisions made here
  scenarios_real_test/     60 held-out scenarios (data/corpus_test); *_asr = ASR-style variants
  gates/                   G2-G6 scorers
  quality.py, stats.py     gold-referenced answer metrics, Wilson CIs, paired bootstrap
  run_quality.py, compare_runs.py, retrieval_eval.py, asr_style.py
  run_all.py, compare.py, ablate.py, edge_cases.py, scenario_gen.py
  results/                 saved runs behind every number in final_report.md
tests/                     contracts, engine, retrieval, controller, session, eval — 91 tests
docs/                      architecture brief, telemetry schema, benchmark report, demo script
```

## Data

- **`fixtures/dev_corpus/`** — synthetic, seeded, format-driven fixture (policy/venue/travel/
  catering documents), used for fast local iteration and as the primary gate-report corpus.
- **`data/corpus/`** — real, publicly-licensed corpus built from **SQuAD v1.1** (Wikipedia
  prose, CC BY-SA 4.0), 14 articles. Used for all tuning, so treated as the **dev** split.
- **`data/corpus_test/`** — 14 *different* SQuAD articles (`--seed 0`), the **held-out test** split.
  Neither is the hidden Theme 4 evaluation corpus.

Both ship with a `qrels.jsonl` (query → gold `Doc_ID §Section`) so retrieval quality and gate
scores are measured against real ground truth, not eyeballed.

## Results at a glance

`final_report.md` is the authoritative, audited report. In short, measured **once on a held-out
set of 14 SQuAD articles never used for any decision**, against the previous committed system on
the same 98 gold questions (paired 95% CIs):

| | Previous system | Current system |
|---|---|---|
| Answer recall (gold answer in the answer) | 65.3% | **74.5%** (+9.2 [+1.0, +17.8]) |
| Claim precision (asserted claims that are correct) | 70.2% | **87.7%** (+13.5 per turn [+4.0, +23.5]) |
| Correct abstention on unanswerable questions | 0/5 | **5/5** |
| Retrieval r@1 (1,000 held-out questions) | 72.6% | **89.4%** (+16.8 [+14.2, +19.6]) |

All gates (G2–G6) pass on both gate suites and there are zero fabricated citations; 91/91 tests
pass. Note that the earlier headline "G4 grounding 97.3%" was **self-graded and measured on the
tuning set** and has been withdrawn (see `final_report.md` §2). Known open weaknesses: streaming
removes the post-speech delay (< 16 ms vs 141 ms p50) but answers ~4 points fewer questions than the
same pipeline run after the utterance ends (`engine.mode=deferred`, §5.6); and on
lowercase/unpunctuated (ASR-style) input answer recall drops ~12 points (§5.5).

Evaluate a configuration against gold answers, with confidence intervals:

```bash
python -m streaming_rag.retrieval.neural --fetch     # one-time: pinned e5 + cross-encoder ONNX models
python -m eval.run_quality --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test
python -m eval.retrieval_eval --corpus-dir data/corpus_test --sample 1000
```

## What each task owns

| Task | Package |
|---|---|
| 1 — Corpus ingestion & hybrid retrieval | `streaming_rag/retrieval/`, `fixtures/`, `data/` |
| 2 — Stream controller & decomposition | `streaming_rag/controller/` |
| 3 — Session state, synthesis, grounding | `streaming_rag/session/` |
| 4 — Engine, telemetry, eval, packaging | `streaming_rag/engine.py`, `llm.py`, `telemetry/`, `eval/`, Docker, `docs/` |
