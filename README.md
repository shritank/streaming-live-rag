# Streaming Live RAG

Real-time incremental retrieval, multi-intent decomposition, and state-preserving answer
refinement - an event-driven RAG engine that starts searching *while the user is still
speaking*, splits compound requests into parallel sub-queries, and patches an existing answer
in place when a late constraint arrives, instead of restarting from scratch. Every claim is a
verbatim corpus sentence with its `[Doc_ID section]` citation; questions the corpus cannot answer
are refused, not guessed.

Built against the Samsung PRISM Theme 4 guide and its four task briefs (Task 1 corpus retrieval, Task 2 stream
controller, Task 3 session synthesis, Task 4 engine, telemetry and evaluation). Those source documents are kept in the
original workspace next to this repository and are **not part of it**; where code comments and docs cite "Task 4 section 4.3"
or "project context section 6.1", they mean those briefs. **Official scope: streaming transcript -> RAG -> grounded, cited
answer.** (An optional Whisper audio front end exists in `streaming_rag/asr/`; it is outside the
official scope and not part of any headline number. See "Audio input (optional)" below.)

## Reproduce it - one command

```bash
docker compose up --build
```

From this folder, on any machine with Docker. It builds the image (installs the pinned dependencies, fetches the
pinned models with SHA-256 checks, builds the corpora, warms the index), then replays the full scenario suite offline
with no API key and no GPU, and prints the gate report ending in `VERDICT: PASS` (gates G2-G6). No `.env` file is needed.
The first build downloads the models and takes several minutes; later runs reuse the cache.

> **Verified:** on 30 Sep - 1 Oct 2026, on Docker Desktop (Windows 11, WSL 2), this command built the image and the container printed
> `VERDICT: PASS` (47 scenarios, G2-G6 100 %, exit code 0; about 22 minutes end to end, mostly downloads). Caveat: it ran from the working
> tree rather than a fresh clone of a committed repository (`final_report.docx` section 0.7). What was verified is the same steps without a container - fresh environment from the
> lockfile, model fetch, corpus rebuild, the container's command, tests (`final_report.docx` section 5.15). If you do not
> have Docker, use "Quickstart - reference CPU path" below.

## Status (30 Sep 2026)

| | State |
|---|---|
| Engine, all four tasks | Complete. 359 tests pass. |
| Official gates G2-G6 | All pass on the legacy, dev and diagnostic suites (audited on an RTX 6000 Ada; re-run on an RTX 4050 laptop, see "Results at a glance"). |
| Gate G1 (`docker compose up`) | **Passed** (30 Sep - 1 Oct 2026, Docker Desktop / WSL 2): image built, `VERDICT: PASS`, exit 0. Caveat: run from the working tree, not a fresh clone. |
| Demo video | **Uploaded** - link in `demo_video.txt` (https://drive.google.com/drive/folders/1LhGSK8NvrYPaHOn5zm-6vZijQDnDRrjQ?usp=sharing). |
| Presentation | **Done** - `SRMIST_VirtualVanguards_Submission.pptx` (12 slides). |
| AI disclosure form | **Done and signed** - `LangAI3.0_AI_Disclosure.docx`. |
| Audio input | Works end to end (`tools/answer_audio.py`), optional, outside the official scope. |

## Required deliverables - where each one is

The guide's engineering deliverables checklist (section 8) and where its files sit in the tree below; **D6** and **D7** are the
other two items of the submission (presentation and AI disclosure form). The tags **[D1]**-**[D7]** mark the same files in the
directory structure.

| # | Deliverable the guide asks for | File(s) | State |
|---|---|---|---|
| **D1** | **Reproducible repository**: source, pinned dependency lockfiles, environment template, one-command run | This whole `streaming-live-rag/` folder. One command: `docker compose up --build` (top of this README). Source: `streaming_rag/`. Lockfiles: `requirements.lock`, `requirements-cpu.lock`, `requirements-gpu.txt`, `requirements-gpu.lock`. Environment template: `.env.example`. One-command run: `docker compose up` (`Dockerfile`, `docker-compose.yml`) or the clean CLI runner `python -m eval.run_all` (this README, "Quickstart"). Manifest: `run.yaml`. Tests: `tests/`. | Code complete, 359 tests pass. Not yet committed / pushed. Gate G1 (`docker compose up`) passed (see Status). |
| **D2** | **System architecture brief** (<= 6 pages): design rationale, retrieval trigger logic, decomposition strategy, data provenance, trade-offs, failure mitigations | `docs/architecture_brief.docx` (about 1,900 words); diagram as Mermaid code in `docs/architecture_diagram.md` | Done |
| **D3** | **Benchmarking & evaluation report**: comparison against the baseline, >= 3 analysed edge-case failures, two architectural ablations | `final_report.docx` (authoritative): baseline vs streaming section 5.10, ablations 5.11, edge-case failures 5.12. Summary tables: `docs/benchmark_report.md`. Raw runs behind every number: `eval/results/final_audit/`; reproduction commands: `final_report.docx` section 12 and `handoff/final_audit/run_final_audit.sh` | Done |
| **D4** | **Telemetry & observability schema**: end-to-end latencies, retrieval trigger events, answer-version updates, inference cost | `docs/telemetry_schema.md` (human-readable) and `streaming_rag/telemetry/schema.json` (machine-readable, every trace line is validated against it). Code: `streaming_rag/telemetry/` (`sinks.py`, `cost.py`, `trace.py`, `report.py`). Example traces: `eval/results/final_audit/demo/demo1_trace.jsonl`, `demo2_trace.jsonl` | Done |
| **D5** | **System demonstration video** (<= 5 minutes) | Link to the uploaded video: `demo_video.txt` (https://drive.google.com/drive/folders/1LhGSK8NvrYPaHOn5zm-6vZijQDnDRrjQ?usp=sharing). | Uploaded |
| **D6** | **Presentation** (PPT or PDF) | `SRMIST_VirtualVanguards_Submission.pptx` (12 slides, Samsung PRISM template) | Done |
| **D7** | **AI usage disclosure form** | `LangAI3.0_AI_Disclosure.docx` (signed) | Done |

Also specified by the guide: the **structured output event record** (section 4) is what
`python -m streaming_rag.cli replay <scenario> --guide-format` and `python tools/answer_audio.py <clip.wav>` print (see "Output format"),
and the six **evaluation gates G1-G6** (section 5) are scored by `python -m eval.run_all` (`eval/gates/`); results are in
"Results at a glance" below and `final_report.docx` section 0.2.

## Project structure

```
SamsungGenAI/                      the original workspace (only streaming-live-rag/ is in this repository)
|-- Theme 4 Guide_RAG.pdf          the brief: requirements, gates, deliverables         (not in this repo)
|-- Task 1 ... Task 4/             one brief per task (project context.md, task context.md)   (not in this repo)
|-- participant-kit/               material for a different theme (interruptible agents); NOT used here   (not in this repo)
`-- streaming-live-rag/            this repository (everything below)
```

```
streaming-live-rag/
|-- streaming_rag/                 [D1] the product code (importable package)
|   |-- contracts.py               shared dataclasses / protocols - the only coupling surface between tasks
|   |-- config.py                  every tunable and ablation switch, with defaults
|   |-- engine.py                  the event loop: Controller -> Retriever -> Synthesizer, speculative retrieval
|   |-- build.py                   component factory (real vs mock), used by the CLI and the eval harness
|   |-- cli.py                     `replay`, `live`, `evidence` commands; `--guide-format` output
|   |-- gpu.py                     GPU selection: CUDA mandatory when an NVIDIA GPU is present
|   |-- llm.py, mocks.py           LLM client (fake/openai/gemini) and deterministic mock components
|   |-- retrieval/                 Task 1: ingestion, BM25, e5 dense (ONNX), RRF fusion, cross-encoder rerank,
|   |                              ONNX session management, CUDA graphs, SQuAD2 reader export
|   |-- controller/                Task 2: intent stability, decomposition, WAIT / RETRIEVE / SUPPRESS policy
|   |-- session/                   Task 3: session store, extractive synthesis, learned refusal gate
|   |                              (refusal_gate.json), grounding, refinement deltas; generative/ = opt-in LLM path
|   |-- telemetry/                 Task 4 [D4]: JSONL sink, schema.json, cost model, trace assembly, text report
|   `-- asr/                       optional streaming Whisper front end (outside the official scope)
|-- tests/                         [D1] 359 tests, mirroring the package: asr, contracts, controller, engine,
|                                  eval, retrieval, session, plus test_cli.py and test_gpu.py
|-- eval/                          evaluation harness
|   |-- run_all.py                 the official gate runner (G2-G6)
|   |-- run_quality.py, compare.py, ablate.py, latency_profile.py, compare_runs.py, error_analysis.py ...
|   |                              gold-referenced quality, streaming-vs-baseline, ablations, latency, paired stats
|   |-- gates/                     scorers for G2-G6
|   |-- scenarios/                 hand-written + generated replay scenarios (fixture corpus)
|   |-- scenarios_real_dev|test|test2/   60-scenario sets on the SQuAD-derived corpora (test2 is SEALED)
|   |-- scenarios_heysquad_*/      real human-spoken HeySQuAD questions: clean text, dataset ASR text,
|   |                              and Whisper (tiny/base/small) transcripts, for dev / diag / test2
|   `-- results/                   saved runs behind every reported number [D3] (final_audit/ = the final pass; final_audit/demo/ = demo traces [D4, D5])
|-- fixtures/                      Task 1 fixture: build_dev_corpus.py + dev_corpus/ (41 synthetic company docs)
|-- data/                          real corpora built from SQuAD v1.1
|   |-- corpus/                    DEV split (14 articles) - every decision was made here
|   |-- corpus_test/               DIAGNOSTIC split (14 different articles) - confirmation only
|   |-- corpus_test2/              SEALED split (16 articles) - run once, not used for tuning
|   |-- corpus_miniwiki/           small out-of-domain check corpus
|   |-- raw/                       downloaded SQuAD / HeySQuAD source files; build_squad_corpus.py rebuilds the corpora from them
|-- docs/                          architecture_brief.docx [D2, Word, 3 pages], architecture_diagram.md [D2], telemetry_schema.md [D4],
|                                  benchmark_report [D3], demo_script [D5], experiment_log, generative_synthesis, landscape_audit
|-- reports/                       default output folder of `eval.edge_cases` / `eval.diagnose` (generated files, git-ignored); the edge-case analysis itself is in final_report.docx 5.12
|-- handoff/                       tooling and records from the measurement passes
|   |-- gpu_tools/                 GPU verification and benchmarks (verify_product_gpu.py, fuse_models.py ...)
|   |-- asr_tools/                 Whisper models (tiny/base/small .en), 997 HeySQuAD WAV clips, transcribe scripts
|   |-- asr_results/, newmachine/    raw outputs and logs from the audio / GPU passes
|   `-- final_audit/               the exact scripts that produced the final-audit results
|-- tools/                         small standalone helpers
|   |-- answer_audio.py            WAV file -> Whisper -> engine -> cited answer (guide-format record)
|   `-- auto_recall_lab.py         answer-recall diagnostics
|-- generated/                     generated practice scenarios (e.g. chit-chat cases)
|-- final_report.docx              the authoritative, audited benchmarking & evaluation report [D3] (Word, 13 pages)
|-- README.md                      this file [D1, run instructions]
|-- demo_video.txt                 link to the uploaded demo video (Google Drive) [D5]
|-- SRMIST_VirtualVanguards_Submission.pptx    the presentation [D6]
|-- LangAI3.0_AI_Disclosure.docx   the signed AI usage disclosure form [D7]
|-- requirements.lock              [D1] top-level pinned CPU environment (Python 3.11-3.12); what the Dockerfile installs
|-- requirements-cpu.lock          [D1] FULLY pinned CPU environment, transitive deps included (Linux x86-64, Python 3.12)
|-- requirements-gpu.txt           [D1] top-level pinned GPU environment (CUDA 12 wheels)
|-- requirements-gpu.lock          [D1] FULLY pinned GPU environment, exactly the tested one (Windows, Python 3.11.9); install with --no-deps
|-- run.yaml                       [D1] manifest: entry points, env vars, corpus locations
|-- Dockerfile, docker-compose.yml, .dockerignore    [D1] CPU reference image, one-command gate run (`docker compose up`)
|-- Makefile                       shortcuts (needs `make`; on Windows run the underlying commands)
|-- pytest.ini, .env.example [D1, environment template], .gitignore
|-- .cache/streaming_rag/          fetched / converted models and cached corpus embeddings (git-ignored)
`-- .venv*/                        virtual environments - see the note under "Set up" (not portable)
```

## Two environments

| | Reference (CPU) | GPU (how every reported number was produced) |
|---|---|---|
| Install | `requirements.lock` (Python 3.11-3.12), Docker image | `requirements-gpu.txt` (adds `onnxruntime-gpu`, CUDA 12 / cuDNN 9 wheels) |
| Neural inference | ONNX Runtime CPU | ONNX Runtime **CUDA only** |
| Selected by | `STREAMING_RAG_ORT_PROVIDER=cpu` (explicit opt-out; also what the image sets) | default `auto` |

On a machine with an NVIDIA GPU the code makes CUDA **mandatory**: if it cannot be used, startup stops
with `GpuUnavailable` - it never silently falls back to the CPU (`streaming_rag/gpu.py`). A machine
without a GPU (e.g. the Docker reference image) runs the identical code on the CPU. Answers are the same
on both (dev answers were bit-identical across CPU and GPU).

The GPU environment has been built and run on two machines: an RTX 6000 Ada workstation (the audited
numbers) and an RTX 4050 Laptop GPU with 6 GB (Windows 11, driver 596.49, Python 3.11.9). The 6 GB card runs
the whole pipeline, including Whisper `small.en`, using about 2.7 GB.

## Set up (Windows, GPU) - tested

```bat
cd streaming-live-rag
python -m venv .venv-local
.venv-local\Scripts\activate.bat                  :: PowerShell: .\.venv-local\Scripts\Activate.ps1
python -m pip install -r requirements-gpu.txt
python -m streaming_rag.retrieval.neural --fetch  :: pinned model revisions, SHA-256 verified
set STREAMING_RAG_ORT_PROVIDER=cuda               :: PowerShell: $env:STREAMING_RAG_ORT_PROVIDER = "cuda"
pytest tests -q                                   :: expect: 359 passed
```

Notes:
* **Do not reuse a `.venv` / `.venv-gpu` folder that came with a copy of the project.** A virtual environment
  stores absolute paths to the machine it was created on; the ones shipped with this folder point at another
  user's profile and do not run. Create a fresh one as above.
* In `cmd`, run the `.bat` activation script. Running the `.ps1` script from `cmd` opens it in Notepad.
* For a byte-for-byte reproduction of the tested environment use the fully pinned file instead of the top-level list:
  `python -m pip install --no-deps -r requirements-gpu.lock` (`--no-deps` is required; see the file header).
  The Whisper models and the HeySQuAD audio clips are not committed (they are git-ignored): fetch them with
  `handoff/asr_tools/fetch_whisper.py` and `fetch_audio.py` before using the audio front end.
* `make` and Docker are not needed for anything below; the Makefile targets are plain commands (see the Makefile).
* The ONNX reader (SQuAD2 -> ONNX) and the fused CUDA graphs ship in `.cache/streaming_rag/models/`. If they are
  missing, `python -m streaming_rag.retrieval.export_reader` and `python -m streaming_rag.retrieval.neural --fuse`
  rebuild them (the first needs torch + transformers 4.46.3 in a throwaway venv, exactly like Dockerfile stage 1).
* Never let plain `onnxruntime` sit next to `onnxruntime-gpu` - they share one package folder and the CPU build
  silently wins. Installing `faster-whisper` (audio, below) pulls the CPU one in; fix with
  `pip uninstall -y onnxruntime onnxruntime-gpu` then `pip install --no-deps onnxruntime-gpu==1.20.0`.

## Quickstart - GPU (Linux / macOS shells)

```bash
python -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements-gpu.txt
# only onnxruntime-gpu may be installed (see the caveat at the top of requirements-gpu.txt); or: make install-gpu
python -m streaming_rag.retrieval.neural --fetch     # pinned model revisions, SHA-256 verified
python -m streaming_rag.retrieval.export_reader      # once: SQuAD2 reader -> ONNX (needs torch + transformers; see requirements-gpu.txt)
python -m streaming_rag.retrieval.neural --fuse      # optional CUDA graphs (~2x faster model calls); falls back to the original graphs on the GPU if absent

python handoff/gpu_tools/verify_product_gpu.py       # controlled test: every model bound to CUDA, per-operator placement
```

`verify_product_gpu.py` also loads Whisper, so it needs the optional audio packages (below) and fails without them.

## Quickstart - reference CPU path / Docker

```bash
docker compose up --build        # builds the image, runs the full legacy gate suite (offline, no API key needed)
```

(`.env` is optional: copy `.env.example` to `.env` only to override a setting such as an LLM provider key.)

Without Docker:

```bash
pip install -r requirements.lock
export STREAMING_RAG_ORT_PROVIDER=cpu          # required if the machine has an NVIDIA GPU but this CPU-only environment
python -m streaming_rag.retrieval.neural --fetch
python -m streaming_rag.retrieval.export_reader
python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus       # the committed fixture corpus is a byte-identical re-build (checked in the clean-room run)
python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1
```

## Run it

```bash
# the guide's reference scenarios (speed-up 8x; use --time-scale 1 for real time)
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --corpus-dir fixtures/dev_corpus --time-scale 8
python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry trace.jsonl
python -m streaming_rag.telemetry.report trace.jsonl                                       # per-turn timeline: decisions, retrievals, versions, latency, cost
python -m streaming_rag.cli evidence "cancellation policy pune" --corpus-dir fixtures/dev_corpus   # BM25 / dense ranks + cross-encoder score
python -m streaming_rag.cli live                     # type or paste text; chunked at speaking speed
make demo                                            # the <= 5 minute demonstration, docs/demo_script.md (without make: run its commands)

# official procedure on the GPU: real-time replay, 3 reps, median, gates G1-G6 (make eval-gpu)
python -m eval.run_all --scenarios eval/scenarios_real_dev --corpus-dir data/corpus --time-scale 1 --reps 3
python -m eval.run_all --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1 --reps 3

# guide deliverables: baseline vs streaming, ablations, gold-referenced quality with confidence intervals
python -m eval.compare --modes streaming,deferred,baseline --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1
python -m eval.ablate --scenarios eval/scenarios_real_dev --corpus-dir data/corpus
python -m eval.run_quality --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test
python -m eval.latency_profile --scenarios eval/scenarios_real_dev --corpus-dir data/corpus --time-scale 1   # p50/p95/p99/max per stage
python -m eval.compare_runs baseline.json candidate.json                                                    # paired bootstrap CIs

pytest tests/ -q
```

`handoff/final_audit/run_final_audit.sh` is the exact script that produced the final-audit results in
`eval/results/final_audit/`.

## Output format

Each completed turn is a `turn_result` record. Add `--guide-format` to `replay` (or use `tools/answer_audio.py`) to
print it as the guide's *structured output event record* - the five fields the guide specifies:

```bash
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --corpus-dir fixtures/dev_corpus --time-scale 8 --guide-format
```

```json
{
  "retrieval_events": [ { "timestamp_s": 0.8, "query": "plan a customer workshop in Pune for 30 people", "trigger": "provisional" },
                        { "timestamp_s": 1.6, "query": "cancellation policy plan customer workshop pune", "trigger": "multi_intent" } ],
  "sub_queries": [ "plan a customer workshop in Pune for 30 people", "cancellation policy plan customer workshop pune" ],
  "answer": "Venue B supports a maximum workshop capacity of 25 attendees ...",
  "citations": [ "Doc_00 §2", "Doc_16 §1" ],
  "uncertainty": null
}
```

(abridged). Without the flag, `replay` prints the full `turn_result`, which adds `session_id`, `utterance_id`,
`answer_version`, `parent_version`, `retrieval_required` and `reason`. `trigger` is why a search fired:
`provisional` (early guess while the user is still speaking), `multi_intent` (a compound request was split),
`refinement` (a late detail arrived). `uncertainty` is `null` unless part of the request could not be verified.

## Audio input (optional, outside the official scope)

A streaming Whisper front end turns a WAV file into the same `transcript_chunk` / `utterance_end` events the engine
already reads (`streaming_rag/asr/whisper.py`; it commits only words two consecutive decodes agree on).

```bash
pip install faster-whisper==1.2.1 ctranslate2==4.8.2     # then repair onnxruntime, see the note under "Set up"
python tools/answer_audio.py handoff/asr_tools/heysquad_audio/5725b81b271a42140099d09b.wav     # -> guide-format record
python tools/answer_audio.py CLIP.wav [CLIP2.wav ...] --corpus-dir data/corpus --time-scale 8   # options: --model, --device, --raw
```

The clip must be WAV (16-bit PCM; other sample rates and stereo are converted). The corpus must be the one the
question is about. The transcript Whisper heard is printed to stderr, the record to stdout.

Measured on the RTX 4050 laptop (this repo, 30 Sep 2026; answer recall is gold-referenced, HeySQuAD audio, 100 clips per set):

| Whisper model | Dev answer recall | Dev claim precision | Dev over-split | Diag answer recall |
|---|---|---|---|---|
| `base.en` | 66% | 93.8% (75/80) | 7% | 72% |
| `small.en` (the tool's default) | 70% | 100% (79/79) | 0% | 76% |

`small.en` is significantly better on dev citation recall (+5 points), claim precision and over-splitting; on the
held-out set it points the same way but is not significant. Typed text on the same kind of question scores higher
(clean 67.5% vs Whisper `base.en` 52.5% vs the dataset's own ASR text 37.5% on a 40-clip check), and the
remaining audio errors are mostly Whisper mishearing rare names and the refusal gate sitting just under its
threshold on borderline items. Post-speech latency is not the text figure: for 41 of 100 clips the last words arrive as the
speaker finishes, so the final search runs after speech. Engine post-speech on 100 clips (RTX 4050): p50 0.74 ms, p95 100 ms,
p99 181 ms, against p99 1.4 ms for text; Whisper's own final decode adds a median 149 ms. Separate text and speech latency
tables: `final_report.docx` section 0.4; what was tried and rejected: section 0.9.

## Repository layout (code)

The annotated tree above is the folder map. Package internals:

```
streaming_rag/
  contracts.py          shared dataclasses/protocols - the only coupling surface (task brief section 6.1; the briefs are kept outside this repository)
  config.py             every tunable + ablation switch
  gpu.py                GPU selection: CUDA mandatory when an NVIDIA GPU is present, no silent CPU fallback
  llm.py                async LLMClient: fake (offline, default) | openai | gemini (never used on the default path)
  mocks.py              deterministic mock Controller/Retriever/Synthesizer, with fault injection
  engine.py             Controller -> Retriever -> Synthesizer on one asyncio loop; speculative retrieval, telemetry
  build.py              component factory (real vs mock), used by the CLI and the eval harness alike
  cli.py                `replay` / `live` / `evidence`
  retrieval/            ingest (format-driven), BM25, e5-small-v2 dense (ONNX), RRF fusion, cross-encoder rerank,
                        neural.py (ONNX sessions, warm-up, CUDA graphs), graph_static.py (exact static-shape rewrite)
  controller/           intent-stability scoring, decomposition, WAIT/RETRIEVE/SUPPRESS policy, preview_final()
  session/              ephemeral session store, extractive synthesis, refusal gate, grounding, refinement delta engine
                        (session/generative/ = opt-in LLM synthesis, disabled by default, unmeasured)
  telemetry/            JSONL sink, schema, cost model, trace assembler, text dashboard
  asr/                  optional streaming Whisper front end (outside the official scope)
```

## Data

- **`fixtures/dev_corpus/`** - synthetic, seeded, format-driven fixture (policy / venue / travel / catering
  documents) used by the legacy gate suite (`docker compose up`).
- **`data/corpus/`** - real, publicly licensed corpus built from **SQuAD v1.1** (Wikipedia prose,
  CC BY-SA 4.0), 14 articles. Every decision was made on this **dev** split.
- **`data/corpus_test/`** - 14 *different* articles: the **diagnostic** split, used only to confirm.
- **`data/corpus_test2/`** - 16 further articles: the **sealed** split. It was run exactly once (28 Sep,
  `final_report.docx` section 5.8) and is not used for any decision. Any further run needs an explicit decision
  and must be labelled a second use.
- **HeySQuAD** (CC BY 4.0) supplies real human-spoken questions (text columns only for the official scope),
  split the same way. Its audio clips (`handoff/asr_tools/heysquad_audio/`) are used only for the optional audio front end.

None of these is the hidden Theme 4 evaluation corpus, and nothing in `streaming_rag/` references any of their
ids or content. Each corpus ships with a `qrels.jsonl` (query -> gold `Doc_ID section`).

## Results at a glance

`final_report.docx` is the authoritative, audited report; the numbers below are from the final audit
(30 Sep 2026, RTX 6000 Ada, `eval/results/final_audit/`). DEV = used to decide, DIAG = held-out confirmation.

**Gates** (`eval.run_all --reps 3`, median of 3 fresh engines; real time = `--time-scale 1`):

| Suite (`eval.run_all --reps 3`) | Scenarios | G2 early retrieval (false triggers) | G3 multi-intent | G4 grounding (fabricated) | G5 refinement | G6 telemetry | Verdict |
|---|---|---|---|---|---|---|---|
| Legacy synthetic (fixtures/dev_corpus), 8x | 47 | 16/16 = 100.0%  (false-trigger rate 0.0%) | 10/10 = 100.0% | 68/68 = 100.0%  (fabricated=0) | 10/10 = 100.0% | 67/67 = 100.0%  (schema errors=0) | **PASS** |
| Real corpus, 15 scenarios, 8x | 15 | 11/11 = 100.0%  (false-trigger rate 0.0%) | 5/5 = 100.0% | 36/36 = 100.0%  (fabricated=0) | 5/5 = 100.0% | 25/25 = 100.0%  (schema errors=0) | **PASS** |
| **DEV**, 60 scenarios, **real time (1x)** | 60 | 45/45 = 100.0%  (false-trigger rate 0.0%) | 20/20 = 100.0% | 113/113 = 100.0%  (fabricated=0) | 20/20 = 100.0% | 90/90 = 100.0%  (schema errors=0) | **PASS** |
| **DIAG**, 60 scenarios, **real time (1x)** | 60 | 49/49 = 100.0%  (false-trigger rate 0.0%) | 20/20 = 100.0% | 114/114 = 100.0%  (fabricated=0) | 20/20 = 100.0% | 90/90 = 100.0%  (schema errors=0) | **PASS** |

G1 (`docker compose up`) **passed** on 30 Sep - 1 Oct 2026 (Docker Desktop, WSL 2, RTX 4050 machine; `VERDICT: PASS`, exit 0), from the working
tree rather than a fresh clone; a documented CLI clean-room substitute also passed - see `final_report.docx` sections 0.7 and 5.15.

**Reproduced on a second machine** (RTX 4050 Laptop, 6 GB, 30 Sep 2026): the legacy synthetic suite at 8x gives
G2-G6 all 100% (47 scenarios, fabricated=0, schema errors=0), **PASS**; the 359 tests pass in ~24-28 s; and the
HeySQuAD DIAG typed set (219 answerable + 219 unanswerable) reproduces the audited row below **exactly**
(answer recall 168/219 = 76.7%, claim precision 184/258 = 71.3%, abstention 152/219 = 69.4%). The demo
commands in `docs/demo_script.md` return the same answers, citations and versions.

**Answer quality** (gold-referenced, 95 % Wilson intervals; the text sets have only 5 unanswerable questions, the
HeySQuAD sets ~50 %):

| Set | Answer recall | Citation recall | Claim precision | Abstention | Exact decomp. | Over-split |
|---|---|---|---|---|---|---|
| Text DEV (102 Q) | 65.7% [56.1%, 74.2%] | 72.5% [63.2%, 80.3%] | 87.4% [78.8%, 92.8%] | 100.0% [56.6%, 100.0%] | 100.0% [83.9%, 100.0%] | 0.0% [0.0%, 9.9%] |
| Text DIAG (98 Q) | 78.6% [69.5%, 85.5%] | 84.7% [76.3%, 90.5%] | 94.3% [87.4%, 97.5%] | 100.0% [56.6%, 100.0%] | 100.0% [83.9%, 100.0%] | 0.0% [0.0%, 9.9%] |
| HeySQuAD DEV typed (279 answerable + 279 unanswerable) | 71.7% [66.1%, 76.6%] | 76.7% [71.4%, 81.3%] | 71.6% [66.3%, 76.3%] | 70.3% [64.6%, 75.3%] | n/a | 0.4% [0.1%, 1.3%] |
| HeySQuAD DIAG typed (219 + 219) | 76.7% [70.7%, 81.8%] | 83.6% [78.1%, 87.9%] | 71.3% [65.5%, 76.5%] | 69.4% [63.0%, 75.1%] | n/a | 1.4% [0.6%, 3.0%] |

**Latency, real time, GPU only, nothing excluded** (ms; the DEV replay, 90 turns; DIAG in `final_report.docx`
section 0.4). "Post-speech" = `utterance_end` handled -> answer emitted, for typed / transcript input:

| Boundary | p50 | p95 | p99 | max |
|---|---|---|---|---|
| Controller decision per chunk | 0.50 | 0.93 | 1.31 | 1.83 |
| Answer assembly | 1.47 | 2.08 | 2.16 | 2.27 |
| **Post-speech, utterance end -> answer** | **2.33** | **3.06** | **3.35** | **3.57** |
| Retrieval per sub-query (during speech) | 82.3 | 140.3 | 179.7 | 254.5 |

p99 < 5 ms holds for the boundaries above (DEV and DIAG: 0 of 90 turns over 5 ms) and **not** for retrieval /
reranking / claim selection (tens to hundreds of ms, hidden behind the user's speech) or for audio input
(outside the official scope). Baseline (retrieve after the utterance ends): 110 ms at the median, 46x slower
on the 75 paired turns; streaming and deferred give identical answers (`final_report.docx` section 5.10).


## Known weaknesses (stated plainly)

- **Answer recall has a ceiling** set by two stages: the refusal gate withholding a correct claim (11 of 35
  dev misses) and a claim taken from a non-gold section ranked above the gold one (10 of 35). Tested levers
  (bigger reranker, other embedder, reader as selector, extra gate feature, and - for audio - cleaning
  punctuation before the gate) did not clear the significance bar.
- **Audio is weaker than text**: Whisper mishears rare names, the gate refuses some correct-looking questions
  that sit just under its 0.4 threshold, and the last words arrive after speech ends (`final_report.docx` section 0.9).
- **Gate G1 passed with caveats**: `docker compose up --build` built the image and printed `VERDICT: PASS` on Docker Desktop
  (RTX 4050 machine), but from the working tree rather than a fresh clone of a committed repository, with two early layers cached
  from earlier failed attempts. A CLI clean-room run (fresh venv, fresh model cache, corpus rebuild, tests) also passed
  (`final_report.docx` section 5.15).
- **The cost-accuracy check against a real provider** (5 live calls within 2 %, Task 4 section 4.3) was not run:
  no API key, and the default pipeline makes no LLM call.
- **`controller.mode=model` (rule vs LLM controller) is not implemented**, so that ablation was not run
  (`eval.ablate` says so); it would need an LLM key.
- The refusal gate uses a SQuAD2-trained reader on SQuAD-derived corpora (in-domain question style): its
  measured benefit is probably optimistic (`final_report.docx` section 10).
- The generative synthesis path (`session/generative/`) is wired and tested for structure but has no
  gold-referenced measurement; it is disabled by default.
- The sealed-test numbers predate the later code changes (GPU layer, speculation, controller fixes); they
  were deliberately not re-run.
- Latency is sensitive to the machine's power plan (CPU wake-up after idle); see `final_report.docx`.

## What each task owns

| Task | Package |
|---|---|
| 1 - Corpus ingestion & hybrid retrieval | `streaming_rag/retrieval/`, `fixtures/`, `data/` |
| 2 - Stream controller & decomposition | `streaming_rag/controller/` |
| 3 - Session state, synthesis, grounding | `streaming_rag/session/` |
| 4 - Engine, telemetry, eval, packaging | `streaming_rag/engine.py`, `llm.py`, `telemetry/`, `eval/`, Docker, `docs/` |

## What only you can run

These need something the machines used so far did not have; nothing below is claimed as done.

1. **Optional - the strictest form of G1**: `docker compose up --build` from a fresh clone of the committed repository on a
   pristine machine, timed (the passed run used the working tree and partly cached layers). Also run it once with egress
   restricted if you want the proxy-log version of the corpus-isolation check (a socket-level version passed:
   `python handoff/final_audit/netcheck.py`).
2. **Python 3.10 and 3.12**: `pip install -r requirements.lock && pip check && python -c "import streaming_rag"`
   (only 3.11 has been available).
3. **Check the demo video against the repository**: the video (link in `demo_video.txt`) was made from `docs/demo_script.md`, whose
   narration names the RTX 6000 Ada and quotes that machine's latencies; if the video says the same, it should say it was
   recorded from audited figures, or quote the RTX 4050 figures instead (`final_report.docx` section 0.4).
4. **Optional - cost accuracy**: with a provider key in `.env`, run `synthesis.mode=generative` on 5 calls and compare the
   summed `cost_usd` with the provider's usage (Task 4 section 4.3).
5. **Decision - the refusal-gate default**: it costs answer recall when few questions are unanswerable and is worth it
   above roughly 9 % unanswerable (`final_report.docx` section 5.11). Switch: `synthesis.refusal_gate` = `learned` (default) | `none`
   (`--set synthesis.refusal_gate=none` on the eval runners, or `config.py`).
