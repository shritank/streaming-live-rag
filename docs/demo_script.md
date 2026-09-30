# Demo script (target 4:30, hard limit 5:00)

**Status: the demo video has been recorded and uploaded (link in `demo_video.txt`); this script is the record of what it
shows and the procedure to repeat it.** Every "expected" value below is copied from a real run of the shipped system on the
RTX 6000 Ada (GPU only), saved in `eval/results/final_audit/demo/`; the commands were re-run on an RTX 4050 laptop with the
same answers, citations and versions (see the Windows section before the recording procedure). Your run will show the same answers, citations and versions; times
(ms) differ by a few tenths.

The demo shows the six guide demonstrations plus the GPU and telemetry evidence. It uses the scripted
transcript-stream scenarios that ship with the repository (the transcript chunks are replayed at their
real timestamps, `--time-scale 1`), the fixture corpus (`fixtures/dev_corpus`, 41 synthetic company documents)
and the real pipeline: BM25 + e5 + cross-encoder on the GPU. Nothing is mocked. `--warmup` makes the process
do one uncounted fast replay first, as the evaluation harness does, so first-call GPU start-up is not
shown as latency.

## Before recording (2 minutes, not on camera)

```bash
# NVIDIA machine, environment from requirements-gpu.txt (README section "GPU")
export STREAMING_RAG_ORT_PROVIDER=cuda        # PowerShell:  $env:STREAMING_RAG_ORT_PROVIDER = "cuda"
python -m streaming_rag.retrieval.neural --verify      # models present and SHA-256 verified
nvidia-smi --query-gpu=name,memory.used --format=csv   # RTX 6000 Ada visible
python -m pytest -q                                    # optional: 359 passed
```
Terminal: 110 columns, font >= 16 pt, dark theme, `clear` between segments. Pre-open `docs/demo_script.md`
is NOT needed on screen. Record at 1080p; the whole run needs no editing except trimming model-load pauses.

## 0:00-0:25  What it is, and that it runs on the GPU

Say: "A streaming transcript arrives in chunks while the user is still speaking. The system decides when to
retrieve, splits multi-part requests, fuses and reranks evidence, and answers with citations and
telemetry - all neural inference runs on the RTX 6000 Ada."

```bash
nvidia-smi --query-gpu=name --format=csv,noheader          # -> NVIDIA RTX 6000 Ada Generation
python -m streaming_rag.retrieval.neural --verify
```

## 0:25-1:35  Segment 1 - streaming chunks, early retrieval, multi-intent decomposition

```bash
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json \
    --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo1.jsonl
```
The scenario streams three chunks and the end of the utterance (t = 0.0, 0.8, 1.6, 2.1 s):
"I need to plan a customer workshop in" / " Pune for 30 people, and I need" / " the cancellation policy and
the catering options."

Point out, then show it in the trace (next command):
* at 0.0 s the controller **waits** (no stable intent yet);
* at **0.8 s, 1.3 s before the user stops**, it retrieves: `trigger=provisional`, query "plan a customer
  workshop in Pune for 30 people";
* at 1.6 s it **decomposes** the rest into two more sub-queries in parallel (`trigger=multi_intent`):
  "cancellation policy ..." and "catering options ...";
* the answer has three sentences, one per intent, with three citations:
  `Doc_00 Â§2`, `Doc_16 Â§1`, `Doc_24 Â§3`, version 1.

Expected JSON (abridged): `"reason": "multi_intent"`, `"answer_version": 1`, `"parent_version": null`,
`"citations": ["Doc_00 Â§2", "Doc_16 Â§1", "Doc_24 Â§3"]`, `"uncertainty": null`.

## 1:35-2:25  Segment 2 - telemetry and the post-speech latency

```bash
python -m streaming_rag.telemetry.report demo1.jsonl
```
Show, top to bottom: `controller_decision` (wait -> retrieve -> retrieve), `retrieval_started` x3 with
`mode=hybrid`, `retrieval_completed` with `latency_ms` and the ranked `chunk_ids`,
`answer_version_created`, `grounding_checked n_claims=3 n_supported=3 fabricated_citations=[]`, and the
`output_emitted` line:
`timings={'final_decision_ms': 0.63, 'retrieval_wait_ms': 0.26, 'synthesis_ms': 1.29, 'post_speech_ms': 2.28}`.
Say: "The answer left 2.3 ms after the user stopped speaking; the retrieval and claim selection had
already been done during speech. Cost is $0.000000 because the default path makes no LLM call."
(Schema: `docs/telemetry_schema.md`; every line validates against `streaming_rag/telemetry/schema.json`.)

## 2:25-3:05  Segment 3 - evidence fusion and reranking

```bash
python -m streaming_rag.cli evidence "cancellation policy plan customer workshop pune" --corpus-dir fixtures/dev_corpus
```
Expected (real output):
```
final  citation      score   BM25-rank  dense-rank  ...
    1  Doc_16 Â§1      4.68          0           0
    2  Doc_16 Â§4      4.07          1           3
    3  Doc_16 Â§3      3.96         12           1
    4  Doc_16 Â§2      3.95         11           2
    5  Doc_22 Â§1     -2.52          9           5
```
Point at row 3: `Doc_16 Â§3` is **12th** for BM25 but **1st** for the dense retriever; reciprocal-rank fusion
keeps it in the candidate pool and the cross-encoder puts it 3rd. The other Pune cancellation sections
outrank the same-worded Indore / Hyderabad policies, whose scores drop below zero. (The score column is the
cross-encoder logit. Ranks start at 0.)

## 3:05-4:05  Segment 4 - late-arriving detail refined without restarting; query suppression

```bash
python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json \
    --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo2.jsonl
python -m streaming_rag.telemetry.report demo2.jsonl
```
Three turns (t = 0-2 s, 4-6 s, 8-9 s):
1. "Summarize the travel reimbursement rule for an employee trip in Sales" -> **version 1**, one citation,
   `Doc_30 Â§1`; retrieval started at 0.0 s, before the sentence was finished.
2. "Actually, the trip was international and the booking was made after travel." -> **version 2,
   `parent_version: 1`**, `reason: refinement_delta`. Only the *new* details were searched
   (`trip was international`, `booking was made after travel ...`, `trigger=refinement`); version 1 was not
   recomputed. The answer keeps the version-1 sentence and its citation and adds two sentences citing
   `Doc_32 Â§3` and `Doc_32 Â§5`.
3. "Please repeat your last answer in two bullets." -> **version 3, `parent_version: 2`**,
   `retrieval_events: []`, `retrieval_required: false`, `reason: presentation_restructure`. The controller
   **suppressed** retrieval: in the trace `decision=suppress, turn_kind=presentation_only`, no
   `retrieval_started` line. The two bullets carry their citations `[Doc_30 Â§1]` and `[Doc_32 Â§3]`.

## 4:05-4:35  Segment 5 - grounded citations and the official gates

Open one cited source next to the answer (any editor):
`fixtures/dev_corpus/Doc_32_*.md`, section 3 - it contains the "senior director approval obtained before
booking" sentence quoted in turn 2. Then show the recorded gate result (do **not** re-run it on camera at real time; the 8x run below takes about a minute):
```bash
python -m eval.run_all --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus --time-scale 8 --reps 1
```
(about 1 minute; prints G2 100 %, G3 100 %, G4 100 % with `fabricated=0`, G5 100 %, G6 100 %, VERDICT: PASS).
Say: "On the real-time, three-repetition runs on the development and diagnostic splits, all of G2-G6
pass; the report has the numbers, and the gate G4 figure is self-graded, so the gold-referenced claim
precision is reported separately."

## 4:35-5:00  Close

Show `final_report.docx` section 0 (requirement checklist) and section 5.10 (streaming vs baseline: post-speech
2.38 ms against 110.18 ms p50 on 75 paired turns). State plainly what is not covered: the status of the Docker `compose up`
gate (G1) is recorded in `final_report.docx` section 0.7 (a CLI clean-room substitute passed), and no LLM provider key was
available, so the optional generative synthesis is unmeasured.

---

## Windows `cmd` version of the commands (RTX 4050 laptop; `make` is not needed)

Run from the `streaming-live-rag` folder after `.venv-local\Scripts\activate.bat` and `set STREAMING_RAG_ORT_PROVIDER=cuda`.
The `\` line breaks above become one line each in `cmd`. The two replay commands also accept `--guide-format` to print
the guide's structured output record instead of the raw JSON.

```bat
nvidia-smi --query-gpu=name --format=csv,noheader
python -m streaming_rag.retrieval.neural --verify
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo1.jsonl
python -m streaming_rag.telemetry.report demo1.jsonl
python -m streaming_rag.cli evidence "cancellation policy plan customer workshop pune" --corpus-dir fixtures/dev_corpus
python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo2.jsonl
python -m streaming_rag.telemetry.report demo2.jsonl
notepad fixtures\dev_corpus\Doc_32_travel_customer.md
python -m eval.run_all --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus --time-scale 8 --reps 1
```

On the RTX 4050 the answers, citations and versions are the same as the expected values in this script; say the GPU is an
"NVIDIA GeForce RTX 4050 Laptop GPU" and quote your own latency (segment 1 post-speech about 0.8 ms; the workstation
figure below is 2.28 ms). Optional extra, outside the official scope: `python tools\answer_audio.py
handoff\asr_tools\heysquad_audio\5725b81b271a42140099d09b.wav` answers a spoken question from a WAV file.
`demo*.jsonl` traces are git-ignored.

---

## Recording procedure (for the person recording)

1. Run every command above once, off camera, to check the environment (`make demo` runs segments 1-4).
2. Record the terminal only; no need for a face cam. Keep the commands as written (copy-paste).
3. Do not speed up or cut inside the replay: the real-time pacing is the point (chunks arrive on the
   scenario's clock and the answer appears 2-3 ms after the utterance end).
4. If a p50 figure is quoted, quote it from `final_report.docx` section 0.4 and say "median"; do not claim the
   sub-5 ms figure holds for anything but the post-speech latency of text input.
5. Save `demo1.jsonl` / `demo2.jsonl` next to the video as evidence.
6. Typed input works too, for a live question at the end: `python -m streaming_rag.cli live`
   (type a sentence, blank line = end of utterance).

## Reference outputs

`eval/results/final_audit/demo/`: `demo1_stdout.txt`, `demo1_report.txt`, `demo1_trace.jsonl`,
`demo2_stdout.txt`, `demo2_report.txt`, `demo2_trace.jsonl`, `demo_evidence.txt`.
