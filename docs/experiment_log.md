# Experiment log — answer-quality optimisation pass (28 Sep 2026)

Protocol: every decision is made on **dev** (text: `data/corpus` scenarios; speech: HeySQuAD dev
clean/ASR) and confirmed on **diagnostic** (text: `data/corpus_test`; speech: HeySQuAD diag).
**Sealed test2 (text and HeySQuAD test2 clean/ASR) is not used for any decision in this log.** It was
already run once for `final_report.docx`; any later sealed run is a *second* use and is labelled so.
Starting point: checkpoint `6fb6305` + the uncommitted reconciliation changes documented in
`final_report.docx` §11 (backed up as a patch before this pass began).

Labels: **DEV** (used to decide), **DIAG** (confirmation, held out from the decision),
**OFFLINE** (computed from saved sweep outputs, not an end-to-end run).

---

## E0 — Baseline (current code, no changes) [DEV/DIAG]

| Set | Answer recall | Citation recall | Claim precision | Abstention | Exact decomp. |
|---|---|---|---|---|---|
| Text dev (102 Q) | 62.7% | 67.6% | 84.5% | 5/5 | 100% |
| Text diag (98 Q) | 76.5% | 83.7% | 93.1% | 5/5 | 100% |
| HeySQuAD dev clean (279 Q) | 71.3% (EA) / 70.6% (run) | — | — | — | — |
| HeySQuAD dev provided-ASR (279 Q) | 37.6% | — | — | — | — |

(EA = error-analysis run; the ±0.7 point difference is one item's non-determinism in the merged
final answer, investigated in E2 notes.)

## E1 — Failure taxonomy and oracle gains [DEV/DIAG]

Tool: `eval/failure_taxonomy.py` over `eval/error_analysis.py` records (extended this pass to split
the learned refusal gate from the selector threshold, detect cross-intent contamination and
wrong-document selection, and flag answer-right-but-citation-wrong).

Oracle gain = answer-recall points recovered if that stage alone were perfect.

| Stage | Text dev | Text diag | HeySQuAD clean | HeySQuAD provided-ASR |
|---|---|---|---|---|
| Learned refusal gate (H1) | +10.8 | +5.1 | **+12.5** | **+21.9** |
| Evidence selection, wrong section (B) | +11.8 | +3.1 | +1.1 | +1.1 |
| Sentence selection (C) | +3.9 | +4.1 | +6.1 | +3.2 |
| Retrieval (A) | +3.9 | +1.0 | +0.4 | +9.7 |
| Decomposition (E) | 0 | 0 | 0 | +3.2 |
| Cross-intent contamination (F) | +1.0 | 0 | — | — |
| Streaming vs deferred (J) | 0 | 0 | — | — |
| **ASR (paired clean→ASR, G)** | — | — | — | **+33.7** |

Paired clean/ASR (same 279 questions): 105 answered from the typed question are lost from its ASR
transcript, 11 go the other way. Where those losses surface: learned gate 48, retrieval 23,
retrieval low-confidence 21, decomposition 7, selector threshold 5, wrong section 1.

**Decision**: attack, in order, (1) the ASR front end, (2) refusal-gate discrimination, (3) sentence
selection. Per-intent evidence isolation (Rule 4) is **not** pursued: synthesis already maps each
sub-query to its own retrieval result by `query_id`, and measured contamination is 1/102 (dev),
0/98 (diag).

## E2 — Data quality of the provided HeySQuAD transcripts [FACT]

Of 4,158 validation rows, 58 transcripts share no content word with their question; most are real
ASR failures on formulas ("What is SiO?" → "what is s i owe"); ~18/104 checked match a *different*
question (source-dataset pairing errors, <1% of rows). Not a material factor.

Provided-transcript WER vs the typed question the speaker read (`eval/asr_wer.py`):
dev **29.3%** (content-word error 25.4%), diag **25.1%** (20.7%).

## E3 — Refusal gate + reader/cross-encoder agreement feature [OFFLINE → DEV/DIAG]

Hypothesis: the gate refuses correct answers because its score features fall when the query is
garbled; agreement between two independent models (the SQuAD2 reader's best span lies inside the
cross-encoder's chosen sentence) is a garble-robust correctness signal.

Offline evidence (HeySQuAD dev, answerable): P(correct | agree) = 97.0% clean / 88.1% ASR vs
P(correct | disagree) = 48.7% / 49.2%. Unanswerables also "agree" 37–38% of the time, so it is a
feature, not a rule.

Gate refit (same pre-registered rule: p_unanswerable = 0.3, c = 1; fit on HeySQuAD dev, applied
unchanged to HeySQuAD diag = CHECK), paired bootstrap on the 438 CHECK items:

| Gate | Answer recall | Wrong on answerable | Answered unanswerable | Utility |
|---|---|---|---|---|
| current (6 features) | 68.0% | 9.1% | 27.9% | +0.329 |
| + agree | 68.9% (+0.9 [−2.3, +4.1]) | **5.9% (−3.2 [−5.3, −1.4]) \*** | 26.3% (−1.6 [−5.7, +2.3]) | **+0.362** |

End-to-end (production path, text scenarios), vs current gate:
dev answer +5.9 [−1.0, +13.0], diag +2.0 [0.0, +5.1]; claim precision, abstention, exact
decomposition unchanged.

End-to-end, HeySQuAD diag (the gate's held-out CHECK set), current gate → + agree, paired:

| | Answer recall | Citation recall | Claim precision | Correct abstention |
|---|---|---|---|---|
| clean (219 Q) | −0.5 [−5.0, +4.1] | −3.2 [−7.8, +1.4] | −0.2 [−0.7, 0.0] | +4.6 [−0.9, +10.0] |
| provided ASR (219 Q) | +3.2 [−1.4, +7.8] | +0.9 [−4.1, +5.9] | +0.3 [−0.9, +1.9] | −0.9 [−6.8, +5.0] |

**Result: no significant end-to-end change on any metric; direction mixed (clean citation −3.2).**
The offline wrong-answer reduction does not carry through end-to-end. Likely reason: the offline
sweep disables the two other gates (retrieval low-confidence logit −5, selector threshold −4) that
production applies *before* the learned gate, so the offline population differs from the production
one.

**Decision: NOT ADOPTED as default** (fails Rule 14: not a statistically meaningful or clearly
Pareto improvement end-to-end). Code path kept (`SquadReader.answer`, the `agree` feature is computed
only if a gate JSON lists it); the shipped gate JSON is unchanged. Revisit after the ASR decision,
since a new ASR front end changes the gate's input distribution and would require a refit anyway.

## E4 — ASR front end: Whisper (faster-whisper, CTranslate2) [DEV/DIAG]

Setup: HeySQuAD validation audio for the dev and diag questions only (996 clips, 195 MB, dataset
revision `2a9aff60`; **no test2 audio downloaded**). faster-whisper 1.2.1 / ctranslate2 4.8.2 in an
isolated folder (not the production environment). Models `Systran/faster-whisper-{tiny,base,small}.en`,
pinned revisions, SHA-256 verified against the Hub's LFS hashes. Decoding: beam 5, English, no
previous-text conditioning, no VAD.

| ASR | Dev WER | Dev content-word err. | Dev exact | Diag WER |
|---|---|---|---|---|
| provided HeySQuAD transcript | 29.3% | 25.4% | 105/558 | 25.1% |
| Whisper tiny.en | 11.7% | 11.0% | 284/558 | 10.0% |
| Whisper base.en | **9.4%** | **8.8%** | 338/558 | **8.3%** |
| Whisper small.en | **7.5%** | **7.2%** | 371/558 | **5.8%** |

Timing so far is **not valid** (contended machine); clean re-timing planned.

End-to-end, HeySQuAD dev (279 paired questions), unchanged RAG pipeline, only the transcript source
differs (`eval/heysquad.py --retranscribe`):

| Input | Answer recall | Citation recall | Claim precision | Abstention | Over-split |
|---|---|---|---|---|---|
| typed question (clean) | 70.6% | 76.0% | 72.7% | 70.3% | 0.4% |
| provided HeySQuAD ASR | 37.3% | 39.8% | 71.8% | 79.2% | 10.9% |
| **Whisper tiny.en** | **58.8%** | **63.1%** | 73.3% | 75.6% | 4.3% |

Whisper tiny.en vs provided ASR: answer **+21.5 [+15.4, +28.0] \***, citation **+23.3 [+16.8, +30.1] \***,
claim precision +1.5 [−0.8, +4.5], abstention −3.6 [−9.3, +1.8]. Gap to typed text: −33.3 → −11.8.

| **Whisper base.en** | **63.4%** | **67.7%** | 70.6% | 73.1% | 3.6% |

base.en vs tiny.en: answer **+4.7 [+1.1, +8.2] \***, citation **+4.7 \***, claim precision −0.6, abstention
−2.5 [−5.0, 0.0]. base.en vs provided ASR: answer **+26.2 [+20.1, +32.6] \***, citation **+28.0 \***,
claim precision +1.0, **abstention −6.1 [−11.8, −0.7] \***.

Abstention cost, explained: garbled transcripts made many adversarial unanswerable questions fail
retrieval outright, so they were refused *by accident*. With accurate transcripts those questions
look as answerable as their typed originals and abstention moves toward the typed-text level
(70.3%), still above it. Utility at the pre-registered weights (p_una=0.3, c=1):
+0.7×0.262 = +0.183 (answers) vs −0.3×0.061 = −0.018 (abstention) → strongly net positive.

| **Whisper small.en** | **65.6%** | **70.3%** | 71.2% | 72.0% | 2.0% |

small.en vs base.en: answer +2.2 (n.s.), citation +2.5 [−0.4, +5.4], claim precision +0.6,
abstention −1.1 [−3.2, +1.1]. Gap to typed text: **−5.0 [−8.6, −1.4]** (provided ASR: −33.3).
Diminishing returns: tiny→base significant, base→small not. The model choice is therefore decided by
streaming latency on this hardware (E5).

### Clean ASR timing (40 clips, mean 6.4 s audio, machine otherwise idle, int8)

| ASR | GPU MX250 ms/clip (RTF) | CPU i5-10210U 6 thr ms/clip (RTF) |
|---|---|---|
| tiny.en | 306 (0.047) | 855 (0.133) |
| base.en | 475 (0.074) | 1,405 (0.218) |
| small.en | 1,193 (0.185) | 4,753 (0.738) |

Whisper pads every input to 30 s, so one re-decode costs about one full-clip decode regardless of
buffer length. Streaming with a 1 s re-decode step keeps up only if a decode takes < 1 s: tiny/base
on the GPU yes; small.en on the GPU no; on the CPU only tiny.en (borderline). Peak memory not
measured (the Windows API probe in `transcribe.py` returned 0). Model sizes 75 / 141 / 464 MB.
Hardware correction: the CPU is an i5-10210U, 4 cores / 8 threads.

## E5 — Real streaming ASR: LocalAgreement-2, base.en on GPU [DEV]

`streaming_rag/asr/local_agreement.py` (unit-tested) and the simulator: re-decode the growing buffer
every 1 s, commit only words two consecutive decodes agree on (append-only, as the controller
requires), flush the final decode when the audio ends; decodes are serial in audio time.

Finding 1 — early junk commits. On a 1 s prefix Whisper can hear "What" as "You"; the next decode
agrees, so it is committed; the final decode disagrees and, since committed words cannot be
retracted, the corrected question is appended ("You is the universal band … free-to-air What is the
universal band … on?"). 469/558 streamed transcripts equal the offline ones; WER 19.5% but
content-word error 8.8% (same as offline — the right words are all present, some twice).

Finding 2 — the long run's wall-clock tail is contaminated: two 6 s clips "took" 48 min and 2.7 h
(machine paused/slept mid-run). The p50 post-speech delay (713 ms) matches the clean per-decode
cost; the p95 from that run is **not reported**. For the quality replay only, gaps > 2 s between
consecutive commits (115/558 scenarios) were capped to 2 s (legitimate gaps ≤ ~1.5 s); text
unchanged. Keep-awake enabled for later timed runs; per-decode durations are now logged.

End-to-end, HeySQuAD dev, streamed vs offline base.en transcript:

| | Offline base.en | Streamed base.en | Paired Δ |
|---|---|---|---|
| Answer recall | 63.4% | 60.6% | **−2.9 [−5.4, −0.7] \*** |
| Citation recall | 67.7% | 64.2% | **−3.6 [−6.1, −1.1] \*** |
| Claim precision | 70.0% | 70.2% | +0.2 [−0.4, +0.8] |
| Abstention | 73.1% | 74.6% | +1.4 [+0.4, +2.9] \* |

Streaming the ASR costs ~3 points against the idealised offline transcript; still **+23.3** over the
dataset-provided transcripts (37.3%).

## E6 — Streaming-ASR tail: diagnosis and fix [DEV, 100 clips, keep-awake on, per-decode timing]

Diagnosis from per-decode logs: almost every 12–17 s decode is the **first decode on a ~1 s
fragment** (e.g. `[16969, 535, 383, …]` ms) — Whisper hallucinating/looping on too-short audio up to
its 448-token limit; occasional runaway decodes also occur mid-stream. Not GPU stalls, not warm-up
(warm-up decode is excluded), not CPU fallback. No samples were removed.

| Setting (base.en, GPU int8, 1 s step) | Identical to offline transcript | Decode p50 / p95 / max | ASR post-speech p50 / p95 / max |
|---|---|---|---|
| first decode at 1 s | 83/100 | 531 / 3,830 / 16,969 ms | 1,103 / 12,203 / — ms |
| first decode at 2 s | 91/100 | 547 / 1,340 / 15,008 ms | 895 / 9,220 / — ms |
| **first decode at 2 s + max_new_tokens=64** | **91/100** | **548 / 1,191 / 3,846 ms** | **768 / 2,711 / 7,177 ms** |

A spoken question is far below 64 tokens, so the cap cannot truncate a legitimate transcript; it only
bounds runaway generation. **Adopted: 2 s minimum buffer + 64-token cap** (end-to-end answer quality
with these settings to be re-measured on the full dev set).

---

# Latency pass (29 Sep 2026): toward p99 < 5 ms where meaningful

## L0 — Measurement fix: the engine clock could not resolve 5 ms [FACT]

`time.monotonic()` on this Windows machine is `GetTickCount64()` with **16 ms resolution** (measured);
`time.perf_counter()` is `QueryPerformanceCounter()`, ~0.002 ms. Every latency the engine reported
before this pass (e.g. "post-speech p50 0 ms / p95 16–47 ms") was quantised to 16 ms ticks. All
timing in `streaming_rag/` and `eval/` now uses `perf_counter` (both clocks are monotonic; interval
measurements only). Earlier latency numbers in `final_report.docx` are superseded by this pass.

## L1 — Instrumentation

- Engine: `output_emitted` now carries `timings` = post-speech breakdown, perf_counter, from the
  `utterance_end` envelope to the turn output: `final_decision_ms` (controller pass on the final
  chunk), `retrieval_wait_ms` (waiting for sub-query retrieval + speculative claim selection still in
  flight), `synthesis_ms` (synthesize/refine), `answer_telemetry_ms`, `post_speech_ms` (total).
  Existing schema-valid event, extra field only (G6 schema unchanged).
- `eval/latency_profile.py`: replays scenarios through the unmodified engine at real time (1×);
  per-stage timers installed only in the profiler process (BM25, e5 encode, dense top-k, RRF,
  cross-encoder by caller: rerank / claim / gate, reader, claim selection, gate, grounding,
  prefetch); every call counted; one warm-up scenario excluded and reported as cold start.

Boundaries (never summed into one number):
A ASR — separate (E4–E6). B controller — `controller.decision_ms` (every chunk).
C retrieval — `retrieval.subquery_ms` (one sub-query, concurrent with speech when speculative).
D reranking — `ce.rerank_ms`. E synthesis — `synthesis.*` (claim selection, gate) and
`engine.synthesis_ms` (answer assembly at utterance end). F total RAG per sub-query —
retrieval + `synthesis.prefetch_ms`. G post-speech — `engine.post_speech_ms`.

## L2 — Baseline profile [DEV text, 60 scenarios, real time, 28 Sep]

| Stage | n | p50 | p95 | p99 | max (ms) |
|---|---|---|---|---|---|
| controller decision (per chunk) | 330 | 0.47 | 1.41 | 1.92 | 5.53 |
| answer assembly at utterance end | 90 | 1.25 | 2.24 | 2.68 | 2.75 |
| final controller pass | 90 | 0.51 | 1.01 | 2.42 | 5.81 |
| **post-speech (user-perceived)** | 90 | 2.30 | 181.6 | **522.0** | 2,135.7 |
| ↳ waiting for in-flight work | 90 | 0.22 | 179.9 | 519.8 | 2,128.8 |
| retrieval per sub-query | 212 | 290.9 | 424.3 | 607.0 | 1,942.3 |
| ↳ cross-encoder rerank (5 passages) | 212 | 245.5 | 367.4 | 514.9 | 1,446.8 |
| ↳ e5 query encode | 212 | 19.2 | 42.3 | 50.3 | 198.6 |
| ↳ BM25 | 212 | 3.8 | 8.5 | 12.1 | 18.2 |
| speculative claim selection + gate | 212 | 468.8 | 710.0 | 955.0 | 1,924.0 |

Diagnosis: the post-speech tail is work that is still running when the user stops (the last clause's
retrieval + claim selection + refusal gate). The model calls are compute-bound on this 15 W CPU: a
5-passage cross-encoder rerank costs 234 ms alone with 4 ORT threads (537 ms single-threaded);
concurrency roughly doubles per-call latency (451 ms with 3 concurrent calls). The controller's rare
5–6 ms maxima coincide with worker-thread inference (GIL), not its own work.

## L3 — int8 dynamic quantization (speed only) [NOT ADOPTED]

ORT `quantize_dynamic` (QInt8) of e5 / cross-encoder / reader (4× smaller files). Speed-up on this CPU
(no VNNI): cross-encoder 192→149 ms (1.29×), reader 228→172 ms (1.33×), e5 16.4→12.1 ms (1.35×).
Modest, and would require a full quality re-evaluation; deferred.

## L4 — Run the refusal gate's reader concurrently with claim selection

Hypothesis: the gate's SQuAD2-reader call (~230 ms) does not depend on which sentence is selected,
so it can run concurrently with cross-encoder claim selection (~200 ms) instead of after it.
Change: `session/synthesis.py` submits `reader.answer` to a 2-worker pool before `_ce_claim`, the
gate collects the result. Same computations, only scheduling differs.
Quality: dev text answers **bit-identical** (0/60 scenarios differ in any answer version; answer,
citation, claim precision, abstention, decomposition diffs all 0.0).
Latency: an unpaired before/after was confounded (untouched stages were ~2× faster on the second day),
so an A/B/A comparison (serial / parallel / serial, back-to-back, same machine state) was run with the
profiler's `--serial-gate` control.

A/B/A result (60 dev scenarios, real time, run back-to-back on 29 Sep, same machine state):

| | Serial A1 | **Parallel B** | Serial A2 |
|---|---|---|---|
| prefetch (claim selection + gate) p50 / p95 / p99 ms | 213.7 / 546.9 / 692.8 | **161.9 / 301.8 / 417.3** | 224.9 / 510.1 / 767.4 |
| post-speech p50 / p95 / p99 / max ms | 1.85 / 66.3 / 207.2 / 342.0 | 2.40 / 5.32 / 149.9 / 162.2 | 1.94 / 4.26 / 136.9 / 179.5 |
| controller decision p99 / max ms | 2.18 / 2.94 | 1.85 / 2.46 | 1.95 / 4.56 |
| answer assembly p99 / max ms | 2.84 / 3.81 | 3.63 / 3.86 | 3.24 / 3.80 |

**Decision: ADOPTED** (quality-neutral, bit-identical answers). It consistently cuts per-sub-query
claim-selection time (~−25% p50, ~−40% p95 vs both serial runs). Its effect on **post-speech p99 is
within run-to-run noise** (serial 207 / 137 ms vs parallel 150 ms; with n = 90 turns, p99 is
essentially the worst single turn) — no post-speech improvement is claimed from this change.

## L5 — GPU opt-in for the RAG models (isolated onnxruntime-gpu, MX250) [NOT ADOPTED as default]

Same protocol, parallel gate reader:

| Stage (ms) | CPU p50 / p99 | GPU p50 / p99 |
|---|---|---|
| cross-encoder rerank | 110 / 297 | 36 / 275 |
| claim selection (CE) | 139 / 376 | 52 / 448 |
| prefetch | 162 / 417 | 75 / 666 |
| post-speech | 2.40 / 149.9 (max 162) | 2.19 / 114.0 (max 481) |

3–4× faster in the typical case, but a heavier tail: concurrent sub-queries and the gate reader queue
on one 2 GB GPU. Not a clean improvement; CPU stays the verified default, GPU remains opt-in
(`STREAMING_RAG_ORT_PROVIDER=cuda`). In the audio-in path the GPU is also needed by Whisper.

## L6 — Where "< 5 ms p99" is and is not meaningful (29 Sep, CPU default, real time, dev text)

| Component (exact boundary) | p99 across 3 CPU runs | < 5 ms? |
|---|---|---|
| Controller decision per transcript chunk (`on_chunk`, incl. decomposition + stability) | 1.85–2.18 ms (max 2.5–4.6) | **yes** |
| Final controller pass at utterance end | 2.1–2.4 ms | **yes** |
| Answer assembly at utterance end when claims were selected during speech (`synthesize`/`refine`) | 2.8–3.6 ms (max 3.9) | **yes** |
| Post-speech latency (utterance_end → answer emitted), all turns | 137–207 ms | **no** |
| Retrieval per sub-query / claim selection / gate | 100s of ms | no — compute-bound model inference on a 15 W CPU |

Post-speech p50 is ~2 ms and p95 ~4–5 ms in 2 of 3 runs: for most turns everything finishes while the
user is still talking. The p99 is the turn whose final words add new content, which must then be
retrieved, reranked, selected and gated after they are heard (≈ 0.3–0.7 s of model inference on this
CPU). No honest boundary makes that 5 ms; the remaining bottleneck is cross-encoder + reader inference
on the final clause.


---

# New workstation pass (29 Sep 2026): RTX 6000 Ada — GPU-only neural inference

Everything below is **NEW MACHINE** unless labelled OLD MACHINE. Labels as before: **DEV** (used to
decide), **DIAG** (held-out confirmation), **PRELIMINARY**. Sealed test2 was **not** run, read for
tuning, or downloaded (no test2 audio exists on this machine).

## W0 — Machine, environment, GPU verification [FACT]

| | OLD MACHINE | NEW MACHINE |
|---|---|---|
| CPU | Intel i5-10210U, 4 C / 8 T, 15 W | Intel Xeon w9-3475X, 36 C / 72 T |
| RAM | ~8 GB (often < 1.6 GB free) | 255 GB (236 GB free) |
| GPU | NVIDIA MX250, 2 GB | NVIDIA RTX 6000 Ada Generation, 48 GB, compute capability 8.9 |
| Driver / CUDA | — | 573.44 / CUDA 12.8 (WDDM) |
| OS | Windows | Windows 11 Pro for Workstations 10.0.26200 |
| Python | LibreOffice-bundled 3.11 | 3.11.9 (python.org) |

The machine had no Python, git or CUDA user-space libraries; installed with the user's approval.
Two environments: `.venv` = `requirements.lock` exactly (CPU `onnxruntime 1.20.1`; kept only for the
CPU/Docker reproducibility path) and `.venv-gpu` = the same pins with `onnxruntime-gpu 1.20.0` (the
version the old machine used), `nvidia-cudnn-cu12 9.26`, `nvidia-cublas-cu12 12.9`, CUDA runtime 12.9,
`faster-whisper 1.2.1`, `ctranslate2 4.8.2`, `onnx 1.23` and (at the user's request) `torch
2.14.0+cu126` — torch is used only by ONNX Runtime's offline graph optimiser, never at run time.
Models re-fetched at the pinned revisions and SHA-256-verified (e5 / MiniLM-L6 CE / Whisper
tiny,base,small.en); the migrated SQuAD2 reader ONNX matches its recorded hash `6e9e62e6…`.
HeySQuAD dev+diag audio (996 clips, dataset revision `2a9aff60`) re-downloaded by `fetch_audio.py`,
which refuses test2 ids.

Pitfall found and fixed: `faster-whisper` pulls in CPU `onnxruntime`, which shares the `onnxruntime/`
package directory with `onnxruntime-gpu` and silently replaces it — both were removed and only
`onnxruntime-gpu` reinstalled.

**GPU verification (controlled test, product code path, defaults only)** —
`handoff/gpu_tools/verify_product_gpu.py`, output `handoff/newmachine/gpu_verify_product.json`:
every session built by `neural.get_e5_encoder / get_cross_encoder / get_reader` bound
`CUDAExecutionProvider`; Whisper (`StreamingWhisper(device="auto")`) loaded on `cuda`; GPU memory
0.8 → 2.6 GB during the test; answers correct. ONNX Runtime's per-operator profiler shows where
kernel time goes (5 × 192 tokens): e5 100 % CUDA; cross-encoder and reader see below (W5).

**GPU made mandatory** (`streaming_rag/gpu.py`, used by `retrieval/neural.py` and `asr/whisper.py`):
`STREAMING_RAG_ORT_PROVIDER` / `STREAMING_RAG_ASR_DEVICE` default `auto` = CUDA whenever an NVIDIA
GPU is present; if the installed runtime cannot use it, `GpuUnavailable` is raised instead of running
neural inference on the CPU. A session that does not bind CUDA first is refused. `cpu` is only an
explicit opt-out (the CPU Docker image has no GPU and is unaffected). Every eval output records
`execution_providers`. Tests: `tests/test_gpu.py`.

## W1 — Baseline reproduction, OLD vs NEW [DEV]

Same code, same configuration, same data.

Quality is hardware-independent for the RAG models: dev text answers **bit-identical** to the old
machine's saved run on all 60 scenarios (new CPU and new GPU alike): answer 62.7 %, citation 67.6 %,
claim precision 84.5 %, abstention 5/5, exact decomposition 100 %, G2–G6 100 %.

**Correction to earlier results (old-hardware timeouts):** the old run of HeySQuAD dev clean
(`r8_final_heydev_clean`, 70.6 %) differs from the new machine (71.3 %) on exactly 2 of 558
scenarios, and both are *timeouts on the old laptop* ("internal error or timeout prevented a
grounded answer"); the new machine answers both. The ±0.7 point "non-determinism" noted in E0 was
this. Timeout answers appear in old result files `r8_final_heydev_clean` (2), `r8_final_heydev_asr`
(2), `r8_final_heydiag_asr` (1), `r3_nogate_heydev_clean` (1) and `SEALED_heysquad_test2_asr` (1 —
read only, not re-run); the final sealed files `SEALED_final_*` contain none.

Timings, OLD vs NEW (same protocol as the old measurement in each row):

| Component | OLD MACHINE | NEW MACHINE | Speed-up |
|---|---|---|---|
| Whisper tiny.en, GPU int8, 40 clips, ms/clip | 306 | 192 | 1.6× |
| Whisper base.en, GPU int8 | 475 | 210 | 2.3× |
| Whisper small.en, GPU int8 | 1,193 | 281 | 4.2× |
| Whisper base.en, CPU int8, 6 threads | 1,405 | 943 | 1.5× |
| Cross-encoder rerank, 5 passages, p50 (real-time pipeline) | 126 ms (CPU) | 89–147 ms (CPU) / **14 ms (GPU)** | 9× (GPU vs old) |
| Claim selection p50 | 170 ms (CPU) | 75–144 (CPU) / **24 ms (GPU)** | 7× |
| Retrieval per sub-query p50 | 150 ms (CPU) | 115–181 (CPU) / **51 ms (GPU)** | 3× |
| Test suite (190 tests) | ~65 s | 14 s | 4.6× |
| HeySQuAD dev clean quality run (558 scen.) wall | 25,419 s (contended) | 461 s | indicative only |
| Dev text quality run (60 scen., time-scale 8) wall | 122.8 s | 86 s | bounded by replay time |

CPU run-to-run variance on the Xeon is large (identical code, idle machine: rerank p50 89 vs 147 ms in
two runs) — another reason all neural work now runs on the GPU.

## W2 — Streaming ASR: 1 s vs 2 s minimum buffer, and the full-dev rerun

The 100-clip comparison was completed on the OLD MACHINE before the migration (recomputed here from
the raw per-decode logs, nearest-rank percentiles, nothing excluded):

| OLD MACHINE, 100 dev clips, base.en MX250 int8 | identical to offline | decode p50 / p95 / p99 / max | ASR post-speech p50 / p95 / p99 / max |
|---|---|---|---|
| 1 s first decode | 83/100 | 531 / 3,964 / 11,298 / 16,969 ms | 1,101 / 12,203 / 15,736 / 18,260 ms |
| 2 s first decode | 91/100 | 547 / 1,420 / 6,572 / 15,008 | 889 / 9,220 / 14,949 / 15,101 |
| **2 s + 64 tokens (adopted)** | 91/100 | 548 / 1,210 / 2,631 / 3,846 | 767 / 2,711 / 4,121 / 7,177 |

The full 558-clip run with the adopted settings (stopped at 300/558 on the old machine, no output)
was re-run **on the NEW MACHINE** (RTX 6000 Ada, base.en int8), 737 s wall:

| NEW MACHINE, 2 s + 64 tokens | clips | identical to offline | decodes n | decode p50 / p95 / p99 / max | decodes > 1 s / > 3 s | ASR post-speech p50 / p95 / p99 / max |
|---|---|---|---|---|---|---|
| all dev | 558 | 499 (89.4 %) | 3,032 | 205 / 432 / 1,279 / 2,364 ms | 54 / 0 | 236 / 479 / 1,240 / 1,713 ms |
| same 100 clips as above | 100 | 93 | — | 204 / 415 / 1,019 / 2,364 | 6 / 0 | 235 / 501 / 1,445 / 1,508 |

The remaining > 1 s decodes are runaway generations stopped by the 64-token cap (~1 s of beam-5
decoding at the cap), not stalls.

## W3 — Whisper output is NOT hardware-independent [FACT]

Offline base.en on the RTX 6000 Ada vs the old MX250, identical model revision, settings and audio:
**70 of 996 transcripts differ** (CTranslate2's int8 GPU kernels differ between Pascal sm_61 and Ada
sm_89). Some differences remove old errors (a "Okay. Okay. Okay…" hallucination loop is gone),
others add new ones. Aggregate WER is equal or slightly better:

| base.en offline | dev WER | dev content-word err. | dev exact | diag WER | diag content-word err. |
|---|---|---|---|---|---|
| OLD MACHINE (MX250) | 9.36 % | 8.83 % | 338/558 | 8.33 % | 8.18 % |
| NEW MACHINE (Ada) | 9.23 % | 8.83 % | 340/558 | 7.58 % | 8.22 % |

Consequence: Whisper-based answer-quality numbers (E4–E6) must be re-measured on this machine, not
carried over (context.md had assumed hardware independence).

## W4 - CUDA graphs: fusion, static shapes, warm-up [DEV text + HeySQuAD dev typed; NEW MACHINE]

Why: on the RTX 6000 Ada an isolated cross-encoder call is ~2 ms, but in the pipeline the same call took
12-25 ms and the profile showed a tail. ONNX Runtime's per-operator profiler on the pinned graphs (5 x 192
tokens) showed why: e5 ran 100 % on CUDA, but the cross-encoder and the SQuAD2 reader put ~20 % of their
kernel time on the **CPU** execution provider - the exports compute sqrt(head_size) and every head
reshape target from tensor shapes on each call (int64 Shape/Gather/Concat chains, which ORT keeps on the
CPU and which force host<->device syncs).

| Graph | nodes (pinned export) | after exact static-shape rewrite | after ORT fusion | CPU-EP kernel share |
|---|---|---|---|---|
| e5-small-v2 | 1,243 | (not needed) | 88 | 0 % |
| MiniLM cross-encoder | 785 | 413 | 198 | 18 % -> ~3 % |
| SQuAD2 reader | 1,258 | 514 | 357 | 21 % -> ~2 % |

* `streaming_rag/retrieval/graph_static.py` replaces each shape-derived value by a constant read by
  running the model at two shapes (the head-split reshape becomes `[0, 0, 12, 32]`), then re-runs the
  original and the rewritten graph at four shapes with ORT's own optimisations **off** and raises unless
  the outputs are **bit-identical**. `python -m streaming_rag.retrieval.neural --fuse` builds
  `.cache/streaming_rag/models/fused/{e5,ce,reader}.onnx` (+ `manifest.json` with hashes). If they are
  missing on a CUDA machine the code runs the original graph **on the GPU** with a warning (never on the
  CPU); `STREAMING_RAG_ORT_FUSED=1` makes them mandatory, `0` disables them.
* Warm-up: every session runs representative shapes at load and `GroundedSynthesizer.setup()` preloads
  the cross-encoder and the reader at engine setup. Before this, the first retrieving turn paid a
  0.5-0.8 s one-off (lazy reader load + first CUDA call); worst prefetch 818 ms -> 81 ms.
* Quality: fusion alone -> **bit-identical answers on 618 scenarios** (60 dev text + 558 HeySQuAD dev
  typed). Static shapes + fusion -> bit-identical on the 60 dev text scenarios; on HeySQuAD dev typed
  **1 of 558 items changed** (an answer became a refusal): its gate probability is 0.40046 / 0.40014 /
  0.39942 under the original / fused / static graphs against a 0.40 threshold, i.e. float rounding at the
  decision boundary. Paired change in answer recall -0.4 [-1.1, +0.0] (n.s.). Accepted, documented.
* Dev post-speech p99 across the steps (real time, GPU, nothing excluded): original graphs 94.7 ms ->
  fused 86.8 -> + warm-up 62.6 -> + reader preload 59.9 ms; e5 query encode p50 16.4 -> 10.0 ms.

## W5 - Speculative final pass [DEV / DIAG; NEW MACHINE]

Finding: in the text scenarios the last chunk arrives 0.7-2.7 s before `utterance_end`, so nearly every
turn is finished while the user is still "talking" (post-speech p50 2.2 ms, p95 3.0 ms). The whole p99 tail
was the few turns whose clause the controller holds back until its **final pass** at `utterance_end`
(e.g. "What is DECnet" -> WAIT, then "DECnet" at the end): their retrieval, claim selection and refusal gate
ran after speech (57 of the 60 ms).

Change: `RetrievalController.preview_final()` computes, on a snapshot that is rolled back, what the final
pass *would* issue; after every chunk the engine starts retrieval + claim selection for those texts and
reuses a result **only for a sub-query with the identical text** (retrieval and claim selection depend
only on that text) - answers unchanged by construction, verified bit-identical on 618 scenarios.
Speculative sub-queries carry their own id namespace (`...~specN`).

| Post-speech (real time, GPU) | n | p50 | p95 | p99 | max (ms) |
|---|---|---|---|---|---|
| DEV, before (fused + warm-up + preload) | 90 | 2.2 | 3.0 | 59.9 | 87.0 |
| DEV, with speculation, run 1 / run 2 | 90 | 2.23 / 2.21 | 3.89 / 3.12 | 4.75 / 3.54 | 5.06 / 3.85 |
| DEV, speculation switched off (ablation) | 90 | 2.47 | 3.71 | 53.7 | 72.4 |
| DIAG, run 1 / run 2 | 90 | 2.34 / 2.61 | 3.51 / 3.90 | 4.81 / 4.26 | 9.18 / 4.64 |

The DIAG set contains no held-back final clauses, so it confirms *no regression* (its p99 was already
4.5 ms without speculation), not the improvement. DIAG run 1's 9.18 ms maximum is a 7.5 ms final
controller pass with nothing in flight (CPU scheduling), not model work. Bugs found in the first
version: a sentinel that was deep-copied in the rollback (caught by the new tests before any
measurement) and speculative ids that could collide with a later real sub-query so its prefetch
overwrote the speculative claim (found in the audio-path latency run; +36-42 ms on ~2/100 turns,
answers unaffected; regression test added).

## W6 - Tested and rejected / neutral [NEW MACHINE]

| Candidate | Result |
|---|---|
| 3 ORT sessions per model (own CUDA streams) | micro-benchmark with 4 concurrent callers: p50 16 -> 8 ms; in the real pipeline (~2-way concurrency) tails got **worse** (rerank p99 41-52 ms vs 26 ms with one session) -> default 1, kept as `STREAMING_RAG_ORT_SESSIONS` |
| CUDA EP `do_copy_in_default_stream=False`, `kSameAsRequested` arenas, pre-grown arenas | no effect beyond noise |
| ORT optimiser levels 1 / 2 (needed torch) | no attention fusion; the shape chains survive |
| GPU keep-warm tick (1 / 5 / 20 ms) | a 1 ms tick held 2,505 MHz (P2) but calls after idle were not faster (3.67 / 8.17 ms vs 3.76 / 7.24 ms) - GPU clocks are not the cost |
| Serialising the gate reader on the GPU | per-call latency down, whole chain unchanged (71.5 vs 74.2 ms) |
| Sequence-length bucketing / fixed shapes | no effect (2.2-2.5 ms p50) |

## W7 - CPU wake-up latency: the remaining tail is a power-plan effect [NEW MACHINE, PRELIMINARY]

The ~80 ms post-speech chain of a sub-query is ~15 ms of model compute; the rest is scheduling. On the
Windows **Balanced** plan (Ultimate Performance and High performance are installed, not selected) a pure
CPU task takes 6.0 ms hot but 23.7 ms after a 50 ms idle (3.8x); a cross-encoder call through ORT CUDA
takes 7.0 ms after a 50 ms sleep versus 3.9 ms when the core was kept busy (2.1 ms back-to-back); and
turning ORT's thread spinning off doubles even back-to-back calls (2.1 -> 4.4 ms). Changing the power
plan is a system setting and was **not** changed; it is the first thing to try on a deployment machine.

## W8 - Speech-input path (OUTSIDE the official transcript -> RAG scope; PRELIMINARY, unpaired)

Kept as a record; not used for the submission's headline numbers. Whisper base.en, int8, RTX 6000 Ada.

| Transcript source (HeySQuAD, answer / citation / claim precision / abstention) | dev (279 Q) | diag (219 Q) |
|---|---|---|
| typed question | 71.3 / 76.7 / 71.8 / 70.3 | 75.8 / 83.1 / 71.8 / 70.3 |
| dataset-provided ASR transcript | 37.6 / 40.1 / 63.9 / 79.2 | 56.6 / 60.3 / 66.0 / 72.6 |
| Whisper base.en, offline | 63.1 / 67.4 / 70.0 / 73.5 | 73.5 / 80.8 / 72.2 / 71.2 |
| Whisper base.en, streamed (LocalAgreement-2, 2 s first decode, 64-token cap) | 63.1 / 67.4 / 70.8 / 74.2 | 71.2 / 78.1 / 72.2 / 72.6 |

Streaming no longer costs answer recall on dev (old machine: -2.9 pts). Hypothesis speculation (the
ASR's uncommitted words used to speculate the last clause; `transcript_hypothesis` events) on 100 dev
clips: answers **identical** with and without it (68 / 100 both); ASR-path post-speech p50 81.7 -> 2.9 ms
but p99 106.7 -> 91.1 ms - about half the turns finish in < 5 ms, the rest are turns where Whisper's final
decode changes the last words. Whisper output itself differs between GPUs (70 / 996 base.en transcripts
differ from the old machine), so older Whisper numbers were re-measured.

---

# Final audit (30 Sep 2026) - Theme 4 scope: streaming transcript -> RAG -> grounded, cited answer

Scope decision for this pass: audio / ASR / Whisper is ignored (kept as the record in W8 only). All neural
inference on the RTX 6000 Ada, CUDA mandatory. Sealed test2 was not read, run or tuned on. Method:
(1) the requirement list of the guide / project context / Task 4 turned into a checklist; (2) mechanical
audits of the hard rules; (3) behavioural probing of the real GPU pipeline with awkward inputs, because
unit tests only cover what someone thought of; (4) fix, regression-test, re-measure DEV and DIAG
answer-by-answer against the run before the fix (`eval/results/final_audit/` and `handoff/newmachine/`).

## FA1 - Hard-rule audits (mechanical) [FACT]

| Rule | Check | Result |
|---|---|---|
| No hardcoding | grep `streaming_rag/` for Doc ids, scenario ids, `ground_truth`; imports of `eval` / `tests` / `fixtures` / `data` | none in product code (Doc ids only in docstrings and in `mocks.py`, the test doubles); a static test already forbids imports |
| Corpus isolation | grep for network APIs in product code | none (the LLM SDKs are only imported when `LLM_PROVIDER` selects them); default `fake` |
| Session-bound state | disk writes in product code; session wipe | writes: telemetry log, `--json` output, corpus-embedding and model-graph caches only; `session_end` wipes the session (isolation re-verified below) |
| Secrets | scoped pattern scan (keys, tokens, private keys) over source, eval scripts, tests, docs, config | 0 hits; `.env` absent and git-ignored; a test checks every env var read is declared in `run.yaml` |
| Input format | the literal guide section 6.6 shape (chunk payloads **without** `session_id`) through the CLI | works; `turn_result` keys as specified |

## FA2 - Behavioural probing: what the tests had not covered [DEV pipeline, GPU]

A battery of awkward inputs through the real engine (`handoff/final_audit/` scratch probes). Correct as
found: out-of-corpus and "ignore the corpus and answer from your own knowledge" questions both abstain
(nothing answered from memory); duplicate chunks are harmless; a 5-intent utterance decomposes into 5; a
29-chunk utterance repeating the same 3 questions 6 times yields exactly 3 sub-queries; two interleaved
sessions do not see each other's answers; non-ASCII / emoji input does not crash and mixed-language
questions split per language. Defects found:

1. **`turn_result.sub_queries` listed superseded partial fragments** (external output contract). The
   engine cancels a provisional fragment correctly and synthesis ignores it (that is why answers and every
   trace-based gate were fine), but the reported field listed every sub-query ever issued, so a two-question
   utterance reported 3 sub-queries incl. a near-duplicate fragment - the guide's over-fragmenting pitfall as
   seen by an external reader. Fixed (`superseded` set per utterance; both the normal and the degraded
   record report only live sub-queries); regression test `tests/engine/test_output_sub_queries.py`
   (verified to fail without the fix).
2. **Presentation-only detection was both too broad and too narrow.** `_PRESENTATION_RE` fired on a single
   cue word anywhere, so with an earlier answer a genuine request such as "Summarize the causes of X" was
   treated as a reformat of the old answer (no retrieval), while real paraphrases ("Now give that in one
   short sentence") were missed - in a multi-turn probe that turn searched the corpus for junk and
   **replaced the good answer with "I could not find anything"** (pitfall 4 plus context loss). A reformat
   request with no earlier answer also searched the corpus. `_CHITCHAT_RE` matched only bare greetings
   ("Thanks, that's all for today." fell through). New general rules (`is_presentation_request`,
   `is_chit_chat`, `controller/controller.py`): presentation-only = formatting cue AND no topic words left
   after removing cue / filler words AND (reference to the earlier answer OR leading command verb); with no
   earlier answer the engine answers with a `clarification` version ("no earlier answer to reformat") and
   searches nothing; chit-chat = only greeting / thanks / closing words, at least one real one.
   `restructure` also understands "one / single". On my own probe lists: 12 / 13 reformat paraphrases
   recognised (the 13th, a bare "tl;dr", was then added) and 0 / 15 genuine questions mis-flagged.
   Verification on the evaluation splits: DEV answers identical to before the change (60 / 60 scenarios);
   DIAG identical except one scenario, where my first version regressed a standalone **"Good morning"**
   (the old rule accepted it; G2 false triggers 1/15). Fixed, and a test now asserts the new chit-chat rule
   accepts **every phrase the old rule did**, in several punctuation / case variants; DIAG then identical
   (0 differences, false triggers 0/15).
3. **Silent sentence-splitter fallback on the GPU** (hazard introduced by installing torch into the GPU
   environment at the user's request). spaCy's first import pulls in torch through `thinc`; if a CUDA
   model has already loaded its cuDNN/cuBLAS, torch's import fails (Windows 0xc0000139) and
   `_get_spacy_sentencizer()` silently switched the whole process to a regex splitter that segments
   differently ("Dr. Smith ... St. Louis" -> 4 sentences instead of 3), which would shift claim boundaries.
   The real pipeline happened to import spaCy first (corpus ingest), which is why earlier runs were
   unaffected, but any other entry point could hit it. Fixed at the choke point (`gpu.add_cuda_dlls()`
   imports spaCy before any CUDA library loads) and the fallback now logs a warning; regression test in a
   fresh interpreter (`tests/test_gpu.py::test_spacy_survives_a_running_cuda_model`).
4. Robustness of input handling that did **not** need a change: see the list above.

5. **Two sessions with the same utterance id merged their transcripts** (hard rule: session-bound state). The
   engine keyed per-utterance state by `utterance_id` alone and so did the controller's six state
   dictionaries; the guide's own example id is "u1", so two live sessions using it saw each other's words
   (session A's sub-queries came back as "catering policyHow long is the cancellation window ..."). Fixed:
   all per-utterance state is keyed by `(session_id, utterance_id)`; query ids keep their `u1.q1` form (they
   are unique within a session, as the contract says). Related: nothing ever freed controller state, so the
   text of finished turns and finished sessions stayed in process memory (ephemeral session memory must not
   outlive the session); the engine now calls `reset_utterance` when a turn ends and `reset_session` when a
   session ends, and - because the guide's input example ends with `session_end` and an **empty** payload,
   which closed nothing - a payload-less `session_end` closes the single open session. Verified: the isolation
   test failed before the fix and passes after; a test inspects the controller and finds no state left;
   DEV and DIAG answers are identical before and after (60 / 60 scenarios each).

## FA3 - Reproducibility and packaging [FACT]

* `.dockerignore` added: without it `COPY . .` would send both virtual environments, the audio folder and
  the git history into the image build context.
* `requirements-gpu.txt` (what every GPU number ran on) + Makefile targets `install-gpu`, `fuse`,
  `eval-gpu`; documents the `onnxruntime` / `onnxruntime-gpu` shared-directory pitfall.
* The CPU Docker image declares its explicit CPU opt-out (`STREAMING_RAG_ORT_PROVIDER=cpu`): the code
  makes CUDA mandatory whenever `nvidia-smi` exists, so a host run with `--gpus all` would otherwise stop
  with `GpuUnavailable` for an image that has no CUDA runtime anyway.
* A missing fused CUDA graph on a GPU machine now falls back to the original graph **on the GPU** with a
  warning (previously an error); `STREAMING_RAG_ORT_FUSED=1` makes fusion mandatory.
* `docs/architecture_brief.docx` (rewritten: it still described the LSA encoder, the heuristic reranker and a
  `controller.mode=model` switch that does not exist), `docs/telemetry_schema.md` (timings / speculation
  fields), README, final report and this log now describe the final implementation.
* G1 could not be run as specified: **Docker is not installed on this machine**. See FA6 for the CLI
  clean-room substitute and what it does not prove.

## FA4 - Quality plateau (no tuning done) [DEV, error taxonomy on the final code]

(Taxonomy at rerank pool 5. After the pool-20 change of FA5 the same analysis gives **35** misses: refusal gate
11, context selection 10, sentence selection 6, low-confidence refusal 2, chunking 2, rerank depth 1, first
stage 1, relevance gate 1, cross-intent contamination 1; wrong claims 13 -> 11; DIAG 21 misses.)

`eval.error_analysis` on the 102 dev questions: 38 misses = refusal gate withheld a correct claim 13,
a claim from a non-gold section ranked above the gold one 10, rerank depth 4, sentence selection 4,
low-confidence refusal 2, chunking 2, first stage 1, relevance gate 1, cross-intent contamination 1;
decomposition and refinement-merge losses 0. Identical to the earlier taxonomy (E1). Per template,
`late_detail` is the weakest (26 / 40 dev, 29 / 40 diagnostic answers) but for the same generic causes,
not because of refinement mechanics. The levers that could move the top two causes (a bigger reranker, a
different embedder, the reader as selector, an extra gate feature) were all tested with paired statistics
in earlier passes and none cleared the bar, so nothing was tuned in this pass; larger models would need
new downloads (not approved).

## FA5 - Rerank pool 5 -> 20, adopted [DEV decides, DIAG confirms; GPU]

Trigger: re-running retrieval quality on the final code, DEV r@1 came out 88.0 % where the earlier report
said 88.7 %. Identical on GPU with fused graphs, GPU with the original graphs and CPU, so not a regression:
the earlier DEV row had been measured with a rerank pool of **20** (817 ms / query is the CPU signature of
pool 20) and mislabelled as the shipped pool 5 (the DIAG row was genuinely pool 5 and reproduces exactly).
Pool 5 had been chosen for CPU latency; on the GPU a bigger pool is almost free, so it was tested properly.

| Retrieval only, 1,000 questions, GPU | DEV r@1 / r@5 / MRR | DIAG r@1 / r@5 / MRR | ms / query |
|---|---|---|---|
| pool 5 (shipped until now) | 88.0 / 96.4 / 0.918 | 89.4 / 96.1 / 0.927 | ~21-22 |
| pool 10 | 88.5 / 97.3 / 0.922 | 89.7 / 97.1 / 0.931 | ~24 |
| pool 20 | 88.7 / 98.3 / 0.927 | 89.9 / 97.7 / 0.935 | ~27-30 |
| pool 32 (all candidates) | 89.0 / 98.5 / 0.930 | 89.9 / 98.1 / 0.936 | ~29 |

End to end (pool 5 -> 20, paired cluster bootstrap; `eval/results/final_audit_pool5/`):

| Set | Answer | Citation | Claim precision | Abstention |
|---|---|---|---|---|
| Text DEV (102 Q) | +2.9 [0.0, +6.4] | +4.9 [+1.0, +9.1] * | +2.3 [0.0, +6.1] | 0 |
| Text DIAG (98 Q) | +2.0 [0.0, +5.3] | +1.0 [0.0, +3.2] | +1.6 [0.0, +5.0] | 0 |
| HeySQuAD DEV typed (279 + 279 Q) | +0.7 [0.0, +1.8] | +0.4 [-0.7, +1.8] | 0.0 | 0.0 |
| HeySQuAD DIAG typed (219 + 219 Q) | +0.9 [0.0, +2.3] | +0.5 [-0.9, +2.3] | 0.0 | -0.9 [-2.7, +0.9] |
| **DEV pooled (381 Q) - decides** | **+1.3 [+0.3, +2.6] \*** | +1.6 [+0.3, +3.1] \* | +0.4 [0.0, +1.1] | 0.0 |
| **DIAG pooled (317 Q) - confirms** | **+1.3 [+0.3, +2.6] \*** | +0.6 [-0.6, +1.9] | +0.3 [-0.6, +1.6] | -0.9 [-2.7, +0.9] |

Every difference is >= 0 for answer recall on all four sets; the deciding split is significant and the
held-out split confirms it to the digit; nothing is significantly worse. **Adopted** (`retrieval.rerank_pool
= 20`). Cost: ~+8 ms per isolated retrieval, but in the real-time pipeline (concurrent rerank, claim
selection and gate reader) per-sub-query retrieval roughly doubled (p50 ~40 -> 82 ms, p99 ~180 ms, max ~255 ms;
cross-encoder rerank p50 13.6 -> 49 ms) - all of it during speech; post-speech latency was unchanged (DEV p99 3.35 ms,
0 / 90 turns > 5 ms). A turn whose last content arrives within ~0.25 s of utterance_end and is NOT covered by the
speculative final pass would feel it (none in the evaluation scenarios; the audio-in path, where the last words arrive
~1 ms before the end, would). Pool 32 adds nothing at r@1 / r@5.

## FA6 - Final-configuration measurements, ablations, edge cases, reproducibility [DEV / DIAG, GPU only; sealed untouched]

Driver: `handoff/final_audit/run_final_audit.sh` (13:46-15:37, 30 Sep 2026); results `eval/results/final_audit/`;
tables `handoff/final_audit/make_tables.py`; write-up `final_report.docx` section 0 and 5.10-5.15.

* **Latency (real time, 1x, providers = CUDA only)**: post-speech p50 / p95 / p99 / max = 2.33 / 3.06 / 3.35 /
  3.57 ms (DEV) and 2.37 / 3.15 / 3.95 / 4.04 ms (DIAG); 0 of 90 turns over 5 ms on either. Only one sample above
  5 ms on any sub-5-ms boundary: a controller decision (5.17 ms; 5.16 ms in the earlier run - a specific chunk).
* **Gates, `--reps 3`**: G2-G6 100 % on the legacy (8x), real-corpus (8x), DEV (1x) and DIAG (1x) suites; false
  triggers 0 %; fabricated ids 0; schema errors 0. G1 not run (no Docker).
* **Baseline comparison (DIAG, paired 75 turns)**: utterance-end -> answer p50 2.38 ms (streaming) vs 110.18 ms
  (baseline), 46x; Task 4 section 4.6 condition MET. Streaming = deferred item for item.
* **Ablations (paired vs default)**: no refusal gate answer recall +10.8 [+4.7, +17.9] DEV / +4.1 [+1.0, +8.2] DIAG
  (significant; abstention on HeySQuAD 70.3 -> 3.6 % and 69.4 -> 4.6 %); no cross-encoder rerank r@1 -8.7 / -6.9
  (significant; end-to-end -2.9 [-9.1, +2.8] / 0.0, n.s.); BM25-only r@1 -0.7 * / 0.0, dense-only +0.2 / -0.5
  (n.s.); lexical selector +4.9 / +2.0 (n.s., confounded with the gate); speculative final pass off: 0 answer
  differences. Hybrid fusion's own contribution is small once the cross-encoder sees 20 candidates (kept, and
  said so, in the report). Gate break-even ~8.8 % unanswerable (utility rule of the gate fit); default unchanged.
* **Edge cases (pool 20)**: EC1 gate withholds a correct claim (P 0.378 < 0.40); EC2 exact-phrase distractor sentence
  wins (P 0.831); EC4 near-tie between neighbouring sentences (P 0.650); EC5 label artefact; **EC3 (Tesla) is now
  correct** - it was a rerank-depth failure at pool 5 (wrong-entity claim at P 0.867), the pool-20 change removes it.
  The trace tool recomputes the gate probability and asserts it equals the synthesizer's for every case.
* **Telemetry overhead**: p95 +2.6 % (jsonl) / -3.0 % (buffered) vs the null sink; criterion <= 5 % met.
* **Reproducibility without Docker (`cleanroom.ps1`, `netcheck.py`)**: fresh venv from `requirements.lock` in 125 s,
  `pip check` clean, pinned models re-fetched and verified, reader hash matches, fixture corpus rebuilt
  byte-identical (42 / 42), the container's command PASS on the CPU (55 s), 356 tests + 1 skipped in the clean
  environment, gate suite with all non-loopback connections refused: PASS, 0 attempts; regex secrets scan of
  14,020 files: 0 hits. Not covered: the image build, compose, Python 3.10 / 3.12.
* **Cost-model criterion**: exact part is tested; the 5-call provider sample needs a key (not available; the default
  path makes no LLM call). Recorded as not verified.
* **Demo tooling (found while capturing the demo)**: a cold first replay in a fresh process showed 171 ms
  post-speech (first CUDA calls); a warm replay 2.3 ms. `cli replay --warmup` reproduces the harness's uncounted
  warm-up; `cli evidence` shows per-channel ranks; Makefile `demo` and a corrected `compare` target; `tests/test_cli.py`.
  Full suite: 359 passed.

---

# SP1 - Second-machine pass (30 Sep 2026): RTX 4050 Laptop GPU (6 GB), Windows 11 Home, Python 3.11.9

Appended after the final audit; earlier entries are history and are not rewritten (for example FA6 records that Docker
was not installed on the workstation). Sealed test2 was not read, run or tuned on.

* **Fresh install and reproduction.** A new virtual environment from `requirements-gpu.txt` (the copies of `.venv` /
  `.venv-gpu` that came with the folder point at another user's profile and do not run). Pinned models re-fetched and
  SHA-256 verified; `pytest` 359 passed (23.7-28.0 s); legacy gate suite PASS; HeySQuAD DIAG typed reproduced the audited row
  exactly (76.7 % / 71.3 % / 69.4 %); every command of `docs/demo_script.md` returned the recorded answers.
* **Audio front end re-measured** (100 dev + 100 diag HeySQuAD clips, hypothesis events kept): Whisper `small.en` beats
  `base.en` on DEV citation recall (+5.0), claim precision (+3.3) and over-split (-7.0), significant; DIAG same direction, not
  significant. On 40 clips, clean text 67.5 %, `base.en` 52.5 %, dataset ASR text 37.5 %. The eight clean-vs-audio losses were
  5 Whisper mishearings / alternate forms of rare names, 2 gate-threshold borderlines (p = 0.326 vs 0.418; 0.364 vs 0.479),
  1 repeated phrase. **Tried and reverted:** stripping the trailing "?" (over-split 2/40 -> 7/40) and scoring the refusal gate on
  a punctuation-normalised query (DEV +3.0, DIAG +2.0, both n.s.; typed DIAG abstention 69.4 -> 67.6 %) - a recall / abstention
  trade-off, not a fix. Nothing of it remains in the code.
* **Latency re-measured, real time.** Text, DEV: post-speech p50 0.84 / p99 1.38 ms, 0 of 90 turns over 5 ms (retrieval about
  twice as slow as on the workstation, during speech). Speech, 100 clips: p50 0.74 / p95 100 / p99 181 ms, 41 of 100 over 5 ms;
  Whisper's own final decode adds a median 149 ms (p99 332 ms). This supersedes the pool-5 W8 figures (2.9 / 91 ms).
* **Packaging.** Fully pinned `requirements-gpu.lock` (84 packages, install with `--no-deps`) and `requirements-cpu.lock`
  (71 packages, Linux / CPython 3.12, every pin has a wheel); `.gitignore` now covers `.venv*/`, Whisper weights, audio clips and
  `demo*.jsonl`; `docker-compose.yml` no longer requires a `.env`; `.dockerignore` excludes every `.venv*`; the Dockerfile sets
  `PIP_DEFAULT_TIMEOUT=120` / `PIP_RETRIES=10`.
* **Docker.** Docker Desktop 4.93.0 was installed with winget; the engine would not start until the WSL 2 package was installed
  (an administrator step; Docker's log: `wsl is not installed`). `docker compose up --build` then failed twice on a slow
  connection (pip read timeout on the ~195 MB torch wheel; a resumed partial download failed its checksum) and was retried on
  a faster network: the image built (2.69 GB) and the container printed G2 16/16, G3 10/10, G4 68/68 (fabricated=0), G5 10/10,
  G6 67/67, **VERDICT: PASS**, exit 0, about 22 minutes end to end. G1 therefore passed, with the caveats in `final_report.docx`
  section 0.7 (working tree, not a fresh clone; two early layers cached from the failed attempts).
* **Output and tools.** `cli replay --guide-format` and `tools/answer_audio.py` print the guide's five-field structured output
  record; `docs/architecture_diagram.md` holds the Mermaid diagram.
* **Submission.** Demo video uploaded (`demo_video.txt`), presentation `SRMIST_VirtualVanguards_Submission.pptx`, signed AI
  disclosure `LangAI3.0_AI_Disclosure.docx`. `final_report.docx` was condensed from 108 KB to 48 KB (every audited number kept);
  `context.md` was deleted. On 1 Oct 2026 the report and `docs/architecture_brief.docx` were converted from Markdown to Word
  (report 13 pages; brief 3 pages, limit 6) and every reference in the project was updated from `.md` to `.docx`.
