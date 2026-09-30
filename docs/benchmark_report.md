# Benchmark & Evaluation Report - final configuration

**Status: current (30 Sep 2026).** This is the guide's benchmark deliverable for the *final* system: baseline
comparison, ablations, edge-case analysis, latency, gate results and limitations, every number from
`eval/results/final_audit/` (GPU only, real time, nothing excluded, sealed test2 untouched). The complete audited
narrative, the requirement checklist (section 0.1) and the reproduction commands are in
[`final_report.docx`](../final_report.docx) (section 0 and 5.10-5.15); the sections below are copies of the tables there.
Text and speech latency are reported separately in `final_report.docx` section 0.4, and the second-machine (RTX 4050)
re-verification and the audio measurements in section 0.9. Everything after the horizontal rule at the end is the
historical 23 September record, kept unchanged; its references to `final_report.docx` sections (for example 6.5, 6.6.2 and
9.3) point to that older revision of the report, not to the current one.

## Final configuration

BM25 + e5-small-v2 (ONNX) -> reciprocal-rank fusion (0.5 / 0.5) -> MiniLM-L6 cross-encoder over the top **20** fused
candidates -> cross-encoder claim selection over the top 3 chunks -> learned refusal gate (threshold 0.40, SQuAD2
reader margin) -> extractive, cited answer; rule-based controller (WAIT / RETRIEVE / SUPPRESS, multi-intent
decomposition, supersession); speculative retrieval during speech and a speculative final pass. All neural inference
on the RTX 6000 Ada (CUDA mandatory, no CPU fallback). Sealed test2 was not used.

## Official gates (`eval.run_all --reps 3`, median of 3)

| Suite (`eval.run_all --reps 3`) | Scenarios | G2 early retrieval (false triggers) | G3 multi-intent | G4 grounding (fabricated) | G5 refinement | G6 telemetry | Verdict |
|---|---|---|---|---|---|---|---|
| Legacy synthetic (fixtures/dev_corpus), 8x | 47 | 16/16 = 100.0%  (false-trigger rate 0.0%) | 10/10 = 100.0% | 68/68 = 100.0%  (fabricated=0) | 10/10 = 100.0% | 67/67 = 100.0%  (schema errors=0) | **PASS** |
| Real corpus, 15 scenarios, 8x | 15 | 11/11 = 100.0%  (false-trigger rate 0.0%) | 5/5 = 100.0% | 36/36 = 100.0%  (fabricated=0) | 5/5 = 100.0% | 25/25 = 100.0%  (schema errors=0) | **PASS** |
| **DEV**, 60 scenarios, **real time (1x)** | 60 | 45/45 = 100.0%  (false-trigger rate 0.0%) | 20/20 = 100.0% | 113/113 = 100.0%  (fabricated=0) | 20/20 = 100.0% | 90/90 = 100.0%  (schema errors=0) | **PASS** |
| **DIAG**, 60 scenarios, **real time (1x)** | 60 | 49/49 = 100.0%  (false-trigger rate 0.0%) | 20/20 = 100.0% | 114/114 = 100.0%  (fabricated=0) | 20/20 = 100.0% | 90/90 = 100.0%  (schema errors=0) | **PASS** |

G1 (clean-machine `docker compose up`) was **not run** on the workstation (no Docker there); its current status is in
`final_report.docx` section 0.7. The CLI clean-room substitute is described in section 5.15 and is not G1.

## Baseline comparison

`python -m eval.compare --modes streaming,deferred,baseline --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1`
(60 scenarios, 90 turns; `eval/results/final_audit/compare_diag.log`). *streaming* is the engine as shipped;
*deferred* is the same pipeline run only after the utterance ends; *baseline* retrieves once, after the
utterance ends, with no controller (no decomposition, no suppression, no refinement delta).

```
mode |       ttfr_p50 |       ttfr_p95 |        e2e_p50 |        e2e_p95 |      total_p50 |      total_p95 |             G2 |             G3 |             G4 |             G5 | tokens_per_turn |  cost_per_turn
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     streaming |           0.00 |        2400.00 |           2.38 |           3.13 |        6302.09 |       13462.84 |           1.00 |           1.00 |           1.00 |           1.00 |           0.00 |           0.00
      deferred |        6300.00 |       13460.00 |         112.08 |         267.09 |        6412.97 |       13719.18 |           0.00 |           1.00 |           1.00 |           1.00 |           0.00 |           0.00
      baseline |        5700.00 |       12920.00 |         110.07 |         168.72 |        5805.48 |       13051.57 |           0.00 |           0.00 |           1.00 |           0.00 |           0.00 |           0.00

Turns that did retrieval: streaming 75, baseline 90; in BOTH (compared below): 75. Baseline has no suppression, so it also retrieves for presentation-only turns that streaming correctly skips - those are excluded here.
  paired turns |   ttfr_p50 |   ttfr_p95 |    e2e_p50 |    e2e_p95 |  total_p50 |  total_p95   (ms)
     streaming |       0.00 |    2400.00 |       2.38 |       3.13 |    6302.09 |   13462.84
      baseline |    6300.00 |   13460.00 |     110.18 |     173.85 |    6408.11 |   13588.00

utterance-end -> answer p50 (paired turns): streaming 2.38 ms vs baseline 110.18 ms (streaming is faster by 107.80 ms, 46x)
(informational) utterance-start -> answer p50 (paired turns): streaming 6302.1 ms vs baseline 6408.1 ms - dominated by how long the user speaks, identical for both
PASS CONDITION (Task 4 section 4.6: streaming beats baseline on utterance-end -> answer p50): MET
```

Reading it:

* **The Task 4 section 4.6 pass condition is met**: on the 75 turns that both streaming and the baseline
  retrieved for, the time from utterance end to answer is **2.38 ms (streaming) vs 110.18 ms (baseline)** at the
  median, 46x faster; p95 3.13 vs 173.85 ms. (The baseline also "retrieves" for the 15 presentation-only turns that
  streaming correctly suppresses; those are excluded from the pairing so both sides are compared on the same turns.)
* Time to *first* retrieval: 0 ms at the median for streaming (it retrieves at the first chunk when the intent is
  already stable), 2.4 s at p95; the baseline can only start after the user stops (5.7 s median, i.e. how long the
  utterances are).
* Total time from the start of the utterance to the answer is dominated by how long the user speaks, so it is
  the same (streaming 6,302 ms vs baseline 6,408 ms at the median) - the gain is in the wait *after* speech.
* **Streaming costs no accuracy**: streaming and deferred give item-for-item identical answers (paired difference
  0.0 % on every metric, CI [0, 0]). Against the baseline, streaming is significantly better on answer
  recall (+19.4 points [+10.7, +28.3]), citation recall (+21.4 [+11.7, +31.4]) and exact decomposition (0 % -> 100 %);
  claim precision is 1.9 points lower [-4.9, +0.0], not significant (n = 62 paired questions; not investigated further).
* G2 (early retrieval) is 1.00 only for streaming by construction (the others do not retrieve early); G3 is 0 for
  the baseline (it does not decompose). Tokens and cost are 0.00 in every mode because the default pipeline
  makes no LLM call.

| Engine mode (DIAG) | Answer | Citation | Claim precision | Abstention | Exact decomp. |
|---|---|---|---|---|---|
| streaming | 78.6% | 84.7% | 94.3% | 100.0% | 100.0% |
| deferred | 78.6% | 84.7% | 94.3% | 100.0% | 100.0% |
| baseline | 59.2% | 63.3% | 96.8% | 100.0% | 0.0% |

Paired, deferred -> streaming:
- answer: 78.6% -> 78.6%, diff +0.0% [+0.0%, +0.0%] (n=98)
- citation: 84.7% -> 84.7%, diff +0.0% [+0.0%, +0.0%] (n=98)
- claim_correct: 95.1% -> 95.1%, diff +0.0% [+0.0%, +0.0%] (n=64)
- abstention: 100.0% -> 100.0%, diff +0.0% [+0.0%, +0.0%] (n=5)
- split_exact: 100.0% -> 100.0%, diff +0.0% [+0.0%, +0.0%] (n=20)
- oversplit: 0.0% -> 0.0%, diff +0.0% [+0.0%, +0.0%] (n=35)

Paired, baseline -> streaming:
- answer: 59.2% -> 78.6%, diff +19.4% [+10.7%, +28.3%]* (n=98)
- citation: 63.3% -> 84.7%, diff +21.4% [+11.7%, +31.4%]* (n=98)
- claim_correct: 96.8% -> 94.9%, diff -1.9% [-4.9%, +0.0%] (n=62)
- abstention: 100.0% -> 100.0%, diff +0.0% [+0.0%, +0.0%] (n=5)
- split_exact: 0.0% -> 100.0%, diff +100.0% [+100.0%, +100.0%]* (n=20)
- oversplit: 0.0% -> 0.0%, diff +0.0% [+0.0%, +0.0%] (n=35)

## Ablations

Seven configurations, each one switch away from the default, all paired against the default on the same
questions (paired cluster bootstrap; `*` = the 95 % interval excludes 0). Generated by `handoff/final_audit/run_final_audit.sh`
(`eval/results/final_audit/abl_*`), tables by `make_tables.py ablations`. The guide's two named examples are
`retrieval.mode` hybrid vs dense, which is included (plus BM25-only and no-rerank), and `controller.mode` rule vs
model, which is **not** run: `controller.mode=model` is not implemented (an LLM-backed controller would need a
provider key and contradicts the "fast rule path first" parsimony rule); see 0.1 row 14.

**DEV** (paired against the default configuration; '*' = 95 % CI excludes 0)

| Configuration | Answer | d answer [95 % CI] | Citation | Claim precision | d claim precision | Abstention |
|---|---|---|---|---|---|---|
| baseline (defaults) | 65.7% | - | 72.5% | 87.4% | - | 100.0% |
| retrieval.mode=sparse (BM25 only) | 63.7% | -2.0% [-5.0%, +0.0%] | 69.6% | 85.9% | -1.6% [-4.8%, +0.0%] | 100.0% |
| retrieval.mode=dense (e5 only) | 65.7% | +0.0% [+0.0%, +0.0%] | 72.5% | 87.4% | +0.0% [+0.0%, +0.0%] | 100.0% |
| no cross-encoder rerank | 62.7% | -2.9% [-9.1%, +2.8%] | 68.6% | 84.7% | -1.6% [-6.7%, +3.2%] | 100.0% |
| no refusal gate | 76.5% | +10.8% [+4.7%, +17.9%]* | 83.3% | 87.9% | -0.3% [-1.6%, +0.7%] | 100.0% |
| lexical claim selector (also no gate) | 70.6% | +4.9% [-4.9%, +14.4%] | 80.4% | 83.7% | -4.0% [-10.7%, +1.9%] | 100.0% |
| speculative final pass off | 65.7% | +0.0% [+0.0%, +0.0%] | 72.5% | 87.4% | +0.0% [+0.0%, +0.0%] | 100.0% |

**DIAG** (paired against the default configuration; '*' = 95 % CI excludes 0)

| Configuration | Answer | d answer [95 % CI] | Citation | Claim precision | d claim precision | Abstention |
|---|---|---|---|---|---|---|
| baseline (defaults) | 78.6% | - | 84.7% | 94.3% | - | 100.0% |
| retrieval.mode=sparse (BM25 only) | 78.6% | +0.0% [+0.0%, +0.0%] | 84.7% | 94.3% | +0.0% [+0.0%, +0.0%] | 100.0% |
| retrieval.mode=dense (e5 only) | 77.6% | -1.0% [-3.2%, +0.0%] | 83.7% | 93.2% | -1.6% [-4.9%, +0.0%] | 100.0% |
| no cross-encoder rerank | 78.6% | +0.0% [-4.2%, +4.0%] | 84.7% | 94.3% | -2.2% [-6.3%, +0.5%] | 100.0% |
| no refusal gate | 82.7% | +4.1% [+1.0%, +8.2%]* | 91.8% | 93.8% | +0.0% [+0.0%, +0.0%] | 100.0% |
| lexical claim selector (also no gate) | 80.6% | +2.0% [-5.4%, +9.6%] | 93.9% | 93.8% | -2.6% [-7.5%, +1.1%] | 100.0% |
| speculative final pass off | 78.6% | +0.0% [+0.0%, +0.0%] | 84.7% | 94.3% | +0.0% [+0.0%, +0.0%] | 100.0% |

**Refusal gate on the HeySQuAD sets** (typed questions; ~50 % are hard SQuAD2-style unanswerable ones)

| Set | Configuration | Answer | Citation | Correct abstention | d answer | d abstention |
|---|---|---|---|---|---|---|
| DEV (279 + 279 Q) | learned gate (default) | 71.7% | 76.7% | 70.3% | - | - |
| DEV (279 + 279 Q) | no gate | 84.2% | 93.2% | 3.6% | +12.5% [+8.6%, +16.5%]* | -66.7% [-72.4%, -60.9%]* |
| DIAG (219 + 219 Q) | learned gate (default) | 76.7% | 83.6% | 69.4% | - | - |
| DIAG (219 + 219 Q) | no gate | 84.9% | 92.7% | 4.6% | +8.2% [+4.6%, +11.9%]* | -64.8% [-70.8%, -58.4%]* |

**Retrieval only** (1,000 questions per split, no answer synthesis; `retrieval_ablation_dev.txt`,
`retrieval_ablation_diag.txt`; the ms / query figures were measured while other jobs ran and are not latency
evidence):

| Retrieval configuration | DEV r@1 | DEV r@5 | DIAG r@1 | DIAG r@5 |
|---|---|---|---|---|
| hybrid (BM25 + e5, RRF) + cross-encoder over 20 = default | 88.7 % | 98.3 % | 89.9 % | 97.7 % |
| BM25 only + cross-encoder | 88.0 % (-0.7 *) | 96.6 % (-1.7 *) | 89.9 % (0.0) | 97.0 % (-0.7) |
| e5 dense only + cross-encoder | 88.9 % (+0.2) | 98.2 % (-0.1) | 89.4 % (-0.5) | 97.6 % (-0.1) |
| hybrid, **no** cross-encoder | 80.0 % (**-8.7 \***) | 96.4 % (-1.9 *) | 83.0 % (**-6.9 \***) | 96.1 % (-1.6 *) |
| hybrid + cross-encoder over 5 (pre-audit default) | 88.0 % (-0.7 *) | 96.4 % (-1.9 *) | 89.4 % (-0.5) | 96.1 % (-1.6 *) |

What the ablations say, including the parts that do not flatter the design:

1. **The cross-encoder rerank is the one retrieval component that clearly matters**: without it r@1 falls 8.7 /
   6.9 points (significant on both splits). End to end the effect is smaller and not significant (answer recall
   -2.9 [-9.1, +2.8] DEV, 0.0 DIAG) because claim selection re-scores sentences anyway.
2. **Hybrid fusion is a small gain, not a large one, once the cross-encoder sees 20 candidates**: it helps
   r@5 over BM25-only (+1.7 DEV, significant; +0.7 DIAG, not) and is indistinguishable from dense-only at r@1
   (+0.2 / -0.5, n.s.). End to end, sparse-only and dense-only differ from the default by at most 2 points of
   answer recall (all intervals include 0). The honest reading is that hybrid is insurance (the two channels miss
   different questions), not a measured big win on these corpora.
3. **The refusal gate trades answer recall for abstention, and its value depends on the share of unanswerable
   questions.** On the text sets (only 5 unanswerable questions, all easy and off-topic, so even no gate abstains
   5 / 5) removing the gate raises answer recall by +10.8 [+4.7, +17.9] (DEV) and +4.1 [+1.0, +8.2] (DIAG), both
   significant, with claim precision unchanged: on those sets the gate only costs recall. On HeySQuAD (about
   50 % hard SQuAD2-style unanswerable questions) removing it raises answer recall by +12.5 [+8.6, +16.5] /
   +8.2 [+4.6, +11.9] but collapses correct abstention from 70.3 % to 3.6 % and 69.4 % to 4.6 %
   (-66.7 / -64.8 points, significant). From the recorded check-set numbers of the gate fit (with gate: recall
   68.0 %, wrong answers 9.1 %, answered-unanswerable 27.9 %; without: 82.9 %, 17.1 %, 99.5 %; a wrong answer
   costs as much as a right one is worth) the gate has the higher utility whenever **more than about 8.8 % of the
   questions are unanswerable** (U_gate = 0.589 (1 - p) - 0.279 p; U_none = 0.658 (1 - p) - 0.995 p). The default
   was left unchanged (an earlier "no gate" default was reverted on the user's instruction); which side of that
   break-even the real workload is on is the deployer's decision.
4. The lexical-selector row is confounded with the gate (the lexical path has none): against the no-gate
   cross-encoder run (76.5 % / 82.7 %) the cross-encoder selector is ahead of the lexical one (70.6 % / 80.6 %),
   not significant at these sizes.
5. **Speculative final pass off: identical answers (0 differences)**, as designed: it changes latency, never
   content.

## Edge cases

The guide asks for at least three analysed edge-case failures. Each was traced stage by stage through the
production components (`handoff/final_audit/trace_case.py`; full output, including the refusal gate's six
feature values, in `eval/results/final_audit/edge_case_traces.txt`; the trace recomputes the gate probability
and asserts it equals the synthesizer's own value - it did for all five). The 35 DEV misses of
`eval.error_analysis` on the final system (refusal gate 11, context selection 10, sentence selection 6, low-
confidence refusal 2, chunking 2, rerank depth 1, first stage 1, relevance gate 1, cross-intent contamination 1)
supply four still-failing cases; the fifth is a former failure that the rerank-pool change removed.

| # | Question (DEV) | What the trace shows | Root cause | Component | Mitigation status |
|---|---|---|---|---|---|
| EC1 | "What did Kenya reveil in 2030?" (gold: Vision 2030) | The gold sentence is ranked **1 of 11**, the claim is correct - and it is refused: P(correct) = **0.378** < 0.40. Every gate feature is close to its mean (top rerank logit only 2.9), so no single feature is at fault | A misspelling lowers every model's score a little; the logistic gate was calibrated on typed questions and this answer lands just under the threshold | Refusal gate (`session/synthesis.py`, `refusal_gate.json`) | None adopted. Lowering the threshold trades directly against wrong answers on unanswerable questions (the threshold comes from an explicit utility rule); a corpus-vocabulary typo corrector was prototyped (+1.1 pt r@1, n.s.) and not adopted. Largest single miss class (11 / 35) |
| EC2 | "Who was the Most Valuable Player of Super Bowl 50?" (gold: Von Miller) | The gold chunk is retrieved (rank 4) but the cross-encoder puts a regular-season sentence, "quarterback Cam Newton was named the NFL Most Valuable Player (MVP)", first; the gate asserts that wrong claim with P = **0.831**. The reader's best span (Bart Starr) is not in the claim either | The question's phrase "Most Valuable Player" occurs verbatim in an unrelated sentence, and the cross-encoder rewards the exact phrase; every gate feature is high for the distractor | Claim selection (cross-encoder sentence scoring) + gate | Reader-vs-selector disagreement is what the earlier "agreement" gate feature measured: offline it cut wrong answers 9.1 % -> 5.9 % (significant) but the end-to-end effect was not significant, so it was not adopted; using the reader as selector or arbiter was worse (5.3). Residual risk, documented (context selection 10 / 35 misses) |
| EC3 | "What year did Tesla die?" (gold: 1943) | **Fixed by the pool change (5.13).** At rerank pool 5 the chunk that names the year was never seen by the cross-encoder; the top chunk was about Tesla's *father* (rerank logit 9.04) and "Milutin Tesla died 1879" was asserted at P = 0.867. At pool 20 the death chunk enters the pool (rank 2, logit 8.97), "Tesla died on 7 January 1943." is selected and asserted at P = 0.903 - correct | In a biography the bare surname names several people and the right chunk sat outside the top 5 of the fused list | Retrieval (rerank depth) | **Adopted**: `retrieval.rerank_pool` 5 -> 20. Residual: the rerank-depth miss class fell from 4 to 1 |
| EC4 | "What is the name of the turf used in Levi's Stadium for the Super Bowl?" (gold: Bermuda 419) | The right chunk is rank 1; inside it the cross-encoder ranks a sentence about field-quality concerns above the sentence that names the turf; asserted at P = 0.650; the reader's span "Bermuda 419" is not in the claim | Near-tie between neighbouring sentences; the answer sentence is the less "question-shaped" one | Claim selection (sentence choice; 6 / 35 misses) | Same disagreement signal as EC2, same status (not adopted) |
| EC5 | "How many tree species are in the rainforest?" (gold: 1,100) | Counted as a miss, but the returned claim - "The total number of tree species in the region is estimated at 16,000." - literally answers the question. The gold figure is a per-area count from another sentence of the same section | Under-specified question and a label written against one sentence; filed under "chunking" by the error analysis | Evaluation labels (a metric artefact), not the pipeline | None needed; a reminder that answer recall is a lower bound |

**Behavioural edge cases found and fixed in this audit** (0.5, `docs/experiment_log.md` FA2): a reformat request
that searched the corpus and overwrote the good answer, a reformat request with no earlier answer, a
chit-chat sentence that was not recognised, superseded fragments listed as sub-queries, two sessions sharing an
utterance id, and a silent regex sentence-splitter fallback on the GPU - each with a regression test.

## Latency (real time, GPU only)

**DEV** - 60 scenarios, real time (1x), providers ['CUDAExecutionProvider'], speculative final-pass retrievals launched 6, reused 2

| Boundary | n | p50 | p95 | p99 | max (ms) | p99 < 5 ms |
|---|---|---|---|---|---|---|
| Controller decision, per transcript chunk | 330 | 0.50 | 0.93 | 1.31 | 1.83 | **yes** |
| Final controller pass at utterance end | 90 | 0.54 | 0.87 | 0.92 | 1.05 | **yes** |
| Answer assembly at utterance end | 90 | 1.47 | 2.08 | 2.16 | 2.27 | **yes** |
| Answer telemetry / bookkeeping | 90 | 0.06 | 0.07 | 0.11 | 0.18 | **yes** |
| **Post-speech: utterance_end -> answer emitted (all turns)** | 90 | 2.33 | 3.06 | 3.35 | 3.57 | **yes** |
|   of which waiting for in-flight retrieval / claim selection | 90 | 0.20 | 0.27 | 0.34 | 0.38 | **yes** |
| *context (not a < 5 ms target)* | | | | | | |
| Retrieval per sub-query (runs during speech) | 212 | 82.3 | 140.3 | 179.7 | 254.5 | - |
|   e5 query encode | 216 | 10.5 | 18.3 | 26.5 | 61.7 | - |
|   BM25 | 216 | 3.5 | 6.6 | 7.4 | 12.1 | - |
|   cross-encoder rerank of 20 passages | 216 | 49.1 | 84.8 | 143.8 | 192.9 | - |
| Claim selection + refusal gate (prefetch, during speech) | 216 | 42.0 | 59.8 | 94.9 | 200.9 | - |

Turns with post-speech > 5 ms: 0 of 90.

**DIAG** - 60 scenarios, real time (1x), providers ['CUDAExecutionProvider'], speculative final-pass retrievals launched 0, reused 0

| Boundary | n | p50 | p95 | p99 | max (ms) | p99 < 5 ms |
|---|---|---|---|---|---|---|
| Controller decision, per transcript chunk | 347 | 0.53 | 0.88 | 1.02 | 5.17 | **yes** |
| Final controller pass at utterance end | 90 | 0.57 | 0.78 | 0.90 | 0.97 | **yes** |
| Answer assembly at utterance end | 90 | 1.52 | 2.16 | 2.89 | 2.89 | **yes** |
| Answer telemetry / bookkeeping | 90 | 0.06 | 0.07 | 0.11 | 0.37 | **yes** |
| **Post-speech: utterance_end -> answer emitted (all turns)** | 90 | 2.37 | 3.15 | 3.95 | 4.04 | **yes** |
|   of which waiting for in-flight retrieval / claim selection | 90 | 0.19 | 0.23 | 0.27 | 0.27 | **yes** |
| *context (not a < 5 ms target)* | | | | | | |
| Retrieval per sub-query (runs during speech) | 240 | 81.6 | 167.0 | 200.0 | 248.1 | - |
|   e5 query encode | 240 | 10.5 | 21.9 | 26.2 | 33.8 | - |
|   BM25 | 240 | 3.9 | 6.6 | 7.7 | 9.2 | - |
|   cross-encoder rerank of 20 passages | 240 | 49.8 | 121.2 | 171.9 | 206.7 | - |
| Claim selection + refusal gate (prefetch, during speech) | 240 | 40.7 | 77.7 | 103.3 | 211.4 | - |

Turns with post-speech > 5 ms: 0 of 90.

## Limitations

See `final_report.docx` section 0.7 and 10: G1 passed with caveats (Docker run from the working tree); cost accuracy against a provider not run (no
key; default path has no LLM call); `controller.mode` rule-vs-model ablation not implemented; answer-recall ceiling
from the refusal gate and sentence selection; the gate's value depends on the share of unanswerable questions
(break-even about 8.8 %); the SQuAD2 reader is in-domain for the question style; generative synthesis unmeasured;
sealed numbers predate the final build; audio input is outside the scope and does not meet p99 < 5 ms (speech latency
and audio answer quality: `final_report.docx` sections 0.4 and 0.9).

---

# Historical record (23 September 2026) - superseded

# Benchmark & Evaluation Report

> **Status (23 Sep 2026): historical record, superseded by `final_report.docx`.** The passes below
> were all tuned and measured on the same `data/corpus` scenarios, and their headline metric (G4)
> is the synthesizer grading its own extractive claims — it never compares against gold answers.
> A later audit found that on a held-out split scored against SQuAD's gold answers, the system
> documented here answered 65.3% of questions and 29.8% of its asserted claims were wrong; it also
> found three bugs that affected these numbers (a citation-resolution bug, a licence file indexed
> as a document, a relevance-gate edge case). See `final_report.docx` §2 for the corrections and §5
> for the current, held-out results. The content below is kept unchanged as the record of what was
> tried.
>
> **Where the current deliverables are (final audit, 30 Sep 2026, `final_report.docx`):** baseline vs streaming
> comparison and engine-mode table (section 5.10), ablations (section 5.11: hybrid vs sparse vs dense, cross-encoder
> rerank, refusal gate, claim selector, speculation), five traced edge-case failures with root cause -> component ->
> mitigation (section 5.12), the rerank-pool decision (section 5.13), telemetry overhead (section 5.14) and the
> reproducibility check (section 5.15). Every number there comes from `eval/results/final_audit/`, produced by
> `handoff/final_audit/run_final_audit.sh`.

## 0. Optimization pass — before / after (real corpus, SQuAD v1.1)

Three successive diagnostic-and-tune passes were run against `data/corpus` (the real, unseen
corpus). Every row below is a measured result, not a projection; the full trace investigation
for each pass is in §7, §0b and §0c respectively.

| Metric | Session-start baseline | Final (Pass 3) | Target | Met? |
|---|---|---|---|---|
| G2 early retrieval | 86.7% (13/15) | **100.0%** (11/11) | > 95% (requested) / ≥ 80% (gate) | ✅ both |
| G4 grounding support | 88.9% (56/63) | **97.5%** (39/40) | > 95% (requested) / ≥ 85% (gate) | ✅ both |
| G4 fabricated citations | 0 | **0** | 0 | ✅ |
| G3 multi-intent | 100% | **100%** | 100% | ✅ |
| G5 refinement continuity | 100% | **100%** | 100% | ✅ |
| G6 telemetry coverage | 100% | **100%** | 100% | ✅ |
| Retrieval r@1 (hybrid, 500-qrel sample) | 62.0% | **63.2%** | > 62.0% | ✅ |
| Retrieval r@5 (hybrid, 500-qrel sample) | 84.0% | **86.8%** | > 84.0% | ✅ |
| Test suite | 72/72 | **72/72** | green | ✅ |
| `--reps 3` determinism | — | **identical to `--reps 1`** (re-confirmed after every pass) | deterministic | ✅ |

Both originally-requested >95% targets are now met, on top of every gate threshold, with zero
fabricated citations maintained throughout all three passes. See §0c for how the final G4 gap was
closed (a general, reusable definitional-query reranking fix, not corpus-specific overfitting).

**Changes made** (each verified to move a real number, not just plausible in theory):

1. **spaCy rule-based sentence segmentation** (`retrieval/text.py`) replaces a naive `.!?` regex
   split, fixing mid-quote fragmentation (`Tesla once wrote, "What can I say?"` no longer breaks
   into a spurious tiny "sentence"). A blank spaCy pipeline + `sentencizer` — no language model
   download, fit once at corpus-ingest cold start. **G4: 88.9% → 90.5%.**
2. **Within-utterance retrieval supersession** (`controller/controller.py::_link_supersession`,
   `engine.py::_cancel_task`): a growing single clause across chunks
   ("Before which..." → "...did Chagatai publicly" → "...dispute Jochi's paternity?") was
   re-triggering a **fresh, uncancelled** retrieval on every qualifying chunk. The engine's
   existing `parent_query_id` cancellation only handled cross-turn refinement, not this
   within-turn case, and — a second, independent bug — `_cancel_task` early-returned without
   cleanup when the superseded task had *already completed* (the common case for a fast local
   retriever), so stale evidence silently reached the synthesizer regardless. Both are fixed.
   **G4: 90.5% → 92.7%; n_claims dropped 63→41 (fewer, more precise claims, exactly as
   intended).**
3. **Weak-anchor "known topic term" heuristic** (`controller/stability.py::has_topic_anchor`,
   `retrieval/bm25.py::specific_terms`): a content token that is a *discriminative* corpus term
   (high idf, i.e. specific rather than generic) now counts as a stability anchor even without
   capitalisation or a number, via an injected `corpus_vocab` callable (kept as a callable, not a
   `Retriever` reference, so the controller stays decoupled from the retrieval protocol).
   **G2: 86.7% → 93.3%, false-trigger rate unchanged at 0.0%.**
4. **RRF fusion weight recalibration** (`config.py`: `sparse_weight` 0.85→0.99, `dense_weight`
   0.15→0.01), swept on a 500-qrel real-corpus sample. The LSA dense encoder was actively
   *hurting* real-corpus r@5 (83.8% at 0.15 vs 86.8% at 0.01) with zero offsetting gain on
   dev_corpus paraphrase recall (identical at every weight tested — RRF's rank fusion still lets
   dense's top hits register at a small weight). **r@1: 62.0%→63.2%, r@5: 84.0%→86.8%, hybrid now
   genuinely beats sparse-only on both metrics** (previously tied/behind it).
5. **Query-relevance gate before claim creation** (`session/synthesis.py::_is_relevant`): the
   retriever's own `low_confidence` flag doesn't catch a chunk that clears the score threshold
   but doesn't answer the *specific* sub-query (e.g. "Tesla" appears in nearly every sentence of
   a Tesla biography, trivially "matching" any query mentioning him). Requires both a minimum
   overlap fraction and, once the query has more than one content term, at least two of them
   present — then falls back through the ranked evidence list (not just evidence[0]) for the
   first chunk that clears it.
6. **`retrieval.k` default 5→8**, giving the relevance-gate fallback more candidates to scan.
   Measured to make no further difference on this corpus (the correct evidence genuinely isn't
   in the top-8 for the remaining hard queries) — kept as a reasonable default regardless.

**Why the requested >95% targets were not fully reached, honestly stated (as of Pass 1)**: the
remaining G2/G4 gaps traced (§7) to genuinely hard, independently-authored SQuAD questions where
either (a) the correct evidence isn't recoverable by a lightweight local BM25+LSA retriever
regardless of tuning (a true retrieval-recall ceiling at this component's capability, not a
bug), or (b) the system is *correctly declining* to assert a topically-adjacent-but-wrong
sentence rather than fabricating. Pass 2 (§0b below) closed most of this gap with root-caused,
verified fixes; both gate-level thresholds (G2 ≥ 80%, G4 ≥ 85%) clear with comfortable margin
and zero fabrication throughout every run in this report.

## 0b. Exhaustive failure-bucket diagnosis — Pass 2

A second, exhaustive pass classified every remaining real-corpus failure into exactly one root
cause using a purpose-built tool (`eval/diagnose.py`) that replays the actual retrieval +
query-relevance-gate + grounding pipeline per gold `(query, doc)` pair — not an approximation of
it — and buckets each failure as **A** (gold doc never retrieved), **B** (gold doc retrieved but
rejected before/after becoming a claim), or **C** (eligible turn's retrieval fired at or after
`utterance_end`).

| Metric | After Pass 1 | After Pass 2 | Delta | Target | Met? |
|---|---|---|---|---|---|
| G2 early retrieval | 93.3% (14/15) | **100.0%** (11/11) | +6.7pp | > 95% requested / ≥ 80% gate | ✅ requested, ✅ gate |
| G4 grounding support | ~92.0% (34/37) | **92.9%** (39/42) | +0.9pp | > 95% requested / ≥ 85% gate | ⚠️ requested, ✅ gate |
| G4 fabricated citations | 0 | **0** | 0 | 0 | ✅ |
| G3 / G5 / G6 | 100% | **100%** | unchanged | 100% | ✅ |
| Test suite | 72/72 | **72/72** | unchanged | green | ✅ |
| `--reps 3` determinism | identical to `--reps 1` | **identical to `--reps 1`** | confirmed again | deterministic | ✅ |

(G2's denominator changed 15→11 and 14→11 across passes because a labeling bug in the scenario
generator — see fix 3 below — was itself inflating the failure count; the 11 remaining turns are
the ones genuinely long enough to have a pre-final chunk at all.)

**Bucket findings and fixes applied this pass:**

1. **[Real bug, not a Bucket A/B/C classification] `decompose()` silently dropped every
   single-clause utterance with fewer than 2 content tokens after stopwording** — e.g. "When are
   the ashes now?" has exactly one content token ("ashes") and produced **zero retrieval for the
   entire turn**, not a late one. Root-caused via the diagnostic tool flagging a turn with no
   `retrieval_started` event at all. The `_MIN_CLAUSE_TOKENS` filter existed to drop junk
   fragments left over from comma-splitting a multi-clause utterance; it should never have
   applied to a single, unsplit clause. Fixed in `controller/decompose.py::split_clauses` to only
   filter when there is more than one candidate clause. This was a real correctness bug, not a
   tuning question — a legitimate, answerable question was being silently discarded.
2. **[Ground-truth labeling bug in the scenario generator, not a system defect] `chunk_utterance`
   groups 6 words per fragment, so an utterance with ≤6 words is delivered as a single chunk
   immediately followed by the empty `is_final` chunk — there is no earlier moment retrieval
   COULD have started. The generator was unconditionally marking such turns
   `eligible_for_early_retrieval: True`, penalising G2 for a structural impossibility.** Fixed in
   `eval/scenario_gen.py::_has_early_opportunity`, applied to `gen_multi_intent`, `gen_late_detail`,
   `gen_suppression`. **Effect: G2 93.3%→100.0%** on the regenerated suite (both from this label
   fix removing the affected turn from the denominator, and one genuinely-eligible turn that now
   correctly retrieves early once fix 1 above stopped `decompose()` from dropping it).
3. **[Bucket C remedy, requested] DF-based bigram anchor**
   (`retrieval/bm25.py::low_df_bigrams`, `controller/stability.py::has_bigram_anchor`): a pair of
   adjacent content tokens that co-occurs in fewer than 5 corpus documents is trusted as a strong
   anchor on its own — "military campaign" pins a passage down far more precisely than either
   word alone. Implemented as requested; no remaining Bucket C findings to attribute the effect
   to specifically (fix 1+2 above resolved the only two G2 misses found), but the mechanism is
   live and covered by the full test suite.
4. **[Bucket B remedy, requested] Dynamic confidence floor by query length**
   (`session/synthesis.py::_min_relevance_for`): the overlap-fraction threshold tapers from 0.2
   (queries ≤3 content tokens, unchanged/strict) down to a floor of 0.12 (≥7 tokens), since a
   long, specific query's correct overlap is naturally spread across more tokens than a short
   one's.
5. **[Bucket B remedy, root-caused via the diagnostic tool] Single-token relevance override for
   discriminative terms**: the diagnostic tool's first run found a genuine over-rejection —
   query "When did Astor provide the money?" retrieved the exactly-correct sentence ("In 1899,
   John Jacob Astor IV invested $100,000...") at rank 0, but the existing "≥2 overlapping tokens"
   rule (added earlier specifically to stop "Tesla" trivially matching every sentence in its own
   biography) rejected it — the paraphrase legitimately shares only the person's name, since
   "provide the money" became "invested $100,000" with zero further lexical overlap. Fixed by
   reusing the retriever's discriminative-term vocabulary (`specific_vocabulary`, already built
   for the controller's weak-anchor heuristic) in `_is_relevant`: a single overlapping token is
   now accepted when that token is itself rare across the corpus (like "Astor"), while a ubiquitous
   term (like "Tesla" in its own biography) still requires a second corroborating token. Re-running
   the diagnostic after this fix found **zero remaining findings** in that scenario.

**Remaining irrecoverable cases (3 of 42 claims, all correctly flagged `uncertainty`, zero
fabricated)**, with justification for why closing them would risk the zero-fabrication rule:

1. `gen_late_detail_000` — a gold answer still sits inside a quoted rhetorical-question fragment
   pattern spaCy's default sentencizer doesn't fully disambiguate. Root cause: sentence
   segmentation, not retrieval or grounding logic; the system correctly abstains rather than
   guessing which side of the quote the fact belongs to.
2. ~~`gen_multi_intent_002` — "What does AC stand for?"~~ **[FIXED in Pass 3, §0c]**.
3. `gen_multi_intent_003` — the gold answer is illustrated via a worked example ("For example, if
   you know that two people...") whose defining sentence uses different terminology than the
   question. A genuine lexical paraphrase gap that a keyword-based retriever cannot close without
   semantic embeddings.

Both remaining cases are retrieval-recall or segmentation limits of the chosen lightweight,
offline architecture — not logic bugs — and in every case the system's response is an honest
`uncertainty` flag with zero citations, never a fabricated or wrong-topic assertion. Closing them
further would require either a stronger (heavier, non-parsimonious) encoder or loosening the
relevance/grounding thresholds specifically for these corpus passages, which would reintroduce
exactly the over-assertion risk the whole grounding pipeline exists to prevent.

## 0c. Pass 3 — closing the definitional-query gap (>95% G4 reached)

Case 2 above turned out to be generalisably fixable, not corpus-specific overfitting: a "what
does X stand for" / "what is X" / "define X" question about a short acronym-like term is a
recognisable *query shape*, and encyclopedic text conventionally introduces an acronym exactly
once, immediately next to its expansion ("alternating current (AC)"). A plain term-overlap
reranker under-ranks that one defining sentence whenever the acronym itself recurs far more often
elsewhere in the document (here, "AC" appears throughout the whole Tesla biography, but the
definition appears exactly once, in §1).

**Fix**: `retrieval/rerank.py::definitional_target` / `definitional_match` — detects the
definitional query shape via regex, extracts the target term, and adds a rerank boost to any
candidate chunk containing the conventional "expansion (TERM)" or "TERM (expansion)" parenthetical
pattern for that term. This is a general, reusable heuristic (any acronym, any document), not a
lookup table tied to "AC" or to Doc_09 — verified by it not touching any other query in either
scenario suite.

**Effect** (immediate re-verification, official `--reps 3` procedure):

```
$ python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 3 --corpus-dir data/corpus
scenarios run: 15
G2 early retrieval : 11/11 = 100.0%  (false-trigger rate 0.0%)
G3 multi-intent    : 5/5 = 100.0%
G4 grounding       : 39/40 = 97.5%  (fabricated=0)
G5 refinement      : 5/5 = 100.0%
G6 telemetry cov.  : 25/25 = 100.0%  (schema errors=0)
VERDICT: PASS
```

`data/corpus/Doc_09_nikola_tesla.md §1` (the definition passage) now ranks #1 for "What does AC
stand for?" (previously ranked below three sections that merely mention "AC" more often). Both
originally-requested targets are now met: **G2 100.0% (>95%) and G4 95.2% (>95%)** as of Pass 3.

## 0d. Pass 4 — a real supersession bug, traced and fixed (G4 95.2% → 97.5%)

A later debugging session traced both remaining real-corpus failures end-to-end through the raw
telemetry rather than re-attempting semantic-similarity fixes. One of the two turned out to be a
genuine, previously-mischaracterised bug, not the segmentation issue originally attributed to it.

**Symptom**: a refinement turn ("Wait, one more thing — What does stong force act upon?" — a real
typo in the SQuAD source, "stong" for "strong") produced **two** `retrieval_completed` events for
what should have been one. The controller re-emits a fresh sub-query on every chunk as a clause
grows; the first, partial version ("...one more thing — What", four near-content-free words)
retrieved an essentially random Nikola Tesla biography passage and was never cancelled — its
wrong-topic evidence reached the final answer alongside the correct one.

**Root cause**: `controller/controller.py` has two supersession mechanisms sharing one field,
`SubQuery.parent_query_id` — `_scope_to_delta` links a sub-query to a *prior turn's* query (for
refinement bookkeeping); `_link_supersession` links it to a *stale sibling in the same
utterance* (for engine-side cancellation, the same mechanism from Pass 2 fix 2 above).
`_scope_to_delta` runs first and sets `parent_query_id`; `_link_supersession` saw a non-empty
value and silently deferred to it — even though the two links mean different things, and the
engine's cancellation check (`engine.py`, `state.pending`) only ever looks up ids within the
*current* utterance, so a cross-turn id sitting in that field was always a harmless no-op there.
The Pass-2 fix cancelled stale siblings correctly for `new_request` turns; it never fired for
`refinement` turns specifically, because `_scope_to_delta` runs only on refinement turns and
always claims the field first.

**Fix**: `_link_supersession` now always checks for a same-utterance stale sibling and overrides
any cross-turn link if one is found — verified safe, since overriding a value that was already a
no-op for cancellation can only enable a cancellation that was previously silently skipped.

**Verified**: `retrieval_cancelled` now fires for the stale partial-query retrieval on this exact
turn; the Tesla-biography passage no longer reaches the answer; the turn is grounded with zero
uncertainty. Full regression suite (77 tests, both gate suites, official `--reps 3`) shows zero
side effects — dev_corpus unchanged at 100% across all gates.

```
$ python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 3 --corpus-dir data/corpus
G2 early retrieval : 11/11 = 100.0%  (false-trigger rate 0.0%)
G3 multi-intent    : 5/5   = 100.0%
G4 grounding       : 39/40 = 97.5%  (fabricated=0)
G5 refinement      : 5/5   = 100.0%
G6 telemetry cov.  : 25/25 = 100.0%  (schema errors=0)
VERDICT: PASS
```

The denominator dropped 42→40 as a direct consequence of the fix: the cancelled retrieval had
been contributing a spurious extra claim, so the system is now being asked one fewer (wrong)
question, not scored more leniently on the same set.

**The remaining case** (`gen_multi_intent_003`) is unrelated to this bug: re-traced after the fix,
decomposition and cancellation are both working correctly for that turn, and the retriever
genuinely never surfaces the correct chunk (Doc_04 §13) in its top-8 for a query
("what geometric shape...") that shares zero vocabulary with its answer ("...the parallelogram
rule..."). This is the retrieval-recall ceiling discussed in §0b/§7 — closing it needs a trained
extractive QA model or an LLM doing real reasoning, not more lexical tuning. Three prior attempts
at a semantic-similarity fix for this exact case (word-vector encoder swap, scoped deeper search,
same-document semantic tie-break) were each tested and reverted after being found to regress
other cases — see `final_report.docx` §6.5 for the full account of those attempts.

## 0e. Pass 5 — missing wh-word stopwords (self-fulfilling grounding, concretely reproduced)

A hand-run end-to-end example ("What disease did Tesla catch?", `final_report.docx` §9.3) surfaced
a wrong, confidently-asserted answer citing an unrelated Tesla passage — a live instance of the
external review's "the grounding gate is nearly self-fulfilling" critique, not a hypothetical one.

**Root cause**: `streaming_rag/retrieval/text.py`'s `STOPWORDS` set was missing "what", "who",
"which", "whom", "whose". `_is_relevant()`'s lexical-overlap safety-net gate counted "what" as a
real content token, so the shared-token set between the question and the wrong passage —
`{"what", "tesla"}` — was enough to clear the gate's minimum-overlap threshold, despite neither
token carrying any signal about *disease*.

**Fix**: added the five missing wh-words to `STOPWORDS` (`streaming_rag/retrieval/text.py`, one
line).

**Verified**: 77/77 tests pass; dev_corpus gate suite unaffected (100% across all gates, no
regression); real_corpus gate suite:

```
$ python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 3 --corpus-dir data/corpus
G2 early retrieval : 11/11 = 100.0%  (false-trigger rate 0.0%)
G3 multi-intent    : 5/5   = 100.0%
G4 grounding       : 36/37 = 97.3%  (fabricated=0)
G5 refinement      : 5/5   = 100.0%
G6 telemetry cov.  : 25/25 = 100.0%  (schema errors=0)
VERDICT: PASS
```

The denominator dropped 40→37: three over-confident false-positive claims that used to slip past
the relevance gate on a wrongly-counted wh-word are now correctly suppressed into `uncertainty`
instead of being asserted. `--reps 1` and `--reps 3` produced identical counts, confirming
determinism. The single previously-known scenario (`gen_multi_intent_003`, the "parallelogram"
zero-vocabulary-overlap case from §0d) remains the only tracked failure in the scored suite — no
new regressions.

**A second issue found, not fixed**: re-running the same "Tesla disease" example after this fix,
the wrong citation changed but did not disappear — `decompose.py`'s short-clause context-carrying
logic (triggered for any clause under 4 content tokens) was found to splice an unrelated sibling
clause's context into this legitimately-3-content-word question, pulling retrieval toward a
different wrong Tesla passage. This scenario is not part of the scored 15-scenario suite, so it
produced no gate regression, and was deliberately left unfixed given this project's established
pattern (§0b–§0d) that further tuning in this exact category (short-clause / vocabulary-gap
precision) has repeatedly traded one failure mode for another. See `final_report.docx` §6.6.2 for
the full account, including direct corpus verification of the true correct answer.

## 1. Test suite

`pytest tests/ -q` → **72/72 passed** (contract conformance, engine E2E with mocks, robustness
& fault injection, gate-scorer unit tests, ground-truth firewall, telemetry schema/coverage,
scenario-generator determinism, retrieval/controller/session unit tests).

## 2. Corpora

| Corpus | Source | Docs | Chunks | Qrels | License |
|---|---|---|---|---|---|
| `fixtures/dev_corpus/` | Synthetic, seeded (`fixtures/build_dev_corpus.py`) | 41 | 155 | 156 (lexical + paraphrase) | N/A (generated) |
| `data/corpus/` | **SQuAD v1.1 validation** (`data/build_squad_corpus.py`) | 14 real Wikipedia articles | 1,098 | 3,458 | CC BY-SA 4.0 |

The SQuAD-derived corpus exists to prove the format-driven ingestion pipeline (§5 of the
architecture brief) behaves the same on real, unseen content, not just the tuned synthetic
fixture — and to benchmark retrieval against real gold evidence. It is not the hidden
evaluation corpus.

## 3. Retrieval quality (standalone, against qrels)

`fixtures/dev_corpus` (156 qrels, split lexical vs paraphrase):

| mode | lexical r@1 | lexical r@3 | paraphrase r@1 | paraphrase r@3 |
|---|---|---|---|---|
| sparse (BM25) | 83.7% | 97.7% | 51.4% | 74.3% |
| dense (LSA) | 83.7% | 88.4% | 48.6% | 74.3% |
| **hybrid (sw=0.99/dw=0.01, tuned)** | 83.7% | **97.7%** | 51.4% | 74.3% |

The dense-only mode never beats hybrid on this corpus (a plain LSA encoder is a weak learner at
this scale), so the tuned fusion weight keeps hybrid at parity with sparse on exact-phrase
queries while still contributing on paraphrases via RRF rank-fusion; recall here is identical
across every dense weight tested (0.15 down to 0.01), which is exactly why the weight was
re-tuned down on the real corpus, where it wasn't harmless. See `docs/architecture_brief.docx`
§5 and §10 for the trade-off.

`data/corpus` (SQuAD, 500-qrel sample; real questions written by independent annotators, not
us) — **before and after the fusion-weight retune**:

| mode | r@1 (before) | r@1 (after) | r@3 (after) | r@5 (before) | r@5 (after) |
|---|---|---|---|---|---|
| sparse | 61.8% | 62.6% | 81.2% | 86.2% | 86.6% |
| dense | 55.8% | 57.2% | 71.4% | 76.0% | 77.6% |
| **hybrid** | 62.0% | **63.2%** | **81.0%** | 84.0% | **86.8%** |

(the "before"/"after" r@1/r@5 columns come from independent 400- and 500-sample draws of the
same qrels pool, hence the small sparse/dense drift too; hybrid's improvement over sparse-only —
previously a virtual tie — is the metric that matters here.)

## 4. Gate report (`python -m eval.run_all`)

**dev_corpus suite** (47 scenarios: 2 hand-written + 45 generated across all 5 templates).
`--reps 1` and the **official `--reps 3` procedure produced byte-identical results**
(current, post Pass-2, run):

```
$ python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 3 --corpus-dir fixtures/dev_corpus
scenarios run: 47
G2 early retrieval : 16/16 = 100.0%  (false-trigger rate 0.0%)
G3 multi-intent    : 10/10 = 100.0%
G4 grounding       : 79/79 = 100.0%  (fabricated=0)
G5 refinement      : 10/10 = 100.0%
G6 telemetry cov.  : 67/67 = 100.0%  (schema errors=0)
VERDICT: PASS
```

**real_corpus suite** (15 scenarios generated against `data/corpus`). `--reps 1` and the
**official `--reps 3` procedure again produced identical results** (current, post Pass-3, run):

```
$ python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 3 --corpus-dir data/corpus
scenarios run: 15
G2 early retrieval : 11/11 = 100.0%  (false-trigger rate 0.0%)
G3 multi-intent    : 5/5   = 100.0%
G4 grounding       : 39/40 = 97.5%  (fabricated=0)
G5 refinement      : 5/5   = 100.0%
G6 telemetry cov.  : 25/25 = 100.0%  (schema errors=0)
VERDICT: PASS
```

Both suites clear every merge-to-main threshold from Task 4 §4.8 (G2≥80%, G3≥70%, G4≥85%/0
fabricated, G5=100%, G6=100%), and both originally-requested >95% targets — see §0/§0b/§0c for
the full optimization delta that got the real-corpus suite here from its pre-tuning baseline. The
real-corpus G4 number is still slightly lower than the tuned dev fixture (100%) — real questions
are noisier and the corpus was never tuned against them — which is the point of running it: the
gates clear comfortably with margin, on content the system has never seen, under the exact
official `--reps 3` median-of-3 procedure.

Determinism is proven, not assumed: `--reps 1` and `--reps 3` landing on the *exact same*
per-gate counts on both suites is only possible because every decision path is free of
randomness (`temperature=0`, a seeded LSA fit, no wall-clock-dependent branching) — a flaky
system would show `--reps 3`'s median diverge from a single `--reps 1` run at least some of
the time across 47+15 scenarios.

## 5. Baseline vs. streaming comparison (`python -m eval.compare`)

Run with `--simulated-latency-ms 300` (models a realistic hosted retrieval backend — the raw
BM25/LSA search on this small corpus is sub-millisecond and would make both modes look
identical regardless of the algorithmic difference being measured):

```
          mode | ttfr_p50 | ttfr_p95 | e2e_p50 | e2e_p95 | total_p50 | total_p95 |  G2  |  G3  |  G4  |  G5
     streaming |     0.0  |  2400.0  |  141.0  |  244.8  |   3687.5  |   8926.8  | 100% | 100% | 100% | 100%
      baseline |  3500.0  |  8300.0  |  312.0  |  328.0  |   3797.0  |   8618.7  |   0% |   0% | 100% |   0%

streaming utterance-start->answer p50 = 3687.5ms vs baseline 3797.0ms (faster by 109.5ms)
post-utterance-end e2e p50: streaming 141.0ms vs baseline 312.0ms
PASS CONDITION: MET
```

Reading it: **ttfr** (time to first retrieval, relative to utterance start) is the clearest
signal — streaming starts searching essentially as soon as an entity is stable (median 0ms
relative to the *first* chunk, since Example-1-style utterances trigger fast), baseline never
starts before the full utterance (median 3500ms, matching the guide's "pauses of several
seconds" complaint). The **post-utterance-end** latency (what the user perceives as "the pause
after I stopped talking") is ~2.2x lower for streaming (141ms vs 312ms) because its retrieval
already ran during speech. G2/G3/G5 are structurally 0% for baseline by construction — it never
attempts early retrieval, decomposition, or refinement, which is exactly the capability gap
this project closes.

## 6. Ablations

### Ablation 1 — `retrieval.mode`: hybrid vs sparse vs dense

Run on the 47-scenario dev suite (post-optimization config: `sw=0.99/dw=0.01`, query-relevance
gate active — n_claims is lower than the pre-optimization 117 because the relevance gate now
filters weak-evidence claims before they're ever asserted):

```
hybrid  G4_support_rate=100.0%  fabricated=0  n_claims=69
sparse  G4_support_rate=100.0%  fabricated=0  n_claims=69
dense   G4_support_rate=100.0%  fabricated=0  n_claims=69
```

Downstream grounding support is unaffected by retrieval mode here because the synthesizer takes
only the top-ranked chunk per sub-query and marks anything below the low-confidence threshold as
uncertainty rather than fabricating — the effect of a weaker retrieval mode shows up as *more
uncertainty flags*, not *worse grounding*, which is the intended failure mode (§3's standalone
recall numbers are where the retrieval-mode difference is actually visible: sparse's r@3 on
lexical queries is 97.7% vs dense's 88.4%).

### Ablation 2 — `controller.mode`: rule vs model

```
rule   G2_early_rate=100.0%  G3_rate=100.0%
model  not run — requires LLM_PROVIDER != fake (see run.yaml / .env.example)
```

The rule-based controller (`controller/stability.py`, `controller/decompose.py`) clears G2/G3
without any LLM calls — zero marginal latency or token cost per decision, consistent with the
guide's "architectural parsimony" rule (prefer a fast heuristic path, LLM only where judgment
is needed). `controller.mode=model` is wired through `RetrievalController.__init__` and ready
to enable with a real `LLM_PROVIDER`/`SECRET_LLM_API_KEY`; it was not run in this offline
environment.

## 7. Edge-case analysis (≥ 3, root cause → component → mitigation)

**1. [FIXED] Grounding shortfall on gold answers embedded in mid-quote fragments**
(`eval/scenarios_real_corpus`, on `data/corpus`).
*Symptom*: `uncertainty_flagged` for a sub-query whose gold SQuAD answer sits inside a sentence
fragment like `"What can I say?" Tesla may have inadvert[ently]...` — the regex sentence splitter
broke on `.!?` inside the quote, producing a spurious tiny "sentence" that outranked the real
answer sentence in `_best_sentence`'s coverage score.
*Root cause*: naive sentence segmentation doesn't understand nested quotation.
*Component*: `retrieval/text.py::split_sentences`.
*Fix applied*: swapped in spaCy's rule-based `sentencizer` (blank pipeline, no model download,
deterministic), which correctly keeps `Tesla once wrote, "What can I say?"` as one sentence.
*Measured effect*: **G4 88.9% → 90.5%** on the real-corpus suite.

**2. [FIXED] Within-utterance retrieval supersession never cancelled a growing clause's stale
partial retrievals** (`eval/scenarios_real_corpus/gen_late_detail_004`-equivalent pattern).
*Symptom*: a single clause growing across chunks ("Before which..." → "...did Chagatai
publicly" → "...dispute Jochi's paternity?") issued a **new, uncancelled** sub-query on every
qualifying chunk. The first partial version ("Before which") retrieved a topically wrong chunk
(a different article entirely); its evidence still reached the synthesizer alongside the final,
correct one, producing an extra wrong-topic claim.
*Root cause*: two compounding bugs — (a) `decompose()`'s "already covered" tracking only
prevents re-emission of *fully* covered clauses, so a clause gaining new words every chunk is
always "new" and gets re-queried; (b) even where `parent_query_id` cancellation existed
(cross-turn refinement), `engine.py::_cancel_task` silently no-op'd when the superseded task had
*already completed* — the common case for a fast local retriever — leaving its stale evidence in
`state.completed_results`.
*Components*: `controller/controller.py`, `engine.py`.
*Fix applied*: `_link_supersession()` links a new candidate to the most recent sub-query it
overlaps heavily with from the same utterance; `_cancel_task` now cleans up completed-but-stale
evidence, not just still-running tasks.
*Measured effect*: **G4 90.5% → 92.7%**; claim count dropped 63→41 across the suite (fewer,
correctly-scoped claims).

**3. [PARTIALLY FIXED] G2 early-retrieval rate drop on the real corpus**
(`eval/scenarios_real_corpus`).
*Symptom*: eligible turns not triggering retrieval before `utterance_end`.
*Root cause*: the strong-anchor check looks for a number or a capitalised non-leading word;
several SQuAD-derived queries reference abstract topic nouns with no such cue until the sentence
is nearly complete — correct conservatism against the "premature, noisy searches" pitfall, but
overly conservative when the noun is a real, specific term the corpus already indexes.
*Component*: `controller/stability.py`.
*Fix applied*: `has_topic_anchor()` treats a discriminative corpus term (high-idf, via
`BM25Index.specific_terms()`) as a valid anchor even without capitalisation, injected through a
`corpus_vocab` callable so the controller stays decoupled from the `Retriever` protocol.
*Measured effect*: **G2 86.7% → 93.3%**, false-trigger rate unchanged at 0.0%.
*Remaining gap*: one scenario still waits through the full utterance — its accumulated text
never clears the stability threshold even at `is_final` in one intermediate branch; not
re-investigated further given G2's already-comfortable 93.3% vs the 80% gate threshold.

**4. [OPEN, by design] Retrieval recall ceiling on genuinely hard/ambiguous real questions.**
*Symptom*: ~3 sub-queries per 15-scenario real-corpus run still resolve to `uncertainty_flagged`
even after (1)-(3) above and a query-relevance gate (`synthesis.py::_is_relevant`) that requires
real, non-trivial term overlap before a retrieved sentence becomes a claim (catching, e.g.,
"Tesla" trivially matching any sentence in a Tesla biography).
*Root cause*: for a handful of real, independently-authored SQuAD questions, the correct
evidence is not recoverable in the top-8 candidates from a lightweight local BM25+LSA retriever
— a genuine capability ceiling of this component, distinct from bugs (1)-(3).
*Component*: `retrieval/hybrid.py` (capability), `session/synthesis.py` (correctly declines).
*Mitigation*: none applied — this is the intended safety behavior (uncertainty over fabrication)
rather than a defect; the `Embedder` protocol swap point exists for a stronger encoder to close
this gap without touching any other layer.

## 8. Cost & telemetry overhead

The `fake` LLM provider used throughout this report has a `(0.0, 0.0)` price table entry, so
`cost_per_turn` is $0 in every run above. Real-provider cost accounting is exercised by
`llm.py::OpenAILLM`/`GeminiLLM`, which emit one `llm_call` event per call with
`tokens_in`/`tokens_out`/`cost_usd` computed from `config.llm.price_table` — never hardcoded per
call site. Telemetry itself is asynchronous and buffered (`JsonlTelemetry`), so `emit()` never
blocks the event loop.
