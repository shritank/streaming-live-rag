# Demo Video Script (≤ 5 minutes)

Six required demonstrations (guide's deliverables checklist), one command per beat.

## 1. Early retrieval triggering (0:00-0:45)

```bash
python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --time-scale 1 --telemetry trace.jsonl
python -m streaming_rag.telemetry.report trace.jsonl
```
Narrate: WAIT at t=0.0s (no anchor yet), RETRIEVE at t=0.8s the moment "Pune"/"30" appear —
before the user has finished the sentence.

## 2. Multi-intent decomposition (0:45-1:30)

Point at the same trace's t=1.6s line: one utterance ("...cancellation policy and the
catering options") becomes 2 parallel sub-queries with distinct `intent_label`s, each with its
own citation in the final answer.

## 3. Late-detail refinement (1:30-2:30)

```bash
python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json --time-scale 1
```
Show turn 1 (Version 1, one citation) -> turn 2 ("Actually, the trip was international...",
Version 2, `parent_version=1`, the original citation still present alongside two new ones) ->
turn 3 (bullet reformat, zero new retrieval, same citations).

## 4. Presentation query suppression (2:30-3:00)

Same replay, turn 3: point at `retrieval_events: []` and `retrieval_required: false,
reason: "presentation_restructure"` in the JSON output — no vector search ran.

## 5. Citation traceability (3:00-3:45)

Open `data/corpus/Doc_00_amazon_rainforest.md` (or any dev_corpus doc) side by side with a
`turn_result`'s `citations` field and show the cited `§Section` contains the asserted fact.
Then run:
```bash
python -m eval.run_all --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus --time-scale 8 --reps 1
```
and point at `G4 grounding: .../... = 100.0% (fabricated=0)`.

## 6. Runtime telemetry (3:45-4:30)

```bash
python -m eval.compare --modes streaming,baseline --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus --simulated-latency-ms 300
```
Show the printed table: streaming's post-utterance-end latency (~100ms) vs baseline's
(~300ms+), and the "PASS CONDITION: MET" line.

## Close (4:30-5:00)

```bash
python -m eval.run_all --scenarios eval/scenarios_real_corpus --corpus-dir data/corpus --time-scale 8 --reps 1
```
Same system, a real Wikipedia-derived corpus (SQuAD v1.1) it has never been tuned against,
all six gates still pass.
