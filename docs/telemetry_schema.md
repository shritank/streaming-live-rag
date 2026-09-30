# Telemetry & Observability Schema

Canonical machine-readable schema: [`streaming_rag/telemetry/schema.json`](../streaming_rag/telemetry/schema.json)
(JSON Schema draft-07, validated by `eval/schema_check.py` and enforced as part of G6).

## Common fields (every event)

| field | type | meaning |
|---|---|---|
| `event` | string (enum) | event name, see table below |
| `ts_ms` | number \| null | **virtual** (scenario) clock — identical at any `--time-scale` |
| `wall_ms` | number | real monotonic time since the sink was created (debugging only) |
| `session_id` | string \| null | |
| `utterance_id` | string \| null | null for session-scoped events (`session_started`/`session_ended`) |
| `component` | string | `engine` \| `controller` \| `retriever` \| `synthesizer` \| an LLM client |

## Events

| Event | Emitted by | Required extra fields |
|---|---|---|
| `session_started` / `session_ended` | engine | — |
| `chunk_received` | engine | `text_len`, `is_final` |
| `controller_decision` | controller | `decision`, `turn_kind`, `reason`, `stability`, `n_sub_queries`, `decision_latency_ms` |
| `sub_queries_emitted` | controller | `query_ids`, `queries`, `trigger` |
| `retrieval_started` | retriever | `request_id`, `query_ids`, `mode` |
| `retrieval_completed` | retriever | `request_id`, `latency_ms`, `chunk_ids`, `low_confidence_query_ids` |
| `retrieval_cancelled` | engine | `request_id`, `query_ids`, `reason` (e.g. `superseded`) |
| `answer_version_created` | synthesizer | `version`, `parent_version`, `change_kind`, `citations`, `n_claims`, `retriever_calls_for_version` |
| `grounding_checked` | synthesizer | `version`, `n_claims`, `n_supported`, `fabricated_citations` |
| `uncertainty_flagged` | synthesizer | `version`, `unsupported_intents` |
| `llm_call` | LLM client | `purpose`, `model`, `tokens_in`, `tokens_out`, `latency_ms`, `cost_usd` |
| `output_emitted` | engine | `kind` (`turn_result`), `version`; optional `timings`, `speculation` (below) |
| `error` | any component | `where`, `error_type`, `message` |

## Latency breakdown and speculation fields (on `output_emitted`)

Extra, optional fields (the schema requires only `kind` and `version`, so older traces stay valid):

| field | meaning |
|---|---|
| `timings.post_speech_ms` | what the user waits after they stop: `utterance_end` envelope handled -> turn result emitted |
| `timings.final_decision_ms` | the controller's final pass on the (empty) end-of-utterance chunk |
| `timings.retrieval_wait_ms` | waiting for retrievals / claim selection still in flight at utterance end |
| `timings.synthesis_ms` | `synthesize` / `refine` / `restructure` call |
| `timings.answer_telemetry_ms` | grounding telemetry and answer bookkeeping |
| `speculation.launched` / `.reused` | retrievals started **during speech** for what the controller's final pass was about to issue (`engine.speculative_final`), and how many of them the real decision adopted |

All intervals are `time.perf_counter()` milliseconds (the Windows `time.monotonic()` clock ticks every
16 ms and cannot resolve 5 ms). `eval/latency_profile.py` aggregates them into p50 / p95 / p99 / max.

**How speculative work appears in the trace.** A speculative retrieval is not traced on its own (it
may be cancelled before it means anything). When the real controller decision later emits a sub-query
with the *identical text*, the engine adopts the speculative result and emits that sub-query's normal
`retrieval_started` / `retrieval_completed` pair (same `request_id`, `latency_ms` = the time spent
waiting for it), so G6's pairing rule and the sub-query -> chunk_id -> citation mapping are unchanged.
Anything not adopted is cancelled silently and only shows in the `launched` counter.

## Reconstructing a turn from the trace alone (G6)

`eval/gates/g6_coverage.py` and `streaming_rag/telemetry/trace.py::assemble_turns` group a flat
trace into one record per `(session_id, utterance_id)`. For a turn to count as "covered", the
trace alone must let you reconstruct:

1. every chunk timestamp (`chunk_received`)
2. every controller decision (`controller_decision`)
3. matched `retrieval_started`/`retrieval_completed` pairs by `request_id` (no orphans)
4. the sub-query -> chunk_id -> citation mapping
5. answer version lineage with no gaps (`parent_version` chains back to an existing version)
6. token cost per LLM call and per turn (`llm_call.cost_usd`, summed)
7. end-to-end latency (`utterance_end` -> `turn_result`, via `output_emitted.ts_ms`)

Causal ordering is checked separately: `ts_ms` is non-decreasing per component within a
session, `retrieval_completed` always follows its `retrieval_started`, and
`answer_version_created` always follows the retrievals it used.

## Cost model

`streaming_rag/telemetry/cost.py`: `cost_usd = tokens_in/1000 * price_in + tokens_out/1000 * price_out`,
looked up from `config.llm.price_table` (never hardcoded per-call). The `fake` provider's price
table entry is `(0.0, 0.0)`, so local replay is always free; real providers are priced from a
config table so switching models doesn't require a code change.

## What the trace does NOT contain

Which execution provider each neural model ran on is a property of the run, not of a turn: it is
recorded in every `eval.run_quality` / `eval.latency_profile` output (`execution_providers`) and
enforced by `streaming_rag/gpu.py` (CUDA is mandatory whenever an NVIDIA GPU is present).

## Dashboards

- **Text**: `python -m streaming_rag.telemetry.report trace.jsonl` — per-turn timeline, final
  answer version + citations, cost.
- **Coverage / schema report**: `python -m eval.run_all` prints G6 coverage rate and any schema
  validation errors inline as part of the gate report.
