"""Runs one scenario through a fresh engine instance and returns both the
turn_results and the raw telemetry trace, which is all the gate scorers
need (they never touch ground_truth directly outside this module's caller).
"""
from __future__ import annotations

import asyncio

from streaming_rag.build import build_components
from streaming_rag.config import Config
from streaming_rag.engine import Engine
from streaming_rag.session.store import SessionStore
from streaming_rag.telemetry.sinks import BufferedTelemetry

from .loader import load_events


async def _feed(events: list[dict], input_queue: "asyncio.Queue", time_scale: float) -> None:
    last_ts = 0
    for e in events:
        ts = e["timestamp_ms"]
        delay = max(0, ts - last_ts) / 1000 / max(time_scale, 1e-9)
        if delay:
            await asyncio.sleep(delay)
        last_ts = ts
        await input_queue.put(e)
    await input_queue.put(None)


class _RecordingSynthesizer:
    """Eval-only wrapper: records every AnswerVersion the synthesizer returns,
    keyed by (session_id, version), so quality scoring can inspect individual
    claims and their citations. The turn_result record itself stays exactly
    the guide's §5 shape; nothing in streaming_rag/ changes."""

    def __init__(self, inner, sink: dict, calls: list | None = None):
        self._inner = inner
        self._sink = sink
        self._calls = calls

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _record(self, session_id, answer, sub_queries=None, results=None):
        if answer is not None:
            self._sink[(session_id, answer.version)] = answer
            if self._calls is not None and sub_queries is not None:
                self._calls.append({"session_id": session_id, "version": answer.version,
                                    "sub_queries": list(sub_queries), "results": list(results)})
        return answer

    async def synthesize(self, session_id, utterance, sub_queries, results):
        answer = await self._inner.synthesize(session_id, utterance, sub_queries, results)
        return self._record(session_id, answer, sub_queries, results)

    async def refine(self, session_id, utterance, sub_queries, results):
        answer = await self._inner.refine(session_id, utterance, sub_queries, results)
        return self._record(session_id, answer, sub_queries, results)

    async def restructure(self, session_id, *args, **kwargs):
        return self._record(session_id, await self._inner.restructure(session_id, *args, **kwargs))


async def run_scenario_detailed(scenario_path: str, config: Config, impl: str = "real",
                                 time_scale: float = 8.0, calls: list | None = None
                                 ) -> tuple[list[dict], list[dict], dict]:
    """Returns (turn_results, trace_events, answers) where answers maps
    (session_id, version) -> AnswerVersion. If `calls` is given, every
    synthesize/refine call's inputs (sub-queries, retrieval results) are
    appended to it for error analysis."""
    answers: dict = {}
    results, trace = await _run(scenario_path, config, impl, time_scale, answers, calls)
    return results, trace, answers


async def run_scenario(scenario_path: str, config: Config, impl: str = "real",
                        time_scale: float = 8.0) -> tuple[list[dict], list[dict]]:
    """Returns (turn_results, trace_events) for one replay."""
    return await _run(scenario_path, config, impl, time_scale, None)


async def _run(scenario_path: str, config: Config, impl: str, time_scale: float,
               answer_sink: dict | None, calls: list | None = None) -> tuple[list[dict], list[dict]]:
    events = load_events(scenario_path)
    telemetry = BufferedTelemetry()
    session_store = SessionStore()
    controller, retriever, synthesizer, llm, session_store = build_components(
        config, telemetry, session_store, impl=impl)
    if answer_sink is not None:
        synthesizer = _RecordingSynthesizer(synthesizer, answer_sink, calls)
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry,
                     session_store=session_store)

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()
    results: list[dict] = []

    async def drain():
        while True:
            item = await output_queue.get()
            if item is None:
                break
            results.append(item)

    async def run_and_close():
        await engine.run(input_queue, output_queue)
        await output_queue.put(None)

    consumer = asyncio.create_task(drain())
    producer = asyncio.create_task(_feed(events, input_queue, time_scale))
    await asyncio.gather(producer, run_and_close())
    await consumer

    return results, telemetry.events


async def run_scenario_median(scenario_path: str, config: Config, impl: str = "real",
                               time_scale: float = 8.0, reps: int = 3):
    """Runs the scenario `reps` times with fresh engine instances and returns
    the median run by trace length (project context §9.8: reps & median)."""
    runs = [await run_scenario(scenario_path, config, impl, time_scale) for _ in range(reps)]
    runs.sort(key=lambda r: len(r[1]))
    return runs[len(runs) // 2]
