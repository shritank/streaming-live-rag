"""Telemetry sinks.

Every component calls `telemetry.emit(event, **fields)` synchronously (the
Telemetry protocol is sync on purpose: it must never block the event loop
waiting on I/O). JsonlTelemetry buffers events in memory and flushes them
to disk from a background asyncio task, so `emit()` itself never does I/O.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ..contracts import NullTelemetry as NullTelemetry  # re-exported, single source of truth

__all__ = ["JsonlTelemetry", "NullTelemetry", "BufferedTelemetry"]

_COMMON_FIELDS = ("event", "ts_ms", "wall_ms", "session_id", "utterance_id", "component")


class BufferedTelemetry:
    """In-memory sink. Used by tests and by the dashboard/coverage checker
    when it wants to inspect events without touching a file."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self._start_wall = time.perf_counter()

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "wall_ms": (time.perf_counter() - self._start_wall) * 1000,
            "component": fields.pop("component", "unknown"),
            "session_id": fields.pop("session_id", None),
            "utterance_id": fields.pop("utterance_id", None),
            "ts_ms": fields.pop("ts_ms", None),
        }
        record.update(fields)
        self.events.append(record)


class JsonlTelemetry:
    """Async-safe JSON-lines sink. `emit()` is non-blocking: it appends to an
    in-memory buffer; a background task periodically flushes to disk.

    Call `await start()` once the event loop is running and `await stop()`
    (or `await flush()`) before reading the file, so no event is lost.
    """

    def __init__(self, out_path: str | Path, buffer_size: int = 50):
        self._path = Path(out_path)
        self._buffer_size = buffer_size
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._start_wall = time.perf_counter()
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        # truncate any previous run's file up front so replays don't append forever
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "wall_ms": (time.perf_counter() - self._start_wall) * 1000,
            "component": fields.pop("component", "unknown"),
            "session_id": fields.pop("session_id", None),
            "utterance_id": fields.pop("utterance_id", None),
            "ts_ms": fields.pop("ts_ms", None),
        }
        record.update(fields)
        self._buffer.append(record)
        if len(self._buffer) >= self._buffer_size and self._task is not None:
            # nudge an immediate flush without blocking the caller
            asyncio.create_task(self.flush())

    async def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while not self._stopping.is_set():
            await self.flush()
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

    async def flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            pending, self._buffer = self._buffer, []
        lines = "\n".join(json.dumps(r, default=str) for r in pending) + "\n"
        await asyncio.to_thread(self._append, lines)

    def _append(self, lines: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(lines)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            await self._task
        await self.flush()
