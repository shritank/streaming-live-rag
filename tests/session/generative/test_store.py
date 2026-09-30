"""Task context §7.6 — session isolation and ephemerality."""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from streaming_rag.contracts import AnswerVersion, Claim
from streaming_rag.session.generative.store import UnknownSessionError

from .fakes import result


def version(n: int, parent: int | None, text: str = "t") -> AnswerVersion:
    return AnswerVersion(version=n, parent_version=parent, change_kind="initial",
                         text=text, claims=[Claim(text=text, citations=["Doc_1 §1"])],
                         citations=["Doc_1 §1"], evidence_ids=[], sub_queries=[],
                         uncertainty=None, created_ms=0)


def test_unknown_session_raises(store):
    with pytest.raises(UnknownSessionError):
        store.view("never-opened")


def test_close_wipes_everything(store, chunks):
    store.add_turn("s1", "hello")
    store.add_evidence("s1", [result("u1.q1", [chunks["capacity"]])])
    store.add_version("s1", version(1, None))
    assert store.view("s1").has_answer()

    store.close("s1")

    assert not store.is_open("s1")
    with pytest.raises(UnknownSessionError):
        store.view("s1")
    store.close("s1")  # closing twice is safe


def test_sessions_are_isolated(store, chunks):
    store.open("s2")
    store.add_evidence("s1", [result("u1.q1", [chunks["capacity"]])])
    store.add_evidence("s2", [result("u1.q1", [chunks["catering"]])])
    store.add_turn("s1", "first session question")
    store.add_turn("s2", "second session question")

    assert store.view("s1").topic_summary() == "first session question"
    assert store.evidence_for("s1").keys() != store.evidence_for("s2").keys()
    assert "Doc_09 §1" not in {ev.chunk.citation for ev in store.evidence_for("s1").values()}


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_leak(store, chunks):
    store.open("a")
    store.open("b")

    async def fill(session_id: str, c):
        for _ in range(20):
            store.add_evidence(session_id, [result("q", [c])])
            await asyncio.sleep(0)

    await asyncio.gather(fill("a", chunks["capacity"]), fill("b", chunks["catering"]))

    a_citations = {ev.chunk.citation for ev in store.evidence_for("a").values()}
    b_citations = {ev.chunk.citation for ev in store.evidence_for("b").values()}
    assert a_citations == {"Doc_12 §2"}
    assert b_citations == {"Doc_09 §1"}


def test_versions_are_append_only_and_sequential(store):
    store.add_version("s1", version(1, None, "v1"))
    first = store.latest_version("s1")
    store.add_version("s1", version(2, 1, "v2"))

    assert first.text == "v1"            # the stored V1 object was not mutated
    assert store.latest_version("s1").version == 2
    assert store.next_version_number("s1") == 3
    with pytest.raises(ValueError):
        store.add_version("s1", version(9, 2))


def test_evidence_merge_keeps_best_score_and_all_query_ids(store, chunks):
    store.add_evidence("s1", [result("q1", [chunks["capacity"]], scores=[0.4])])
    store.add_evidence("s1", [result("q2", [chunks["capacity"]], scores=[0.9])])

    cached = store.evidence_for("s1")
    assert len(cached) == 1
    only = next(iter(cached.values()))
    assert only.score == 0.9
    assert set(only.query_ids) == {"q1", "q2"}


def test_low_confidence_tracking(store, chunks):
    store.add_evidence("s1", [result("q1", [], low_confidence=True)])
    assert store.state("s1").low_confidence_query_ids == {"q1"}
    store.add_evidence("s1", [result("q1", [chunks["capacity"]])])
    assert store.state("s1").low_confidence_query_ids == set()


def test_no_disk_writes(store, chunks, monkeypatch):
    """Session state is in-process only (hard rule: session-bound state)."""
    writes: list[str] = []
    real_open = open

    def spy_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy_open)
    monkeypatch.setattr(pathlib.Path, "write_text",
                        lambda self, *a, **k: writes.append(str(self)))
    monkeypatch.setattr(pathlib.Path, "write_bytes",
                        lambda self, *a, **k: writes.append(str(self)))

    store.add_turn("s1", "question")
    store.add_evidence("s1", [result("q1", [chunks["capacity"]])])
    store.add_version("s1", version(1, None))
    store.close("s1")

    assert writes == []


def test_topic_summary_includes_intent_labels(store, sub_queries):
    store.add_turn("s1", "I need a hall and the refund rules")
    store.add_sub_queries("s1", sub_queries)
    summary = store.view("s1").topic_summary()
    assert "venue capacity" in summary and "cancellation terms" in summary


def test_view_is_read_only_snapshot_of_sub_queries(store, sub_queries):
    store.add_sub_queries("s1", sub_queries)
    view = store.view("s1")
    view.prior_sub_queries().clear()
    assert len(store.view("s1").prior_sub_queries()) == 3
