from __future__ import annotations

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Chunk, SubQuery
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.retrieval.ingest import parse_document


CHUNKS = [
    Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1",
          text="The cancellation policy allows a full refund up to 72 hours before the event."),
    Chunk(chunk_id="Doc_02 §1 #1", doc_id="Doc_02", section="1",
          text="Venue A supports a workshop capacity of 40 attendees."),
    Chunk(chunk_id="Doc_03 §1 #1", doc_id="Doc_03", section="1",
          text="Catering options include vegetarian and vegan meals."),
]


def _q(text: str) -> SubQuery:
    return SubQuery(query_id="q1", text=text, intent_label="t", trigger="final", utterance_id="u1")


@pytest.mark.parametrize("mode", ["hybrid", "sparse", "dense"])
async def test_search_returns_relevant_chunk_first(mode):
    cfg = load_config(); cfg.retrieval.mode = mode
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    results = await retriever.search([_q("what is the cancellation refund policy")], k=3)
    assert len(results) == 1
    assert results[0].evidence[0].chunk.doc_id == "Doc_01"


async def test_search_is_deterministic_across_calls():
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    r1 = await retriever.search([_q("workshop capacity")], k=3)
    r2 = await retriever.search([_q("workshop capacity")], k=3)
    assert [e.chunk.chunk_id for e in r1[0].evidence] == [e.chunk.chunk_id for e in r2[0].evidence]


async def test_low_confidence_flag_on_unrelated_query():
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    results = await retriever.search([_q("quantum computing hardware roadmap")], k=3)
    assert results[0].low_confidence


async def test_get_chunk_by_citation_resolves_real_citation():
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    await retriever.setup()
    assert retriever.get_chunk_by_citation("Doc_01 §1") is not None
    assert retriever.get_chunk_by_citation("Doc_999 §1") is None


def test_ingest_parses_sections_and_frontmatter(tmp_path):
    md = """---
doc_id: Doc_42
title: Example
---

# Example

## §1 Overview

This is the overview section with enough content to form a chunk.

## §2 Details

Second section body text here.
"""
    path = tmp_path / "Doc_42_example.md"
    path.write_text(md, encoding="utf-8")
    chunks = parse_document(path)
    assert len(chunks) == 2
    assert chunks[0].doc_id == "Doc_42"
    assert chunks[0].section == "1"
    assert chunks[1].section == "2"
    assert chunks[0].citation == "Doc_42 §1"
