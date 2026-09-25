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


def test_provenance_files_are_not_indexed_as_documents(tmp_path):
    from streaming_rag.retrieval.ingest import load_corpus
    (tmp_path / "Doc_01_a.md").write_text("---\ndoc_id: Doc_01\ntitle: A\n---\n## §1 One\n\nReal content.\n",
                                          encoding="utf-8")
    (tmp_path / "SOURCE.md").write_text("Source: licence note listing every article title.\n", encoding="utf-8")
    (tmp_path / "README.txt").write_text("How this corpus was built.\n", encoding="utf-8")
    assert {c.doc_id for c in load_corpus(tmp_path)} == {"Doc_01"}


async def test_citation_resolves_to_whole_section_so_later_chunk_claims_verify(tmp_path):
    import asyncio
    from streaming_rag.config import load_config
    from streaming_rag.contracts import Claim
    from streaming_rag.retrieval import HybridRetriever
    from streaming_rag.session.grounding import verify
    sentences = " ".join(f"Sentence number {i} talks about topic {i} in some detail here." for i in range(30))
    (tmp_path / "Doc_01_a.md").write_text(f"---\ndoc_id: Doc_01\ntitle: A\n---\n## §1 One\n\n{sentences}\n",
                                          encoding="utf-8")
    config = load_config()
    config.corpus_dir = str(tmp_path)
    retriever = HybridRetriever(config)
    await retriever.setup()
    section_chunks = [c for c in retriever.chunks if c.citation == "Doc_01 §1"]
    assert len(section_chunks) > 1, "fixture must span several chunks"
    last_sentence = "Sentence number 29 talks about topic 29 in some detail here."
    assert last_sentence not in section_chunks[0].text
    report = verify([Claim(text=last_sentence, citations=["Doc_01 §1"])], retriever.get_chunk_by_citation)
    assert report.n_supported == 1


def test_pinned_model_files_match_their_checksums():
    import pytest
    from streaming_rag.retrieval import neural
    problems = neural.verify()
    if problems and all("not in the local Hugging Face cache" in p for p in problems):
        pytest.skip("neural models not fetched on this machine")
    assert problems == []
