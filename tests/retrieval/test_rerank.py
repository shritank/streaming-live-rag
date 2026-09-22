from __future__ import annotations

from streaming_rag.contracts import Chunk
from streaming_rag.retrieval.rerank import definitional_match, definitional_target, rerank


def test_definitional_target_extracts_acronym_from_stand_for_query():
    assert definitional_target("What does AC stand for?") == "AC"


def test_definitional_target_extracts_acronym_from_what_is_query():
    assert definitional_target("What is DNS?") == "DNS"


def test_definitional_target_extracts_acronym_from_define_query():
    assert definitional_target("Define RAM") == "RAM"


def test_definitional_target_none_for_non_definitional_query():
    assert definitional_target("When was the treaty signed?") is None


def test_definitional_match_finds_expansion_before_acronym():
    text = "The system uses modern alternating current (AC) electricity."
    assert definitional_match("AC", text)


def test_definitional_match_finds_acronym_before_expansion():
    text = "AC (alternating current) is used throughout the grid."
    assert definitional_match("AC", text)


def test_definitional_match_false_when_term_appears_without_parens():
    text = "The AC motor was patented in 1888 and sold widely."
    assert not definitional_match("AC", text)


def test_rerank_boosts_chunk_with_definitional_pattern_to_top():
    ubiquitous = Chunk(chunk_id="Doc_01 §27 #1", doc_id="Doc_01", section="27",
                        text="Westinghouse negotiated a deal over Tesla's AC patents for a lump sum.")
    definitional = Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1",
                          text="Tesla is best known for the modern alternating current (AC) system.")
    # Give the ubiquitous (wrong) chunk a HIGHER base fused score, as BM25
    # term-frequency naturally would for a term that recurs throughout a
    # document — the definitional boost must be able to overturn that.
    candidates = [
        (ubiquitous, 0.9, ubiquitous.text),
        (definitional, 0.3, definitional.text),
    ]
    ranked = rerank("What does AC stand for?", candidates)
    assert ranked[0][0].chunk_id == "Doc_01 §1 #1"


def test_rerank_unaffected_by_boost_on_non_definitional_query():
    a = Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1", text="Alternating current (AC) power.")
    b = Chunk(chunk_id="Doc_01 §2 #1", doc_id="Doc_01", section="2", text="Direct current power systems.")
    candidates = [(a, 0.5, a.text), (b, 0.9, b.text)]
    ranked = rerank("Tell me about direct current power systems", candidates)
    assert ranked[0][0].chunk_id == "Doc_01 §2 #1"
