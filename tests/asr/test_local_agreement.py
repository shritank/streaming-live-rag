from __future__ import annotations

from streaming_rag.asr import LocalAgreement


def test_words_commit_only_once_two_consecutive_decodes_agree():
    la = LocalAgreement()
    assert la.update("what is the".split()) == []            # nothing to agree with yet
    assert la.update("what is the universal".split()) == ["what", "is", "the"]
    assert la.update("what is the universal band".split()) == ["universal"]
    assert la.committed == ["what", "is", "the", "universal"]


def test_a_revised_word_is_held_back_until_the_decodes_agree_again():
    la = LocalAgreement()
    la.update("who wrote the savory".split())
    assert la.update("who wrote of the savery".split()) == ["who", "wrote"]   # disagreement at word 3
    assert la.update("who wrote of the savery water".split()) == ["of", "the", "savery"]


def test_committed_words_are_never_retracted():
    la = LocalAgreement()
    la.update("a b c".split())
    la.update("a b c d".split())
    before = list(la.committed)
    la.update("x y z".split())                                 # a wildly different hypothesis
    assert la.committed[:len(before)] == before


def test_flush_appends_the_rest_of_the_final_decode():
    la = LocalAgreement()
    la.update("what is the".split())
    la.update("what is the universal".split())        # commits "what is the"; "universal" seen once
    assert la.flush("what is the universal band on?".split()) == ["universal", "band", "on?"]
    assert " ".join(la.committed) == "what is the universal band on?"


def test_flush_after_a_final_revision_appends_the_revised_tail():
    la = LocalAgreement()
    la.update("a b c".split())
    la.update("a b c".split())
    assert la.committed == ["a", "b", "c"]
    # final decode revises "b" -> "x": the corrected tail is appended, nothing retracted
    assert la.flush("a x c d".split()) == ["x", "c", "d"]
    assert la.committed == ["a", "b", "c", "x", "c", "d"]
