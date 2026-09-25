from __future__ import annotations

from streaming_rag.controller.decompose import decompose, is_self_contained, strip_discourse_markers
from streaming_rag.session.synthesis import GroundedSynthesizer

GUIDE_EXAMPLE = ("I need to plan a customer workshop in Pune for 30 people, and I need the "
                 "cancellation policy and the catering options.")


def test_discourse_markers_are_not_search_content():
    assert strip_discourse_markers("Wait, one more thing — What does stong force act upon?") == \
        "What does stong force act upon?"
    assert strip_discourse_markers("Actually, the trip was international") == "the trip was international"
    assert strip_discourse_markers("It turns out the booking was made after travel") == \
        "the booking was made after travel"


def test_self_contained_question_gets_no_foreign_context():
    u = "Who demonstrated the Egg of Columbus, and What disease did Tesla catch?"
    texts = [q.text for q in decompose(u, set(), "u1", "final", 0, context_carry="elliptical")]
    assert "What disease did Tesla catch?" in texts
    legacy = [q.text for q in decompose(u, set(), "u1", "final", 0, context_carry="short")]
    assert any("columbus" in t and "disease" in t.lower() for t in legacy)  # the old contamination


def test_elliptical_clauses_still_carry_the_topic():
    assert not is_self_contained("the catering options")
    texts = [q.text for q in decompose(GUIDE_EXAMPLE, set(), "u1", "final", 0, context_carry="elliptical")]
    assert texts == [q.text for q in decompose(GUIDE_EXAMPLE, set(), "u1", "final", 0, context_carry="short")]
    assert all("pune" in t.lower() for t in texts)


def test_relevance_gate_never_passes_a_query_with_no_content_words():
    assert GroundedSynthesizer._is_relevant("What is it?", "Anything at all can go here.") is False


async def _live_after(chunks):
    from streaming_rag.config import load_config
    from streaming_rag.contracts import TranscriptChunk
    from streaming_rag.controller import RetrievalController
    from streaming_rag.session.store import SessionStore
    controller, store = RetrievalController(None, None, load_config()), SessionStore()
    store.open("s1")
    live = {}
    for i, text in enumerate(chunks + [""]):
        d = await controller.on_chunk(TranscriptChunk("s1", "u1", i, text, i == len(chunks)), store.view("s1"))
        for qid in d.superseded_query_ids:
            live.pop(qid, None)
        for q in d.sub_queries:
            live.pop(q.parent_query_id, None)
            live[q.query_id] = q.text
    return list(live.values())


async def test_sibling_questions_are_never_cancelled_as_supersessions():
    final = await _live_after(["Who first showed that Newton's Theory of Gravity was not as correct",
                               " as another theory?, What are associated with normal forces?"])
    assert len(final) == 2
    assert any("normal forces" in t for t in final) and any("Newton" in t for t in final)


async def test_a_polluted_partial_clause_is_superseded_by_its_completion():
    final = await _live_after(["What came into force after the new constitution was herald?, What attracts",
                               " the tourists to Kenya?"])
    assert final == ["What came into force after the new constitution was herald?",
                     "What attracts the tourists to Kenya?"]


async def test_a_provisional_comma_split_orphan_is_also_superseded():
    # "Along with diesel engines, what engines..." has no "?" yet while
    # streaming, so the provisional splitter (correctly) treats the comma as
    # a list boundary and emits TWO live fragments; once the full question
    # arrives, both must be superseded, not just whichever one the 1:1
    # parent_query_id link happens to pick.
    final = await _live_after(["Along with diesel engines, what engines",
                               " have overtaken steam engines for marine",
                               " propulsion?, What is 565", " degrees C the creep limit of?"])
    assert final == ["Along with diesel engines, what engines have overtaken steam engines for marine propulsion?",
                     "565 degrees C the creep limit of?"]


def test_short_question_clause_is_kept_in_a_compound_request():
    texts = [q.text for q in decompose("business allowed for private companies to do what, and What is CSNET",
                                       set(), "u1", "final", 0, context_carry="elliptical")]
    assert any("CSNET" in t for t in texts)
