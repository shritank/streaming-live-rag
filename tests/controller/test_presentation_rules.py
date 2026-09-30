"""Presentation-only / chit-chat classification (pitfall 4: never query the
corpus for a reformat request) - and, as important, never mistake a real
question for one."""
from __future__ import annotations

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Decision, TranscriptChunk, TurnKind
from streaming_rag.controller import RetrievalController
from streaming_rag.controller.controller import is_chit_chat, is_presentation_request
from streaming_rag.session.store import SessionStore

REFORMAT = [
    "Please repeat your last answer in two bullets.", "Now give that in one short sentence.",
    "Can you say that again?", "Make it shorter", "Say it in simpler words", "Translate that into Hindi",
    "Shorten", "Summarize that in one sentence", "Put that in a table", "In two bullets please",
    "Give me that as bullet points", "tl;dr", "Could you rephrase your previous answer more briefly?",
]
REAL_QUESTIONS = [
    "Summarize the travel reimbursement rule for an employee trip.", "What is a list?",
    "Give me a summary of the cancellation policy", "What does 'brief' mean here?",
    "Repeat the cancellation policy", "What is the difference between a sentence and a paragraph?",
    "Explain how the tourism industry expanded in Kenya", "Translate the term aliyah",
    "What attracts the tourists to Kenya?", "Actually the trip was international",
]


@pytest.mark.parametrize("text", REFORMAT)
def test_reformat_requests_are_presentation_only(text):
    assert is_presentation_request(text)


@pytest.mark.parametrize("text", REAL_QUESTIONS)
def test_real_questions_are_never_presentation_only(text):
    assert not is_presentation_request(text)


@pytest.mark.parametrize("text", ["Thanks, that's all for today.", "Thank you so much!", "hello",
                                  "Okay great, bye", "ok", "perfect"])
def test_chit_chat(text):
    assert is_chit_chat(text)


@pytest.mark.parametrize("text", ["That's all for the tourism policy?", "Thanks for nothing, what is the cancellation policy",
                                  "Good morning, what attracts tourists to Kenya?", "all", "Great Britain history"])
def test_questions_containing_greeting_words_are_not_chit_chat(text):
    assert not is_chit_chat(text)


async def _decide(text, with_answer):
    c = RetrievalController(None, None, load_config())
    store = SessionStore()
    store.open("s1")
    session = store.view("s1")
    if with_answer:
        class _Answered:                       # a session that already holds an answer
            session_id = "s1"
            has_answer = staticmethod(lambda: True)
            latest_answer = staticmethod(lambda: None)
            topic_summary = staticmethod(lambda: "tourism kenya")
            prior_sub_queries = staticmethod(lambda: [])
        session = _Answered()
    return await c.on_chunk(TranscriptChunk("s1", "u1", 0, text, True), session)


async def test_reformat_with_an_answer_is_suppressed_with_the_documented_reason():
    d = await _decide("Now give that in one short sentence.", with_answer=True)
    assert (d.decision, d.turn_kind, d.reason) == (Decision.SUPPRESS, TurnKind.PRESENTATION_ONLY,
                                                    "presentation_restructure")


async def test_reformat_without_an_answer_does_not_search_the_corpus():
    d = await _decide("Please repeat your last answer in two bullets.", with_answer=False)
    assert d.decision == Decision.SUPPRESS and d.turn_kind == TurnKind.PRESENTATION_ONLY
    assert d.reason == "presentation_without_answer" and not d.sub_queries


async def test_a_new_request_that_starts_with_summarize_still_retrieves_after_an_answer():
    d = await _decide("Summarize the travel reimbursement rule for an employee trip.", with_answer=True)
    assert d.decision == Decision.RETRIEVE


# every phrase the previous (regex) chit-chat rule accepted must still be chit-chat
_OLD_RULE_PHRASES = ["hi", "hello", "hey", "thanks", "thank you", "thankyou", "cheers", "ok", "okay", "got it",
                     "great", "perfect", "bye", "goodbye", "good morning", "good afternoon"]


@pytest.mark.parametrize("phrase", _OLD_RULE_PHRASES)
@pytest.mark.parametrize("suffix", ["", ".", "!", " ", ", ", "!!"])
def test_new_chit_chat_rule_accepts_everything_the_old_rule_did(phrase, suffix):
    assert is_chit_chat(phrase + suffix)
    assert is_chit_chat((phrase + suffix).title())
