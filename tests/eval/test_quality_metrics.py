"""The gold-referenced quality metrics must score what they claim to: a right
answer as right, a confidently wrong (but self-supported) claim as wrong, and
the statistics must behave at the edges."""
from __future__ import annotations

from streaming_rag.contracts import AnswerVersion, Claim
from eval.quality import contains_answer, normalize, score_quality
from eval.stats import paired_bootstrap, wilson


def _answer(version, claims, parent=None, change_kind="initial", uncertainty=None):
    return AnswerVersion(version=version, parent_version=parent, change_kind=change_kind,
                         text=" ".join(c.text for c in claims), claims=claims,
                         citations=sorted({x for c in claims for x in c.citations}),
                         evidence_ids=[], sub_queries=[], uncertainty=uncertainty, created_ms=0)


QRELS = {
    "What disease did Tesla catch?": {"relevant": ["Doc_09 §8"], "answers": ["cholera"]},
    "Who demonstrated the Egg of Columbus?": {"relevant": ["Doc_09 §34"], "answers": ["Tesla"]},
}


TITLES = {"Doc_09": "Nikola Tesla"}


def _result(answer: AnswerVersion, uid="u1"):
    return {"kind": "turn_result", "session_id": "s1", "utterance_id": uid, "answer": answer.text,
            "citations": answer.citations, "uncertainty": answer.uncertainty,
            "answer_version": answer.version, "parent_version": answer.parent_version}


def test_normalisation_follows_squad_conventions():
    assert normalize("The  Cholera!") == "cholera"
    assert contains_answer("In 1873 he contracted cholera.", ["Cholera"])
    assert not contains_answer("He studied choleric temperaments.", ["cholera"])  # whole words only


def test_right_answer_counts_and_confident_wrong_claim_is_caught():
    gt = {"turns": {"u1": {"kind": "new_request", "gold_sub_intents": list(QRELS)}}}
    right = Claim(text="Tesla contracted cholera and was bedridden.", citations=["Doc_09 §8"])
    # Cited, verbatim, self-supported — and answers nothing that was asked.
    wrong = Claim(text="Tesla demonstrated a bladeless turbine.", citations=["Doc_09 §49"])
    ans = _answer(1, [right, wrong])
    q = score_quality("sc", gt, [_result(ans)], {("s1", 1): ans}, {}, QRELS, TITLES)
    assert q.answer[("sc", "u1:0")] == 1          # cholera found
    # "Tesla" is the article's own subject, so finding it proves nothing: the
    # question is judged by its gold citation (Doc_09 §34), which is absent.
    assert q.answer[("sc", "u1:1")] == 0
    assert q.citation[("sc", "u1:0")] == 1 and q.citation[("sc", "u1:1")] == 0
    assert q.claim_correct[("sc", "u1:c0")] == 1
    assert q.claim_correct[("sc", "u1:c1")] == 0
    assert len(q.wrong_claims) == 1


def test_carried_claims_are_not_rescored_on_refinement():
    gt = {"turns": {"u1": {"kind": "new_request", "gold_sub_intents": ["Who demonstrated the Egg of Columbus?"]},
                    "u2": {"kind": "refinement", "gold_sub_intents": ["What disease did Tesla catch?"]}}}
    c1 = Claim(text="Tesla demonstrated the Egg of Columbus.", citations=["Doc_09 §34"])
    c2 = Claim(text="Tesla contracted cholera.", citations=["Doc_09 §8"])
    v1 = _answer(1, [c1])
    v2 = _answer(2, [c1, c2], parent=1, change_kind="refinement")
    q = score_quality("sc", gt, [_result(v1), _result(v2, "u2")], {("s1", 1): v1, ("s1", 2): v2}, {}, QRELS, TITLES)
    assert set(k for k in q.claim_correct if k[1].startswith("u2")) == {("sc", "u2:c0")}
    assert q.claim_correct[("sc", "u2:c0")] == 1


def test_unanswerable_turn_rewards_abstention_only():
    gt = {"turns": {"u1": {"kind": "new_request", "gold_sub_intents": ["What is x?"], "expect_uncertainty": True}}}
    abstain = _answer(1, [], uncertainty="Could not be verified")
    q = score_quality("sc", gt, [_result(abstain)], {("s1", 1): abstain}, {}, QRELS, TITLES)
    assert q.abstention[("sc", "u1")] == 1
    asserted = _answer(1, [Claim(text="Something.", citations=["Doc_01 §1"])])
    q = score_quality("sc", gt, [_result(asserted)], {("s1", 1): asserted}, {}, QRELS, TITLES)
    assert q.abstention[("sc", "u1")] == 0
    assert q.claim_correct[("sc", "u1:c0")] == 0


def test_wilson_interval_edges():
    lo, hi = wilson(37, 37)
    assert hi == 1.0 and 0.88 < lo < 0.92       # 37/37 is NOT evidence of >95%
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(36, 37)
    assert lo < 0.95 < hi


def test_paired_bootstrap_detects_consistent_gain_and_not_noise():
    a = {(i, 0): 0.0 for i in range(40)}
    b = {(i, 0): 1.0 if i < 20 else 0.0 for i in range(40)}
    diff, lo, hi = paired_bootstrap(b, a, n_boot=1000)
    assert diff == 0.5 and lo > 0
    diff, lo, hi = paired_bootstrap(a, dict(a), n_boot=200)
    assert diff == 0 and lo == 0 and hi == 0
