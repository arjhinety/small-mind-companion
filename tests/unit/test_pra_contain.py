"""Pins the pre-registered ``pra_contain`` rule by behaviour.

``docs/preregistrations/study002-answer-containment.md`` fixes the rule before any Study 002 run.
These literals come from that document; a change here needs a dated amendment there.
"""

from __future__ import annotations

from onebee.evaluation.graders.rule import contains_answer, normalize_text
from onebee.evaluation.metrics.personalized import Probe, ProbeResult, pra_contain


def _result(gold: str, response: str, *, answerable: bool = True) -> ProbeResult:
    probe = Probe(
        probe_id="p",
        persona_id="x",
        question="q",
        gold_answer=gold,
        gold_supporting_memory_ids=[],
        category="factual" if answerable else "unanswerable",
        answerable=answerable,
    )
    return ProbeResult(probe=probe, response=response, strict_correct=False, abstained=False)


def test_normalize_text_literals() -> None:
    assert normalize_text("9:00 AM") == "900 am"
    assert normalize_text("Hip-Hop") == "hiphop"
    assert normalize_text("  New   York. ") == "new york"


def test_worked_example_is_three_fifths() -> None:
    results = [
        _result("Dublin", "You moved to Dublin."),
        _result("New York", "you live in new york."),
        _result("Harvard", "You graduated from either Berkeley or Harvard."),
        _result("April 30", "Your birthday is April 30th."),
        _result("mystery", "You prefer mysteries."),
        _result("anything", "I don't know.", answerable=False),
    ]
    assert pra_contain(results) == 0.6


def test_whole_word_only() -> None:
    assert contains_answer("I live in Dublin, Ireland", "Dublin")
    assert not contains_answer("Dubliners is a book", "Dublin")
    assert not contains_answer("You were 250 then", "25")


def test_empty_gold_never_contained() -> None:
    assert not contains_answer("anything at all", "")
    assert not contains_answer("anything at all", " ... ")


def test_nothing_answerable_is_none() -> None:
    assert pra_contain([_result("x", "x", answerable=False)]) is None
    assert pra_contain([]) is None
