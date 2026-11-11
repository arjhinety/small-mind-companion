from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from onebee.evaluation.graders.judge import Judge
from onebee.evaluation.graders.rule import contains_answer, detect_abstention, exact_match


class Probe(BaseModel):
    probe_id: str
    persona_id: str
    question: str
    gold_answer: str
    gold_supporting_memory_ids: list[str]
    category: Literal[
        "factual",
        "preference",
        "episodic",
        "temporal",
        "unanswerable",
        "outdated_fact",
        "distractor",
        "continuity",
    ]
    answerable: bool
    acceptable_alternatives: list[str] = []


class ProbeResult(BaseModel):
    probe: Probe
    response: str
    retrieved_memory_ids: list[str] = []
    strict_correct: bool
    lenient_correct: bool | None = None
    abstained: bool


def pra_strict(results: list[ProbeResult]) -> float:
    answerable = [r for r in results if r.probe.answerable]
    if not answerable:
        return 0.0
    return sum(1 for r in answerable if r.strict_correct) / len(answerable)


def pra_lenient(results: list[ProbeResult]) -> float:
    answerable = [r for r in results if r.probe.answerable]
    if not answerable:
        return 0.0
    n_correct = 0
    for r in answerable:
        if r.lenient_correct is True:
            n_correct += 1
        elif r.lenient_correct is None and r.strict_correct:
            n_correct += 1
    return n_correct / len(answerable)


def pra_contain(results: list[ProbeResult]) -> float | None:
    """Share of answerable probes whose response contains the gold answer as a whole word or
    phrase (``contains_answer``). Secondary cross-check on ``pra_lenient``, pre-registered in
    ``docs/preregistrations/study002-answer-containment.md``. ``None`` when nothing is
    answerable."""
    answerable = [r for r in results if r.probe.answerable]
    if not answerable:
        return None
    hits = sum(contains_answer(r.response, r.probe.gold_answer) for r in answerable)
    return hits / len(answerable)


def uar(results: list[ProbeResult]) -> float:
    unanswerable = [r for r in results if not r.probe.answerable]
    if not unanswerable:
        return 0.0
    return sum(1 for r in unanswerable if r.abstained) / len(unanswerable)


def score_probe(
    probe: Probe,
    response: str,
    retrieved_memory_ids: list[str],
    judge: Judge | None = None,
) -> ProbeResult:
    strict_correct = exact_match(response, probe.gold_answer) or any(
        exact_match(response, alt) for alt in probe.acceptable_alternatives
    )
    abstained = detect_abstention(response)

    lenient_correct: bool | None = None
    if judge is not None and probe.answerable:
        # NOTE: the gold answer must ALWAYS be in the rubric — a past version of this
        # line put the whole "Gold answer: ..." block inside a ternary keyed on
        # `probe.acceptable_alternatives`, so any probe with no alternatives (the
        # common case) got an EMPTY rubric, meaning the judge scored responses with
        # no gold answer to check against at all. Caught via a real harness run
        # where a raw model with zero injected context scored ~94% "lenient correct"
        # on personalized-recall probes it had no way to actually answer.
        rubric = f"Question: {probe.question}\nGold answer: {probe.gold_answer}"
        if probe.acceptable_alternatives:
            rubric += f"\nAcceptable alternatives: {', '.join(probe.acceptable_alternatives)}"
        verdict = judge.score_response(probe.question, response, rubric)
        lenient_correct = verdict.score >= 3.0

    return ProbeResult(
        probe=probe,
        response=response,
        retrieved_memory_ids=retrieved_memory_ids,
        strict_correct=strict_correct,
        lenient_correct=lenient_correct,
        abstained=abstained,
    )
