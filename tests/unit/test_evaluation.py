from __future__ import annotations

import json
import math

import pytest

from onebee.evaluation import (
    FakeJudge,
    FakeNLI,
    HarnessResult,
    JudgeVerdict,
    Probe,
    ProbeResult,
    SystemConfig,
    bootstrap_ci,
    cohens_h,
    contradiction_rate,
    detect_abstention,
    dual_order_score,
    entity_f1,
    exact_match,
    fmr,
    fuzzy_match,
    holm_bonferroni,
    minimum_detectable_effect,
    mrr,
    mur,
    normalize_text,
    paired_bootstrap_diff,
    pra_lenient,
    pra_strict,
    precision_at_k,
    recall_at_k,
    run_harness,
    save_harness_result,
    score_probe,
    uar,
)

# ---------------------------------------------------------------------------
# rule.py tests
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("Hello   World\n\tTest") == "hello world test"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_punctuation_only(self):
        assert normalize_text("!@# $%^") == ""


class TestExactMatch:
    def test_match_case_and_punctuation_diff(self):
        assert exact_match("Hello, World!", "hello world") is True

    def test_no_match(self):
        assert exact_match("Hello", "World") is False

    def test_empty_both(self):
        assert exact_match("", "") is True

    def test_empty_vs_nonempty(self):
        assert exact_match("", "hello") is False


class TestFuzzyMatch:
    def test_full_overlap(self):
        assert fuzzy_match("the cat sat", "cat sat the") is True

    def test_partial_overlap_above_threshold(self):
        assert fuzzy_match("the cat sat on the mat", "cat sat on mat", threshold=0.5) is True

    def test_partial_overlap_below_threshold(self):
        assert fuzzy_match("hello world", "goodbye universe", threshold=0.5) is False

    def test_empty_both(self):
        assert fuzzy_match("", "") is True

    def test_one_empty(self):
        assert fuzzy_match("hello", "") is False

    def test_default_threshold(self):
        assert fuzzy_match("hello world foo", "hello world bar", threshold=0.5) is True
        assert fuzzy_match("hello world foo", "hello world bar") is False


class TestEntityF1:
    def test_perfect_match(self):
        result = entity_f1(["Alice", "Bob", "Charlie"], ["alice", "bob", "charlie"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_partial_overlap(self):
        result = entity_f1(["Alice", "Bob"], ["Bob", "Charlie"])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert math.isclose(result["f1"], 0.5)

    def test_no_overlap(self):
        result = entity_f1(["Alice"], ["Bob"])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_empty_pred(self):
        result = entity_f1([], ["Alice", "Bob"])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_empty_gold(self):
        result = entity_f1(["Alice", "Bob"], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_both_empty(self):
        result = entity_f1([], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_case_insensitive(self):
        result = entity_f1(["aLiCe"], ["alice"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_whitespace_stripping(self):
        result = entity_f1([" Alice "], ["alice"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


class TestDetectAbstention:
    def test_i_dont_know(self):
        assert detect_abstention("I don't know the answer") is True

    def test_im_not_sure(self):
        assert detect_abstention("I'm not sure about that") is True

    def test_dont_have_information(self):
        assert detect_abstention("I don't have that information, sorry") is True

    def test_you_havent_told_me(self):
        assert detect_abstention("You haven't told me about that") is True

    def test_dont_recall(self):
        assert detect_abstention("I don't recall that happening") is True

    def test_sft_abstention_template(self):
        # Literal template from generate_sft_data.py's abstention examples -- a model that
        # learns to reproduce this exact intended abstention phrasing must be scored as
        # abstaining (see docs/model_quirks.md's dedup-collapse follow-up entry).
        assert (
            detect_abstention("I don't think you've told me that — I don't want to guess.") is True
        )

    def test_sft_irrelevant_retrieval_template(self):
        assert (
            detect_abstention(
                "I don't have anything about that in what I remember about you — "
                "could you tell me more?"
            )
            is True
        )

    def test_sft_diversified_abstain_paraphrases(self):
        assert (
            detect_abstention(
                "That's not something I have on record from our conversations — "
                "do you want to tell me?"
            )
            is True
        )
        assert (
            detect_abstention(
                "I don't recall you sharing that with me, so I don't want to make something up."
            )
            is True
        )

    def test_sft_diversified_irrelevant_paraphrases(self):
        assert (
            detect_abstention(
                "That doesn't ring a bell from what you've shared with me — mind filling me in?"
            )
            is True
        )
        assert (
            detect_abstention(
                "Hmm, I don't think you've mentioned that to me before — tell me more?"
            )
            is True
        )

    def test_normal_response(self):
        assert detect_abstention("The capital of France is Paris") is False

    def test_empty_string(self):
        assert detect_abstention("") is False


# ---------------------------------------------------------------------------
# judge.py tests
# ---------------------------------------------------------------------------


class TestFakeJudge:
    def test_determinism(self):
        judge = FakeJudge()
        v1 = judge.score_response("q", "hello world test", "rubric")
        v2 = judge.score_response("q", "hello world test", "rubric")
        assert v1.score == v2.score

    def test_score_derives_from_length(self):
        judge = FakeJudge()
        v = judge.score_response("q", "one two three four five six seven eight nine ten", "r")
        assert v.score == 1.0

    def test_score_capped_at_five(self):
        judge = FakeJudge()
        long_response = "word " * 100
        v = judge.score_response("q", long_response, "r")
        assert v.score == 5.0

    def test_empty_response_score_zero(self):
        judge = FakeJudge()
        v = judge.score_response("q", "", "r")
        assert v.score == 0.0

    def test_compare_pairwise_picks_longer(self):
        judge = FakeJudge()
        v = judge.compare_pairwise("q", "short", "long response here", "r", "AB")
        assert v.score == 0.0
        assert v.order == "AB"

    def test_compare_pairwise_tie(self):
        judge = FakeJudge()
        v = judge.compare_pairwise("q", "same len", "same val", "r", "AB")
        assert v.score == 0.5

    def test_compare_pairwise_a_longer(self):
        judge = FakeJudge()
        v = judge.compare_pairwise("q", "much longer response here", "short", "r", "BA")
        assert v.score == 1.0

    def test_judge_verdict_is_pydantic(self):
        v = JudgeVerdict(score=1.5, justification="test")
        d = v.model_dump()
        assert d["score"] == 1.5
        assert d["order"] is None


class TestDualOrderScore:
    def test_symmetry_equal_responses(self):
        judge = FakeJudge()
        s = dual_order_score(judge, "q", "short", "short", "r")
        assert s == pytest.approx(0.5)

    def test_preference_toward_longer(self):
        judge = FakeJudge()
        s = dual_order_score(judge, "q", "longer response text here", "short", "r")
        assert s > 0.5

    def test_preference_toward_longer_swapped(self):
        judge = FakeJudge()
        s = dual_order_score(judge, "q", "short", "longer response text here", "r")
        assert s < 0.5

    def test_symmetry_swapping_preserves_relative_preference(self):
        judge = FakeJudge()
        s1 = dual_order_score(judge, "q", "longer response text here", "short", "r")
        s2 = dual_order_score(judge, "q", "short", "longer response text here", "r")
        assert s1 == pytest.approx(1.0 - s2)


# ---------------------------------------------------------------------------
# nli.py tests
# ---------------------------------------------------------------------------


class TestFakeNLI:
    def test_entailment_high_overlap(self):
        nli = FakeNLI()
        label, conf = nli.check(
            "The cat sat on the mat and looked happy",
            "the cat sat on the mat",
        )
        assert label == "entailment"
        assert conf == 1.0

    def test_contradiction_negation_marker(self):
        nli = FakeNLI()
        label, conf = nli.check(
            "The cat sat on the mat and was happy",
            "the cat never sat on the mat",
        )
        assert label == "contradiction"
        assert conf == 1.0

    def test_contradiction_not_difference(self):
        nli = FakeNLI()
        label, conf = nli.check(
            "The dog is friendly and playful",
            "the dog is not friendly",
        )
        assert label == "contradiction"
        assert conf == 1.0

    def test_neutral_low_overlap(self):
        nli = FakeNLI()
        label, conf = nli.check(
            "The cat sat on the mat",
            "the dog ran in the park",
        )
        assert label == "neutral"
        assert conf == 1.0

    def test_neutral_exact_same_text(self):
        nli = FakeNLI()
        label, conf = nli.check("hello world", "hello world")
        assert label == "entailment"

    def test_empty_inputs(self):
        nli = FakeNLI()
        label, conf = nli.check("", "")
        assert label == "neutral"
        assert conf == 1.0


# ---------------------------------------------------------------------------
# personalized.py tests
# ---------------------------------------------------------------------------


def _make_probe(
    probe_id: str = "p1",
    category: str = "factual",
    answerable: bool = True,
    gold_answer: str = "Paris",
    acceptable: list[str] | None = None,
    gold_mem_ids: list[str] | None = None,
) -> Probe:
    return Probe(
        probe_id=probe_id,
        persona_id="persona_1",
        question="What is the capital of France?",
        gold_answer=gold_answer,
        gold_supporting_memory_ids=gold_mem_ids or [],
        category=category,
        answerable=answerable,
        acceptable_alternatives=acceptable or [],
    )


def _make_result(
    probe_id: str = "p1",
    category: str = "factual",
    answerable: bool = True,
    strict_correct: bool = True,
    lenient_correct: bool | None = None,
    abstained: bool = False,
    retrieved_ids: list[str] | None = None,
    gold_mem_ids: list[str] | None = None,
) -> ProbeResult:
    return ProbeResult(
        probe=_make_probe(
            probe_id=probe_id,
            category=category,
            answerable=answerable,
            gold_mem_ids=gold_mem_ids,
        ),
        response="Paris",
        retrieved_memory_ids=retrieved_ids or [],
        strict_correct=strict_correct,
        lenient_correct=lenient_correct,
        abstained=abstained,
    )


class TestPRAStrict:
    def test_all_correct(self):
        results = [
            _make_result("p1", strict_correct=True),
            _make_result("p2", strict_correct=True),
        ]
        assert pra_strict(results) == 1.0

    def test_half_correct(self):
        results = [
            _make_result("p1", strict_correct=True),
            _make_result("p2", strict_correct=False),
        ]
        assert pra_strict(results) == 0.5

    def test_excludes_unanswerable(self):
        results = [
            _make_result("p1", answerable=True, strict_correct=True),
            _make_result("p2", answerable=True, strict_correct=False),
            _make_result("p3", answerable=False, strict_correct=False),
        ]
        assert pra_strict(results) == 0.5

    def test_no_answerable(self):
        results = [_make_result("p3", answerable=False, strict_correct=False)]
        assert pra_strict(results) == 0.0


class TestPRALenient:
    def test_falls_back_to_strict_when_none(self):
        results = [
            _make_result("p1", strict_correct=True, lenient_correct=None),
            _make_result("p2", strict_correct=False, lenient_correct=None),
        ]
        assert pra_lenient(results) == 0.5

    def test_uses_lenient_when_set(self):
        results = [
            _make_result("p1", strict_correct=False, lenient_correct=True),
            _make_result("p2", strict_correct=True, lenient_correct=False),
        ]
        assert pra_lenient(results) == 0.5

    def test_excludes_unanswerable(self):
        results = [
            _make_result("p1", answerable=True, strict_correct=True, lenient_correct=None),
            _make_result("p2", answerable=False, strict_correct=False, lenient_correct=True),
        ]
        assert pra_lenient(results) == 1.0


class TestUAR:
    def test_all_abstained(self):
        results = [
            _make_result("p1", answerable=False, abstained=True),
            _make_result("p2", answerable=False, abstained=True),
        ]
        assert uar(results) == 1.0

    def test_none_abstained(self):
        results = [
            _make_result("p1", answerable=False, abstained=False),
            _make_result("p2", answerable=False, abstained=False),
        ]
        assert uar(results) == 0.0

    def test_only_answerable_gives_zero(self):
        results = [
            _make_result("p1", answerable=True, abstained=True),
            _make_result("p2", answerable=True, abstained=False),
        ]
        assert uar(results) == 0.0

    def test_no_unanswerable(self):
        results = [_make_result("p1", answerable=True)]
        assert uar(results) == 0.0


class TestScoreProbe:
    def test_strict_correct_exact_match(self):
        probe = _make_probe(gold_answer="Paris")
        result = score_probe(probe, "Paris", [])
        assert result.strict_correct is True
        assert result.abstained is False
        assert result.lenient_correct is None

    def test_strict_correct_acceptable_alternative(self):
        probe = _make_probe(gold_answer="Paris", acceptable=["City of Lights"])
        result = score_probe(probe, "city of lights", [])
        assert result.strict_correct is True

    def test_strict_incorrect(self):
        probe = _make_probe(gold_answer="Paris")
        result = score_probe(probe, "London", [])
        assert result.strict_correct is False

    def test_abstention_detected(self):
        probe = _make_probe(gold_answer="Paris")
        result = score_probe(probe, "I don't know", [])
        assert result.abstained is True

    def test_with_judge_sets_lenient(self):
        judge = FakeJudge()
        probe = _make_probe(gold_answer="Paris")
        result = score_probe(probe, "very long response " * 10, [], judge=judge)
        assert result.lenient_correct is True

    def test_judge_not_called_for_unanswerable(self):
        judge = FakeJudge()
        probe = _make_probe(gold_answer="Paris", category="unanswerable", answerable=False)
        result = score_probe(probe, "Paris", [], judge=judge)
        assert result.lenient_correct is None

    def test_rubric_always_contains_gold_answer(self):
        # Regression test: the rubric string must include the gold answer even when
        # acceptable_alternatives is empty (the common case) — a past bug put the
        # whole "Gold answer: ..." block inside a ternary keyed on
        # acceptable_alternatives, so probes with none got an EMPTY rubric and the
        # judge scored responses with no gold answer to check against at all.
        captured_rubrics: list[str] = []

        class RubricCapturingJudge:
            def score_response(self, question, response, rubric):
                captured_rubrics.append(rubric)
                return JudgeVerdict(score=5.0, justification="ok")

            def compare_pairwise(self, *args, **kwargs):
                raise NotImplementedError

        probe = _make_probe(gold_answer="Paris", acceptable=[])
        score_probe(probe, "some response", [], judge=RubricCapturingJudge())
        assert len(captured_rubrics) == 1
        assert "Paris" in captured_rubrics[0]
        assert captured_rubrics[0] != ""

        probe_with_alts = _make_probe(gold_answer="Paris", acceptable=["City of Lights"])
        score_probe(probe_with_alts, "some response", [], judge=RubricCapturingJudge())
        assert "Paris" in captured_rubrics[1]
        assert "City of Lights" in captured_rubrics[1]


# ---------------------------------------------------------------------------
# memory_quality.py tests
# ---------------------------------------------------------------------------


class TestMUR:
    def test_all_utilised(self):
        results = [
            _make_result(probe_id="p1", retrieved_ids=["m1", "m2"], gold_mem_ids=["m1"]),
            _make_result(probe_id="p2", retrieved_ids=["m3"], gold_mem_ids=["m3"]),
        ]
        assert mur(results) == 1.0

    def test_none_utilised(self):
        results = [
            _make_result(probe_id="p1", retrieved_ids=["m1"], gold_mem_ids=["m2"]),
        ]
        assert mur(results) == 0.0

    def test_ignores_empty_gold_ids(self):
        results = [
            _make_result(probe_id="p1", retrieved_ids=["m1"], gold_mem_ids=[]),
        ]
        assert mur(results) == 0.0

    def test_mixed(self):
        results = [
            _make_result(probe_id="p1", retrieved_ids=["m1"], gold_mem_ids=["m1"]),
            _make_result(probe_id="p2", retrieved_ids=["m2"], gold_mem_ids=["m3"]),
        ]
        assert mur(results) == 0.5


class TestFMR:
    def test_all_abstained_on_unanswerable_no_memory(self):
        results = [
            _make_result(
                probe_id="p1",
                answerable=False,
                abstained=True,
                retrieved_ids=["m99"],
                gold_mem_ids=["m1"],
            ),
        ]
        nli = FakeNLI()
        assert fmr(results, nli, "") == 0.0

    def test_none_abstained_on_unanswerable_no_memory(self):
        results = [
            _make_result(
                probe_id="p1",
                answerable=False,
                abstained=False,
                retrieved_ids=["m99"],
                gold_mem_ids=["m1"],
            ),
        ]
        nli = FakeNLI()
        assert fmr(results, nli, "") == 1.0

    def test_excludes_probes_with_overlapping_memory(self):
        results = [
            _make_result(
                probe_id="p1",
                answerable=False,
                abstained=False,
                retrieved_ids=["m1"],
                gold_mem_ids=["m1"],
            ),
            _make_result(
                probe_id="p2",
                answerable=False,
                abstained=True,
                retrieved_ids=["m99"],
                gold_mem_ids=["m2"],
            ),
        ]
        nli = FakeNLI()
        assert fmr(results, nli, "") == 0.0

    def test_empty_set(self):
        results = [_make_result(probe_id="p1", answerable=True, gold_mem_ids=["m1"])]
        nli = FakeNLI()
        assert fmr(results, nli, "") == 0.0


class TestContradictionRate:
    def test_all_contradictions(self):
        nli = FakeNLI()
        pairs = [
            ("The cat sat on the mat and was happy", "the cat never sat on the mat"),
            ("The dog is friendly and playful", "the dog is not friendly"),
        ]
        assert contradiction_rate(pairs, nli) == 1.0

    def test_no_contradictions(self):
        nli = FakeNLI()
        pairs = [("hello world", "goodbye universe")]
        assert contradiction_rate(pairs, nli) == 0.0

    def test_empty_list(self):
        nli = FakeNLI()
        assert contradiction_rate([], nli) == 0.0

    def test_respects_threshold(self):
        nli = FakeNLI()
        pairs = [
            ("The cat sat on the mat and was happy", "the cat never sat on the mat"),
        ]
        assert contradiction_rate(pairs, nli, threshold=2.0) == 0.0


class TestIRMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d"]
        gold = ["b", "d", "e"]
        assert precision_at_k(retrieved, gold, 2) == 0.5
        assert precision_at_k(retrieved, gold, 4) == 0.5

    def test_precision_at_k_empty_gold(self):
        assert precision_at_k(["a"], [], 1) == 0.0

    def test_precision_at_k_k_zero(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d"]
        gold = ["b", "d", "e"]
        assert recall_at_k(retrieved, gold, 2) == pytest.approx(1.0 / 3.0)
        assert recall_at_k(retrieved, gold, 4) == pytest.approx(2.0 / 3.0)

    def test_recall_at_k_empty_gold(self):
        assert recall_at_k(["a"], [], 1) == 0.0

    def test_mrr(self):
        retrieved = ["a", "b", "c"]
        gold = ["b", "d"]
        assert mrr(retrieved, gold) == 0.5

    def test_mrr_no_match(self):
        assert mrr(["a"], ["b"]) == 0.0

    def test_mrr_empty_gold(self):
        assert mrr(["a"], []) == 0.0

    def test_mrr_first_position(self):
        assert mrr(["a", "b"], ["a"]) == 1.0


# ---------------------------------------------------------------------------
# stats.py tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_reproducible_with_seed(self):
        values = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        r1 = bootstrap_ci(values, n_resamples=1000, seed=42)
        r2 = bootstrap_ci(values, n_resamples=1000, seed=42)
        assert r1 == r2

    def test_mean_within_ci(self):
        values = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        mean, lo, hi = bootstrap_ci(values, n_resamples=1000, seed=42)
        assert lo <= mean <= hi

    def test_constant_values_zero_width_ci(self):
        values = [0.5, 0.5, 0.5, 0.5]
        mean, lo, hi = bootstrap_ci(values, n_resamples=1000, seed=42)
        assert lo == hi
        assert lo == pytest.approx(0.5)

    def test_empty_list(self):
        mean, lo, hi = bootstrap_ci([], seed=42)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0


class TestPairedBootstrapDiff:
    def test_basic(self):
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 1.0]
        mean, lo, hi = paired_bootstrap_diff(a, b, n_resamples=1000, seed=42)
        assert lo <= mean <= hi

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            paired_bootstrap_diff([1.0, 2.0], [1.0])

    def test_reproducible(self):
        a = [0.8, 0.9, 0.7, 0.85]
        b = [0.6, 0.7, 0.5, 0.65]
        r1 = paired_bootstrap_diff(a, b, n_resamples=1000, seed=7)
        r2 = paired_bootstrap_diff(a, b, n_resamples=1000, seed=7)
        assert r1 == r2


class TestHolmBonferroni:
    def test_worked_example(self):
        p_vals = {"h1": 0.01, "h2": 0.04, "h3": 0.03}
        result = holm_bonferroni(p_vals, alpha=0.05)
        assert result["h1"] is True
        assert result["h2"] is False
        assert result["h3"] is False

    def test_some_rejected(self):
        p_vals = {"h1": 0.001, "h2": 0.04, "h3": 0.10}
        result = holm_bonferroni(p_vals, alpha=0.05)
        assert result["h1"] is True
        assert result["h2"] is False
        assert result["h3"] is False

    def test_none_rejected(self):
        p_vals = {"h1": 0.10, "h2": 0.20}
        result = holm_bonferroni(p_vals, alpha=0.05)
        assert result["h1"] is False
        assert result["h2"] is False

    def test_empty(self):
        assert holm_bonferroni({}) == {}


class TestCohensH:
    def test_same_proportions_zero(self):
        assert cohens_h(0.5, 0.5) == 0.0

    def test_sign_flips_when_args_swap(self):
        assert cohens_h(0.7, 0.3) == pytest.approx(-cohens_h(0.3, 0.7))

    def test_zero_proportions(self):
        assert cohens_h(0.0, 0.0) == 0.0


class TestMinimumDetectableEffect:
    def test_positive_result(self):
        mde = minimum_detectable_effect(n=100, baseline_p=0.5)
        assert mde > 0.0

    def test_larger_n_smaller_mde(self):
        mde1 = minimum_detectable_effect(n=100, baseline_p=0.5)
        mde2 = minimum_detectable_effect(n=1000, baseline_p=0.5)
        assert mde2 < mde1


# ---------------------------------------------------------------------------
# harness.py tests
# ---------------------------------------------------------------------------


def _fake_response_fn_fn(response_text: str, retrieved_ids: list[str]):
    def fn(probe: Probe) -> tuple[str, list[str]]:
        return response_text, retrieved_ids

    return fn


class TestRunHarness:
    def test_basic_run(self):
        probes = [
            _make_probe("p1", "factual", answerable=True, gold_answer="Paris"),
            _make_probe("p2", "unanswerable", answerable=False, gold_answer="N/A"),
        ]

        def response_fn(probe: Probe) -> tuple[str, list[str]]:
            if probe.probe_id == "p1":
                return "Paris", ["m1"]
            return "I don't know", []

        result = run_harness(probes, response_fn, system_name="test", seed=42)

        assert result.system == "test"
        assert result.n_probes == 2
        assert result.metrics["pra_strict"] == pytest.approx(1.0)
        assert result.metrics["uar"] == pytest.approx(1.0)
        assert "pra_strict" in result.metrics_ci
        assert "uar" in result.metrics_ci
        assert "factual" in result.per_category
        assert "unanswerable" in result.per_category

    def test_with_judge_sets_pra_lenient(self):
        probes = [
            _make_probe("p1", "factual", answerable=True, gold_answer="Paris"),
        ]
        judge = FakeJudge()

        def response_fn(probe: Probe) -> tuple[str, list[str]]:
            return "some valid response text", ["m1"]

        result = run_harness(probes, response_fn, judge=judge, system_name="test", seed=42)
        assert "pra_lenient" in result.metrics

    def test_per_category_keys_match_probes(self):
        probes = [
            _make_probe("p1", "factual", answerable=True),
            _make_probe("p2", "preference", answerable=True),
        ]

        def response_fn(probe: Probe) -> tuple[str, list[str]]:
            return "Paris", []

        result = run_harness(probes, response_fn, system_name="test", seed=42)
        assert set(result.per_category.keys()) == {"factual", "preference"}

    def test_no_answerable_probes_still_runs(self):
        probes = [
            _make_probe("p1", "unanswerable", answerable=False, gold_answer="N/A"),
        ]

        def response_fn(probe: Probe) -> tuple[str, list[str]]:
            return "hello", []

        result = run_harness(probes, response_fn, system_name="test", seed=42)
        assert result.metrics["pra_strict"] == 0.0
        assert "unanswerable" in result.per_category

    def test_mur_in_metrics(self):
        probes = [
            _make_probe("p1", "factual", answerable=True, gold_mem_ids=["m1"]),
        ]

        def response_fn(probe: Probe) -> tuple[str, list[str]]:
            return "Paris", ["m1"]

        result = run_harness(probes, response_fn, system_name="test", seed=42)
        assert result.metrics["mur"] == pytest.approx(1.0)


class TestSaveHarnessResult:
    def test_writes_valid_files(self, tmp_path):
        result = HarnessResult(
            system="test",
            n_probes=2,
            metrics={"pra_strict": 0.5, "uar": 1.0},
            metrics_ci={
                "pra_strict": (0.5, 0.2, 0.8),
                "uar": (1.0, 1.0, 1.0),
            },
            per_category={
                "factual": {"pra_strict": 0.5},
                "unanswerable": {"uar": 1.0},
            },
            raw_results=[
                {"probe_id": "p1", "strict_correct": True},
                {"probe_id": "p2", "abstained": True},
            ],
        )

        out_dir = tmp_path / "results" / "v1" / "test"
        save_harness_result(result, str(out_dir))

        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "raw.jsonl").exists()

        with open(out_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["system"] == "test"
        assert metrics["metrics"]["pra_strict"] == 0.5

        with open(out_dir / "raw.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["probe_id"] == "p1"


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------


class TestSystemConfig:
    def test_create_config(self):
        cfg = SystemConfig(name="my-system", description="A test system")
        assert cfg.name == "my-system"
        assert cfg.description == "A test system"
        d = cfg.model_dump()
        assert d["name"] == "my-system"
