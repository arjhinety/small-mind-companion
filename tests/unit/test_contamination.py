from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.check_contamination import (  # type: ignore[import-not-found]
    check_contamination,
    extract_text_fields,
    load_jsonl_texts,
    main,
    ngrams,
    tokenize,
)


class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("Hello, world! How are you?") == [
            "hello",
            "world",
            "how",
            "are",
            "you",
        ]

    def test_numbers_and_punctuation(self) -> None:
        assert tokenize("abc123 456-def") == ["abc123", "456", "def"]

    def test_empty(self) -> None:
        assert tokenize("") == []
        assert tokenize("!@#$%") == []


class TestNgrams:
    def test_basic(self) -> None:
        tokens = ["the", "quick", "brown", "fox"]
        result = ngrams(tokens, 2)
        assert result == {("the", "quick"), ("quick", "brown"), ("brown", "fox")}

    def test_n_larger_than_tokens(self) -> None:
        tokens = ["a", "b", "c"]
        assert ngrams(tokens, 5) == set()

    def test_n_equals_tokens(self) -> None:
        tokens = ["a", "b", "c"]
        assert ngrams(tokens, 3) == {("a", "b", "c")}

    def test_n_is_1(self) -> None:
        tokens = ["x", "y"]
        assert ngrams(tokens, 1) == {("x",), ("y",)}


class TestExtractTextFields:
    def test_probe_shape(self) -> None:
        obj = {
            "probe_id": "p000_p0001",
            "persona_id": "p000",
            "question": "What is Alice's company?",
            "gold_answer": "Stripe",
            "category": "factual",
        }
        fields = extract_text_fields(obj)
        assert "What is Alice's company?" in fields
        assert "Stripe" in fields

    def test_messages_shape(self) -> None:
        obj = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        fields = extract_text_fields(obj)
        # Every string value is read, including the "role" labels: the check now scans
        # labels as well as content. That is noise, not leakage, and it is deliberate.
        assert fields == ["user", "Hello!", "assistant", "Hi there!"]

    def test_reads_string_values_under_any_key(self) -> None:
        obj: dict = {"foo": "bar", "baz": 123}
        assert extract_text_fields(obj) == ["bar"]

    def test_reads_dpo_shape(self) -> None:
        obj = {"prompt": "Ask me.", "chosen": "Sure.", "rejected": "No."}
        assert extract_text_fields(obj) == ["Ask me.", "Sure.", "No."]

    def test_reads_nested_lists_and_dicts(self) -> None:
        obj = {"turns": [{"who": {"name": "Ana"}, "said": ["hi", ""]}]}
        assert extract_text_fields(obj) == ["Ana", "hi"]

    def test_text_key(self) -> None:
        obj = {"text": "some text content"}
        assert extract_text_fields(obj) == ["some text content"]

    def test_content_key(self) -> None:
        obj = {"content": "some content"}
        assert extract_text_fields(obj) == ["some content"]

    def test_skips_non_string_values(self) -> None:
        obj = {"question": None, "gold_answer": 42}
        assert extract_text_fields(obj) == []


EMOTIONAL_RANGE_PROBES = (
    Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "emotional_range" / "probes.jsonl"
)


class TestEmotionalRangeRealRow:
    """Regression: the emotional_range probes use "context" and "register_description",
    which the old fixed key list never read, so this benchmark contributed zero texts."""

    def test_real_row_text_is_read(self) -> None:
        with EMOTIONAL_RANGE_PROBES.open(encoding="utf-8") as fh:
            row = json.loads(fh.readline())
        fields = extract_text_fields(row)
        assert row["context"] in fields
        assert row["register_description"] in fields

    def test_real_file_yields_texts(self) -> None:
        texts = load_jsonl_texts(str(EMOTIONAL_RANGE_PROBES))
        assert len(texts) > 0


class TestLoadJsonlTexts:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"question": "Q1?", "gold_answer": "A1"}\n'
            '{"question": "Q2?", "gold_answer": "A2"}\n'
        )
        texts = load_jsonl_texts(str(f))
        assert texts == ["Q1?", "A1", "Q2?", "A2"]

    def test_malformed_line_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "test.jsonl"
        f.write_text('{"question": "Q1?"}\n' "not valid json\n" '{"question": "Q3?"}\n')
        texts = load_jsonl_texts(str(f))
        assert texts == ["Q1?", "Q3?"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_jsonl_texts(str(f)) == []


class TestCheckContamination:
    def test_overlap_detected(self, tmp_path: Path) -> None:
        long_text = (
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
        )
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        train_f.write_text(json.dumps({"text": long_text}) + "\n")
        eval_f.write_text(json.dumps({"text": long_text}) + "\n")

        findings = check_contamination(
            [str(train_f)],
            [str(eval_f)],
            n=13,
            min_overlap=1,
        )
        assert len(findings) == 1
        assert findings[0]["overlap_count"] >= 1
        assert findings[0]["eval_file"] == str(eval_f)

    def test_no_overlap(self, tmp_path: Path) -> None:
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        train_f.write_text(json.dumps({"text": "the quick brown fox"}) + "\n")
        eval_f.write_text(json.dumps({"text": "completely different content"}) + "\n")

        findings = check_contamination(
            [str(train_f)],
            [str(eval_f)],
            n=13,
            min_overlap=1,
        )
        assert findings == []

    def test_below_min_overlap(self, tmp_path: Path) -> None:
        long_text = (
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
        )
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        train_f.write_text(json.dumps({"text": long_text}) + "\n")
        eval_f.write_text(json.dumps({"text": long_text}) + "\n")

        findings = check_contamination(
            [str(train_f)],
            [str(eval_f)],
            n=13,
            min_overlap=999,
        )
        assert findings == []


class TestMain:
    def test_clean_case(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        train_f.write_text(json.dumps({"text": "the quick brown fox"}) + "\n")
        eval_f.write_text(json.dumps({"text": "completely different"}) + "\n")

        old_argv = sys.argv
        try:
            sys.argv = [
                "check_contamination.py",
                "--train-glob",
                str(train_f),
                "--eval-glob",
                str(eval_f),
            ]
            rc = main()
        finally:
            sys.argv = old_argv

        assert rc == 0
        captured = capsys.readouterr()
        assert "No contamination" in captured.out

    def test_contaminated_case(self, tmp_path: Path) -> None:
        long_text = (
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
            "the quick brown fox jumps over the lazy dog "
        )
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        train_f.write_text(json.dumps({"text": long_text}) + "\n")
        eval_f.write_text(json.dumps({"text": long_text}) + "\n")

        old_argv = sys.argv
        try:
            sys.argv = [
                "check_contamination.py",
                "--train-glob",
                str(train_f),
                "--eval-glob",
                str(eval_f),
                "--n",
                "13",
            ]
            rc = main()
        finally:
            sys.argv = old_argv

        assert rc == 1

    def test_zero_glob_match_warns_and_exits_0(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "check_contamination.py",
                "--train-glob",
                str(tmp_path / "does_not_exist*.jsonl"),
                "--eval-glob",
                str(tmp_path / "also_not_there*.jsonl"),
            ]
            rc = main()
        finally:
            sys.argv = old_argv

        assert rc == 0
        captured = capsys.readouterr()
        assert "zero" in captured.err or "nothing to check" in captured.err

    def test_real_pmb_fixture_data(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        probes_path = repo_root / "data" / "benchmarks" / "pmb_v0" / "probes.jsonl"

        if not probes_path.exists():
            pytest.skip("PMB fixture data not found")

        probe_texts = load_jsonl_texts(str(probes_path))
        assert len(probe_texts) > 0

        train_f = tmp_path / "train.jsonl"
        train_f.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "this is unrelated synthetic text"},
                    ]
                }
            )
            + "\n"
        )

        findings = check_contamination(
            [str(train_f)],
            [str(probes_path)],
            n=13,
            min_overlap=1,
        )
        assert findings == []
