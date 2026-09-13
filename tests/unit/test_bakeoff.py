from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from onebee.evaluation.graders.judge import FakeJudge
from onebee.evaluation.graders.openai_judge import OpenAIJudge
from scripts.model_bakeoff import (  # type: ignore[import-not-found]
    CANDIDATE_MODELS,
    CATEGORIES,
    REPO_ROOT,
    build_smoke_prompts,
    main,
    score_with_judge,
    write_adr,
)


class TestBuildSmokePrompts:
    def test_returns_exactly_40_prompts_8_per_category(self):
        prompts = build_smoke_prompts()
        assert len(prompts) == 40
        counts: dict[str, int] = {}
        for p in prompts:
            counts[p["category"]] = counts.get(p["category"], 0) + 1
        assert counts == {category: 8 for category in CATEGORIES}

    def test_all_prompts_have_nonempty_text(self):
        prompts = build_smoke_prompts()
        assert all(p["prompt"].strip() for p in prompts)

    def test_structured_context_entries_all_have_context(self):
        prompts = build_smoke_prompts()
        structured = [p for p in prompts if p["category"] == "structured_context"]
        assert len(structured) == 8
        assert all(p["context"] is not None for p in structured)

    def test_vision_prompts_point_at_existing_images(self):
        prompts = build_smoke_prompts()
        for p in prompts:
            if p["category"] == "vision":
                assert p["image_path"] is not None
                assert Path(p["image_path"]).is_file()
            else:
                assert p["image_path"] is None

    def test_ids_are_unique(self):
        prompts = build_smoke_prompts()
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))


class TestOpenAIJudgeConstruction:
    def test_construction_needs_no_openai_package_or_env(self):
        assert "openai" not in sys.modules
        judge = OpenAIJudge(model="gpt-4o")
        assert judge.model == "gpt-4o"
        assert judge.api_key is None
        assert judge.temperature == 0.0
        assert "openai" not in sys.modules

    def test_api_key_and_base_url_are_stored(self):
        judge = OpenAIJudge(model="gpt-4o", api_key="sk-test", base_url="http://x", temperature=0.3)
        assert judge.api_key == "sk-test"
        assert judge.base_url == "http://x"
        assert judge.temperature == 0.3

    def test_retries_without_temperature_when_unsupported(self, monkeypatch):
        # Some models (reasoning-style, e.g. gpt-5.6-luna) reject any non-default
        # temperature. Simulate that: first call with temperature=... raises
        # BadRequestError mentioning "temperature"; the retry without it succeeds.
        import types

        calls: list[dict] = []

        class FakeBadRequestError(Exception):
            pass

        class FakeResponse:
            def __init__(self, content):
                self.choices = [
                    types.SimpleNamespace(message=types.SimpleNamespace(content=content))
                ]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if "temperature" in kwargs:
                    raise FakeBadRequestError(
                        "Unsupported value: 'temperature' does not support 0 with this model."
                    )
                return FakeResponse('{"score": 4.0, "justification": "ok"}')

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAIClient:
            def __init__(self, **kwargs):
                self.chat = FakeChat()

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAIClient
        fake_openai.BadRequestError = FakeBadRequestError
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        judge = OpenAIJudge(model="gpt-5.6-luna", api_key="sk-test")
        verdict = judge.score_response("q", "r", "rubric")

        assert verdict.score == 4.0
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]
        assert judge._temperature_unsupported is True

        # Second call for the same instance should skip the failing attempt entirely.
        calls.clear()
        judge.score_response("q2", "r2", "rubric")
        assert len(calls) == 1
        assert "temperature" not in calls[0]


class TestScoreWithJudge:
    def test_aggregation_math(self):
        prompts = [
            {"id": "instruction-01", "category": "instruction", "prompt": "q1", "context": None},
            {"id": "instruction-02", "category": "instruction", "prompt": "q2", "context": None},
            {
                "id": "structured_context-01",
                "category": "structured_context",
                "prompt": "q3",
                "context": "ctx",
            },
        ]
        responses_by_model = {
            "model-a": [
                {"prompt_id": "instruction-01", "model": "model-a", "response": "word " * 20},
                {"prompt_id": "instruction-02", "model": "model-a", "response": "word " * 40},
                {
                    "prompt_id": "structured_context-01",
                    "model": "model-a",
                    "response": "word " * 30,
                },
                {"prompt_id": "structured_context-01", "model": "model-a", "error": "boom"},
            ],
            "model-b": [
                {"prompt_id": "instruction-01", "model": "model-b", "response": "word " * 10},
                {"prompt_id": "instruction-02", "model": "model-b", "response": "word " * 20},
                {
                    "prompt_id": "structured_context-01",
                    "model": "model-b",
                    "response": "word " * 50,
                },
            ],
        }

        results = score_with_judge(FakeJudge(), prompts, responses_by_model)

        assert results["model-a"]["instruction"] == pytest.approx(3.0)
        assert results["model-a"]["structured_context"] == pytest.approx(3.0)
        assert results["model-b"]["instruction"] == pytest.approx(1.5)
        assert results["model-b"]["structured_context"] == pytest.approx(5.0)


class TestWriteAdr:
    def test_replaces_decision_only_and_keeps_prose(self, tmp_path):
        real_adr = REPO_ROOT / "docs/adr/0001-model-selection.md"
        original = real_adr.read_text(encoding="utf-8")
        copy = tmp_path / "0001-model-selection.md"
        copy.write_text(original, encoding="utf-8")

        results = {
            "model-a": {"instruction": 3.0, "structured_context": 3.0},
            "model-b": {"instruction": 1.5, "structured_context": 5.0},
        }
        pinned_shas = {"model-a": "abc123", "model-b": "def456"}
        write_adr(results, str(copy), pinned_shas)

        new_text = copy.read_text(encoding="utf-8")

        orig_prefix = original[: original.index("## Decision\n")]
        new_prefix = new_text[: new_text.index("## Decision\n")]
        assert new_prefix == orig_prefix

        orig_suffix = original[original.index("## Consequences") :]
        new_suffix = new_text[new_text.index("## Consequences") :]
        assert new_suffix == orig_suffix

        assert "TBD" not in new_text
        assert (
            "| Model | instruction | en_dialogue | ja_dialogue | structured_context "
            "| vision | Overall |" in new_text
        )
        assert "| model-a | 3.00 | — | — | 3.00 | — | 3.00 |" in new_text
        assert "| model-b | 1.50 | — | — | 5.00 | — | 3.25 |" in new_text
        assert "**Recommendation:** pin **model-b**" in new_text
        assert "- model-a: `abc123`" in new_text
        assert "- model-b: `def456`" in new_text


class TestMainSkipDownload:
    def test_runs_end_to_end_without_heavy_deps(self, tmp_path):
        out_dir = tmp_path / "out"
        adr_path = tmp_path / "0001-model-selection.md"
        real_adr = REPO_ROOT / "docs/adr/0001-model-selection.md"
        adr_path.write_text(real_adr.read_text(encoding="utf-8"), encoding="utf-8")

        rc = main(["--skip-download", "--out-dir", str(out_dir), "--adr-path", str(adr_path)])
        assert rc == 0

        raw = json.loads((out_dir / "bakeoff_raw.json").read_text(encoding="utf-8"))
        scores = json.loads((out_dir / "bakeoff_scores.json").read_text(encoding="utf-8"))

        assert set(raw.keys()) == set(CANDIDATE_MODELS.keys())
        assert all(len(records) == 40 for records in raw.values())

        assert set(scores.keys()) == set(CANDIDATE_MODELS.keys())
        for by_category in scores.values():
            assert set(by_category.keys()) == set(CATEGORIES)

        for mod in ("openai", "torch", "transformers", "huggingface_hub"):
            assert mod not in sys.modules


class TestCliHelp:
    def test_help_works_without_heavy_imports(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/model_bakeoff.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--skip-download" in result.stdout
        assert "--judge-model" in result.stdout
