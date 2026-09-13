from __future__ import annotations

import json
from unittest import mock

import pytest

from onebee.training.distill import (
    DistillationTrainingConfig,
    build_lora_config,
    load_distill_config,
    run_distillation,
)


def _make_fake_peft_module():
    fake_peft = mock.MagicMock()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_peft.LoraConfig = FakeLoraConfig
    return fake_peft


class TestDistillationTrainingConfig:
    def test_required_fields(self):
        with pytest.raises(Exception):
            DistillationTrainingConfig(output_dir="out")

    def test_defaults(self):
        cfg = DistillationTrainingConfig(
            base_model="student-model",
            teacher_model="teacher-model",
            train_file="train.jsonl",
            val_file="val.jsonl",
            output_dir="out",
        )
        assert cfg.learning_rate == 1e-6
        assert cfg.beta == 1.0
        assert cfg.temperature == 1.0
        assert cfg.lora_r == 16


class TestLoadDistillConfig:
    def test_round_trip(self, tmp_path):
        cfg_path = tmp_path / "distill.yaml"
        cfg_path.write_text(
            "base_model: student\n"
            "teacher_model: teacher\n"
            "train_file: train.jsonl\n"
            "val_file: val.jsonl\n"
            "output_dir: out\n"
        )
        cfg = load_distill_config(str(cfg_path))
        assert cfg.base_model == "student"
        assert cfg.teacher_model == "teacher"


class TestBuildLoraConfig:
    def test_all_linear_passes_string(self):
        cfg = DistillationTrainingConfig(
            base_model="m", teacher_model="t", train_file="tr", val_file="v", output_dir="o"
        )
        with mock.patch.dict("sys.modules", {"peft": _make_fake_peft_module()}):
            lora = build_lora_config(cfg)
        assert lora.kwargs["target_modules"] == "all-linear"


class TestRunDistillation:
    def test_dry_run_with_fakes_completes(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"prompt": [{"role": "user", "content": "hi"}]}) + "\n")
        val_path.write_text(json.dumps({"prompt": [{"role": "user", "content": "hello"}]}) + "\n")

        cfg = DistillationTrainingConfig(
            base_model="student-model",
            teacher_model="teacher-model",
            train_file=str(train_path),
            val_file=str(val_path),
            output_dir=str(tmp_path / "output"),
        )

        tokenizer_calls = []
        trainer_calls = []

        def fake_tokenizer_loader(model_name, revision):
            tokenizer_calls.append((model_name, revision))
            tok = mock.MagicMock()
            tok.pad_token_id = None
            tok.eos_token_id = 0
            return tok

        def fake_trainer_factory(**kwargs):
            trainer_calls.append(kwargs)
            return mock.MagicMock()

        with mock.patch.dict("sys.modules", {"peft": _make_fake_peft_module()}):
            run_distillation(
                cfg,
                dry_run=True,
                tokenizer_loader=fake_tokenizer_loader,
                trainer_factory=fake_trainer_factory,
            )

        assert len(tokenizer_calls) == 1
        assert tokenizer_calls[0] == ("student-model", "main")

        assert len(trainer_calls) == 1
        tc = trainer_calls[0]
        assert tc["model"] == "student-model"
        assert len(tc["train_dataset"]) == 1
        assert tc["train_dataset"][0]["prompt"][0]["content"] == "hi"
        assert tc["_config_kwargs"]["teacher_model_name_or_path"] == "teacher-model"
        assert tc["_config_kwargs"]["model_init_kwargs"] == {"dtype": "bfloat16"}
        assert tc["_config_kwargs"]["teacher_model_init_kwargs"] == {"dtype": "bfloat16"}
        # Regression test: teacher_model must be passed explicitly to the trainer
        # constructor, not just as config metadata (real bug found via a live training run
        # -- trainer.teacher_model stayed None and crashed on the first training step).
        assert tc["_teacher_model"] == "teacher-model"
