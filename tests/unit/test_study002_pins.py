"""Every Study 002 config's revision pins reach every model, tokenizer and teacher load.

STUDIES item 13 / ERRATA E4: a pin that no loader reads pins nothing. The validator checks the pins'
shape; this checks they are consumed, through the repo's own config loaders and run functions with
injected fakes, so nothing is downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from onebee.training.distill import load_distill_config, run_distillation
from onebee.training.dpo import load_dpo_config, run_dpo

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs" / "training" / "study002").glob("*.yaml"))


def _fake_modules() -> dict[str, Any]:
    fake_peft = mock.MagicMock()
    fake_trl = mock.MagicMock()

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake_peft.LoraConfig = FakeConfig
    fake_trl.DPOConfig = FakeConfig
    return {"peft": fake_peft, "trl": fake_trl}


def _fake_tokenizer() -> Any:
    tokenizer = mock.MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 0
    return tokenizer


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_study002_pins_reach_every_load(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)  # the configs' data paths are repo-relative
    loads: list[tuple[str, str, str]] = []
    trainer_kwargs: list[dict[str, Any]] = []

    def model_loader(name: str, revision: str) -> Any:
        loads.append(("model", name, revision))
        return mock.MagicMock()

    def tokenizer_loader(name: str, revision: str) -> Any:
        loads.append(("tokenizer", name, revision))
        return _fake_tokenizer()

    def trainer_factory(**kwargs: Any) -> Any:
        trainer_kwargs.append(kwargs)
        return mock.MagicMock()

    with mock.patch.dict(sys.modules, _fake_modules()):
        if path.name.startswith("distill"):
            config = load_distill_config(str(path))
            run_distillation(
                config,
                dry_run=True,
                tokenizer_loader=tokenizer_loader,
                trainer_factory=trainer_factory,
            )
            [kwargs] = trainer_kwargs
            trl_args = kwargs["_config_kwargs"]
            # trl loads the student from `model` with `model_init_kwargs`, and the teacher from
            # `teacher_model` with `teacher_model_revision`.
            assert kwargs["model"] == config.base_model
            assert trl_args["model_init_kwargs"]["revision"] == config.base_model_revision
            assert kwargs["_teacher_model"] == config.teacher_model
            assert trl_args["teacher_model_revision"] == config.teacher_model_revision
            expected = [("tokenizer", config.base_model, config.base_model_revision)]
        else:
            config = load_dpo_config(str(path))
            run_dpo(
                config,
                dry_run=True,
                model_loader=model_loader,
                tokenizer_loader=tokenizer_loader,
                trainer_factory=trainer_factory,
            )
            expected = [
                ("model", config.base_model, config.base_model_revision),
                ("tokenizer", config.base_model, config.base_model_revision),
            ]
    assert loads == expected
    assert all(len(revision) == 40 for _, _, revision in loads)
