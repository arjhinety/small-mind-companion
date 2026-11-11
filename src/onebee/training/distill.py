from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import BaseModel


class DistillationTrainingConfig(BaseModel):
    base_model: str
    base_model_revision: str = "main"
    teacher_model: str
    teacher_model_revision: str = "main"
    train_file: str
    val_file: str
    output_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    learning_rate: float = 1e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_completion_length: int = 128
    temperature: float = 1.0
    beta: float = 1.0
    bf16: bool = True
    seed: int = 1337
    report_to: str = "none"
    run_name: str | None = None


def load_distill_config(path: str) -> DistillationTrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DistillationTrainingConfig(**raw)


def build_lora_config(config: DistillationTrainingConfig):
    from peft import LoraConfig

    if config.lora_target_modules == "all-linear":
        target_modules: str | list[str] = "all-linear"
    else:
        target_modules = [m.strip() for m in config.lora_target_modules.split(",")]

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


def _load_jsonl(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run_distillation(
    config: DistillationTrainingConfig,
    dry_run: bool = False,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    trainer_factory: Callable[..., Any] | None = None,
) -> None:
    # Unlike sft.py/dpo.py, DistillationTrainer loads both student and teacher itself from
    # string model IDs (see its docstring) -- no separate model_loader injection point needed.
    if tokenizer_loader is None:

        def tokenizer_loader(model_name: str, revision: str):
            from transformers import AutoTokenizer

            # DistillationTrainer requires left-padding for generation-based training.
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, padding_side="left"
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer

    if trainer_factory is None:

        def trainer_factory(**kwargs):
            from trl import DistillationConfig as TrlDistillationConfig
            from trl import DistillationTrainer

            base_kwargs = kwargs.pop("_config_kwargs")
            trl_config = TrlDistillationConfig(**base_kwargs)

            from datasets import Dataset

            if isinstance(kwargs.get("train_dataset"), list):
                kwargs["train_dataset"] = Dataset.from_list(kwargs["train_dataset"])
            if isinstance(kwargs.get("eval_dataset"), list):
                kwargs["eval_dataset"] = Dataset.from_list(kwargs["eval_dataset"])

            # `teacher_model_name_or_path` in DistillationConfig is metadata only -- the
            # actual `teacher_model` constructor argument must be passed separately, or
            # trainer.teacher_model stays None and every training step crashes with
            # AttributeError: 'NoneType' object has no attribute 'eval' (found via a real
            # training run, not caught by a dry-run since teacher loading is lazy).
            teacher_model = kwargs.pop("_teacher_model")
            return DistillationTrainer(args=trl_config, teacher_model=teacher_model, **kwargs)

    tokenizer = tokenizer_loader(config.base_model, config.base_model_revision)
    train_dataset = _load_jsonl(config.train_file)
    val_dataset = _load_jsonl(config.val_file)

    lora_config = build_lora_config(config)

    config_kwargs = dict(
        output_dir=config.output_dir,
        teacher_model_name_or_path=config.teacher_model,
        teacher_model_revision=config.teacher_model_revision,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.beta,
        bf16=config.bf16,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
        # Defaults to float32 otherwise (see DistillationTrainer's docstring) -- would make
        # the 8B teacher use ~32GB just for weights instead of ~16GB.
        # `revision` too: trl loads a string student with `model_init_kwargs` and reads its revision
        # from there only, so without it a Hub student would load from "main" whatever
        # `base_model_revision` says (STUDIES item 13). Ignored for a local folder.
        model_init_kwargs={"dtype": "bfloat16", "revision": config.base_model_revision},
        teacher_model_init_kwargs={"dtype": "bfloat16"},
    )

    trainer = trainer_factory(
        model=config.base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        _config_kwargs=config_kwargs,
        _teacher_model=config.teacher_model,
    )

    if dry_run:
        return

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, "adapter"))
    merged_model = trainer.model.merge_and_unload()
    merged_dir = os.path.join(config.output_dir, "merged")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.base_model, revision=config.base_model_revision
        )
        processor.save_pretrained(merged_dir)
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="On-policy distillation training (H23)")
    parser.add_argument("--config", required=True, help="Path to YAML distillation config file")
    parser.add_argument("--dry-run", action="store_true", help="Build everything but skip training")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_distill_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    run_distillation(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
