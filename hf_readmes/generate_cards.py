"""Generates hf_readmes/README_*.md from the shared template below with real per-checkpoint
data. Run manually, then re-upload each output with hf_hub_download/upload_file -- not run in
CI, this is a one-shot authoring tool."""

from __future__ import annotations

from pathlib import Path

GITHUB = "https://github.com/arjhinety/small-mind-companion"
BASE_MODEL = "google/gemma-4-E2B-it"

OTHER_CHECKPOINTS_TABLE = """
| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arjhinety/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arjhinety/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arjhinety/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arjhinety/onebee-gf-distill-v1) | SFT+DPO+distillation — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations |
""".strip()

CHECKPOINTS = {
    "sft-v0": dict(
        repo="onebee-gf-sft-v0",
        one_liner="LoRA SFT checkpoint on gemma-4-E2B-it, Day-4 v0 scale (4 personas, 202 examples) — early-stage baseline, superseded by sft-v1.",
        parameters="~2B effective (base) + LoRA rank 16 adapter",
        training_method="LoRA SFT (memory-aware conversational data: persona + retrieved memories + recent turns → response)",
        overview=(
            "Early-scale LoRA SFT checkpoint on top of `gemma-4-E2B-it`, trained on 202 examples "
            "generated from 4 personas. This was the Day-4 baseline before the project scaled up "
            "data 10x for `sft-v1` — kept published for reproducibility of the v0-scale results, "
            "not recommended as a starting point for new work."
        ),
        dataset_link="",
        intended_use=(
            "Reproducing this project's v0-scale results (`docs/day4_sft_results.md`). Not "
            "recommended as a base for new work — use `onebee-gf-sft-v1` or "
            "`onebee-gf-distill-v1` instead."
        ),
        capabilities=[
            "Companion-persona conversational responses conditioned on a small set of retrieved memories",
            "No meaningful preference-alignment (DPO) applied at this stage",
        ],
        eval_table=(
            "| System | pra_lenient | UAR |\n|---|---|---|\n"
            "| B (SFT v0, no memory) | 0.16% | 16.25% |\n"
            "| E (SFT v0 + memory) | 17.76% | 33.75% |"
        ),
        results_doc="docs/day4_sft_results.md",
        limitations=(
            "Small data scale (202 examples, 4 personas) — superseded by `sft-v1`'s 10x-larger, "
            "rebalanced dataset. Known false-abstention issues at this scale, root-caused and "
            "fixed in the v1 pass (`docs/proper_scale_results.md`, `docs/model_quirks.md` #16-17)."
        ),
    ),
    "sft-v1": dict(
        repo="onebee-gf-sft-v1",
        one_liner="Proper-scale LoRA SFT checkpoint on gemma-4-E2B-it (40 personas, 2232 examples) — current best SFT-only checkpoint.",
        parameters="~2B effective (base) + LoRA rank 16 adapter",
        training_method="LoRA SFT, 2 epochs, batch 8 / grad-accum 4",
        overview=(
            "Proper-scale LoRA SFT checkpoint — 10x the data of `sft-v0` (2232 train examples, "
            "40 personas, memory-aware conversational format), 2 epochs. Best SFT-only "
            "checkpoint in this project; the DPO and distillation checkpoints chain off this one."
        ),
        dataset_link="",
        intended_use=(
            "As a base for further post-training (DPO/distillation), or for studying the "
            "isolated effect of SFT before preference optimization is applied."
        ),
        capabilities=[
            "Companion-persona conversational responses conditioned on retrieved memories",
            "Improved abstention calibration over sft-v0 after a documented bug-fix cycle",
        ],
        eval_table=(
            "| System | pra_lenient | UAR |\n|---|---|---|\n"
            "| SFT v1 + memory | 15.30% | 70.0% |\n"
            "| + DPO (dpo-v1-scale) | — | — (evaluated pairwise only: 45.7% vs 21.0%, 24.7pp gap) |"
        ),
        results_doc="docs/proper_scale_results.md",
        limitations=(
            "SFT alone, no preference optimization — use `onebee-gf-dpo-v1-scale` or "
            "`onebee-gf-distill-v1` for the strongest results. Single seed/run."
        ),
    ),
    "dpo-v0": dict(
        repo="onebee-gf-dpo-v0",
        one_liner="Week-2 DPO checkpoint on top of sft-v0 (200 preference pairs, 1 epoch) — early-stage baseline, superseded by dpo-v1-scale.",
        parameters="~2B effective (base) + LoRA rank 16 adapter",
        training_method="LoRA DPO, 1 epoch, 200 preference pairs",
        overview=(
            "Early-scale DPO checkpoint on top of `sft-v0`, 200 preference pairs, 1 epoch. "
            "Kept published for reproducibility of the v0-scale preference-optimization "
            "results; superseded by `dpo-v1-scale`'s 10x-larger dataset."
        ),
        dataset_link="",
        intended_use="Reproducing this project's v0-scale DPO results. Not recommended as a starting point for new work.",
        capabilities=[
            "Companion-persona responses with early-stage preference alignment",
        ],
        eval_table="See `docs/dpo_results.md` for the full v0-scale pairwise win-rate numbers.",
        results_doc="docs/dpo_results.md",
        limitations="Small preference dataset (200 pairs) — superseded by `dpo-v1-scale`.",
    ),
    "dpo-v1-4epoch": dict(
        repo="onebee-gf-dpo-v1-4epoch",
        one_liner="DPO on the v0 preference dataset run for 4 epochs — a deliberate overfitting experiment, not a recommended checkpoint.",
        parameters="~2B effective (base) + LoRA rank 16 adapter",
        training_method="LoRA DPO, 4 epochs (vs. 1 epoch for dpo-v0), same 200-pair v0 dataset",
        overview=(
            "A deliberate overfitting experiment: the same 200-pair v0 preference dataset as "
            "`dpo-v0`, but trained for 4 epochs instead of 1, to study how DPO degrades when "
            "over-trained on a small preference set. Published for reproducibility of that "
            "specific experiment, not as a general-purpose checkpoint."
        ),
        dataset_link="",
        intended_use="Studying DPO overfitting behavior on small preference datasets. Not recommended for deployment or as a training base.",
        capabilities=[
            "Same base capabilities as dpo-v0, with observed overfitting artifacts from extended training"
        ],
        eval_table="See `docs/dpo_results.md` for the overfitting-experiment comparison against dpo-v0.",
        results_doc="docs/dpo_results.md",
        limitations="Explicitly overfit by design — do not use this checkpoint as a general-purpose companion model.",
    ),
    "dpo-v1-scale": dict(
        repo="onebee-gf-dpo-v1-scale",
        one_liner="Proper-scale LoRA DPO checkpoint on top of sft-v1 (2049 preference pairs) — pre-distillation, strongest preference-optimization signal in this project.",
        parameters="~2B effective (base) + LoRA rank 16 adapter",
        training_method="LoRA DPO, 1 epoch, 2049 preference pairs, chained off sft-v1",
        overview=(
            "Proper-scale DPO checkpoint on top of `sft-v1` — 2049 preference pairs (~10x "
            "`dpo-v0`'s scale), 1 epoch. Strongest and cleanest preference-optimization signal "
            "observed across every run in this project (24.7pp pairwise win-rate gap). "
            "**Superseded by `onebee-gf-distill-v1`** (adds on-policy distillation on top of "
            "this checkpoint) as the current best overall, but this remains the pre-distillation "
            "baseline used in that comparison, and the checkpoint the published GGUF "
            "quantizations are built from."
        ),
        dataset_link="",
        intended_use="As a base for distillation or quantization; as a strong standalone companion checkpoint if distillation-specific behavior is not desired.",
        capabilities=[
            "Companion-persona conversational responses with strong preference alignment",
            "Preference alignment: 45.7% vs 21.0% pairwise win-rate over SFT-only (24.7pp gap)",
        ],
        eval_table=(
            "| System | pairwise win-rate | UAR |\n|---|---|---|\n"
            "| dpo-v1-scale | 45.7% vs 21.0% (24.7pp gap) | not measured |"
        ),
        results_doc="docs/proper_scale_results.md",
        limitations="Single seed/run at this data scale. See `onebee-gf-distill-v1` for the further-improved current-best checkpoint.",
    ),
}

TEMPLATE = """---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: {base_model}
tags:
  - text-generation
  - fine-tuning
  - companion
  - multimodal
  - lora
{extra_tags}---

# {repo}

> {one_liner}

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)]({github})

## Model Overview

{overview}

## Model Details

| Property | Details |
|---|---|
| Model | `{repo}` |
| Parameters | {parameters} |
| Architecture | Gemma4 (multimodal, text + vision) |
| Base Model | [`{base_model}`](https://huggingface.co/{base_model}) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | {training_method} |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

{intended_use}

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README]({github}) for the full research framing before using it in any
production context.

## Capabilities

{capabilities}

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("arjhinety/{repo}")
processor = AutoProcessor.from_pretrained("arjhinety/{repo}")

messages = [
    {{"role": "system", "content": "You are a warm AI companion who remembers this user."}},
    {{"role": "user", "content": "What conference did I say I was attending?"}},
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Evaluation

Scored against **PMB** (Personalized Memory Benchmark), 688 adversarial probes across 8
categories, with an LLM judge under dual-order (position-bias-controlled) scoring plus a
rule-based abstention detector.

{eval_table}

Full methodology, all numbers, and honest limitations:
[`{results_doc}`]({github}/blob/main/{results_doc}).

## Limitations

{limitations}

This project reports negative/inconclusive results as honestly as positive ones — read the
linked docs before assuming any number here is a clean win.

## Other Checkpoints From This Project

{other_checkpoints}

## Citation

```bibtex
@software{{small_mind_companion,
  title  = {{small-mind-companion: Post-training and cognitive architecture for a small multimodal companion LLM}},
  author = {{arjhinety}},
  year   = {{2026}},
  url    = {{{github}}}
}}
```

## License

Apache-2.0, inherited from the base model (`{base_model}`).
"""


def render(name: str, fields: dict) -> str:
    capabilities_md = "\n".join(f"- {c}" for c in fields["capabilities"])
    return TEMPLATE.format(
        repo=fields["repo"],
        one_liner=fields["one_liner"],
        base_model=BASE_MODEL,
        extra_tags="",
        github=GITHUB,
        overview=fields["overview"],
        parameters=fields["parameters"],
        training_method=fields["training_method"],
        intended_use=fields["intended_use"],
        capabilities=capabilities_md,
        eval_table=fields["eval_table"],
        results_doc=fields["results_doc"],
        limitations=fields["limitations"],
        other_checkpoints=OTHER_CHECKPOINTS_TABLE,
    )


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    for name, fields in CHECKPOINTS.items():
        text = render(name, fields)
        out_path = out_dir / f"README_{name}.md"
        out_path.write_text(text)
        print("wrote", out_path)
