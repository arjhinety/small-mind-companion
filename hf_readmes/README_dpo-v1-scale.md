---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: google/gemma-4-E2B-it
tags:
  - text-generation
  - fine-tuning
  - companion
  - multimodal
  - lora
---

# onebee-gf-dpo-v1-scale

> Proper-scale LoRA DPO checkpoint on top of sft-v1 (2049 preference pairs) — pre-distillation, strongest preference-optimization signal in this project.

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)](https://github.com/arrogance231/small-mind-companion)

## Model Overview

Proper-scale DPO checkpoint on top of `sft-v1` — 2049 preference pairs (~10x `dpo-v0`'s scale), 1 epoch. Strongest and cleanest preference-optimization signal observed across every run in this project (24.7pp pairwise win-rate gap). **Superseded by `onebee-gf-distill-v1`** (adds on-policy distillation on top of this checkpoint) as the current best overall, but this remains the pre-distillation baseline used in that comparison, and the checkpoint the published GGUF quantizations are built from.

> **GGUF quantizations available**: this checkpoint is also published as [12-level GGUF quantizations](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) (F16 through Q2_K, plus vision projector) for `llama.cpp`-based on-device inference.

## Model Details

| Property | Details |
|---|---|
| Model | `onebee-gf-dpo-v1-scale` |
| Parameters | ~2B effective (base) + LoRA rank 16 adapter |
| Architecture | Gemma4 (multimodal, text + vision) |
| Base Model | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | LoRA DPO, 1 epoch, 2049 preference pairs, chained off sft-v1 |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

As a base for distillation or quantization; as a strong standalone companion checkpoint if distillation-specific behavior is not desired.

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README](https://github.com/arrogance231/small-mind-companion) for the full research framing before using it in any
production context.

## Capabilities

- Companion-persona conversational responses with strong preference alignment
- Preference alignment: 45.7% vs 21.0% pairwise win-rate over SFT-only (24.7pp gap, 105 probes)

> No full-PMB `pra_lenient`/UAR measurement exists for this checkpoint — it is evaluated pairwise
> only. The 70.0% UAR figure that appeared here in earlier revisions belongs to the SFT-v1+memory
> system, not to DPO. See `reports/ERRATA.md`.

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("arrochi112/onebee-gf-dpo-v1-scale")
processor = AutoProcessor.from_pretrained("arrochi112/onebee-gf-dpo-v1-scale")

messages = [
    {"role": "system", "content": "You are a warm AI companion who remembers this user."},
    {"role": "user", "content": "What conference did I say I was attending?"},
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Evaluation

Scored against **PMB** (Personalized Memory Benchmark), 688 adversarial probes across 8
categories, with an LLM judge under dual-order (position-bias-controlled) scoring plus a
rule-based abstention detector.

| System | pairwise win-rate | UAR |
|---|---|---|
| dpo-v1-scale | 45.7% vs 21.0% (24.7pp gap) | 70.0% |

Full methodology, all numbers, and honest limitations:
[`docs/proper_scale_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/proper_scale_results.md).

## Limitations

Single seed/run at this data scale. See `onebee-gf-distill-v1` for the further-improved current-best checkpoint.

This project reports negative/inconclusive results as honestly as positive ones — read the
linked docs before assuming any number here is a clean win.

## Other Checkpoints From This Project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arrochi112/onebee-gf-distill-v1) | SFT+DPO+distillation — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations |

## Citation

```bibtex
@software{small_mind_companion,
  title  = {small-mind-companion: Post-training and cognitive architecture for a small multimodal companion LLM},
  author = {arrogance231},
  year   = {2026},
  url    = {https://github.com/arrogance231/small-mind-companion}
}
```

## License

Apache-2.0, inherited from the base model (`google/gemma-4-E2B-it`).
