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

# onebee-gf-sft-v0

> LoRA SFT checkpoint on gemma-4-E2B-it, Day-4 v0 scale (4 personas, 202 examples) — early-stage baseline, superseded by sft-v1.

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)](https://github.com/arjhinety/small-mind-companion)

## Model Overview

Early-scale LoRA SFT checkpoint on top of `gemma-4-E2B-it`, trained on 202 examples generated from 4 personas. This was the Day-4 baseline before the project scaled up data 10x for `sft-v1` — kept published for reproducibility of the v0-scale results, not recommended as a starting point for new work.

> **GGUF quantizations available**: this project's current-best checkpoint is also published as [quantizations (F16 reference plus 12 quant levels)\](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf) for `llama.cpp`-based on-device inference (quantized from `dpo-v1-scale`, not this checkpoint).

## Model Details

| Property | Details |
|---|---|
| Model | `onebee-gf-sft-v0` |
| Parameters | ~2B effective (base) + LoRA rank 16 adapter |
| Architecture | Gemma4 (multimodal, text + vision) |
| Base Model | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | LoRA SFT (memory-aware conversational data: persona + retrieved memories + recent turns → response) |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

Reproducing this project's v0-scale results (`docs/day4_sft_results.md`). Not recommended as a base for new work — use `onebee-gf-sft-v1` or `onebee-gf-distill-v1` instead.

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README](https://github.com/arjhinety/small-mind-companion) for the full research framing before using it in any
production context.

## Capabilities

- Companion-persona conversational responses conditioned on a small set of retrieved memories
- No meaningful preference-alignment (DPO) applied at this stage

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("arjhinety/onebee-gf-sft-v0")
processor = AutoProcessor.from_pretrained("arjhinety/onebee-gf-sft-v0")

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

| System | pra_lenient | UAR |
|---|---|---|
| B (SFT v0, no memory) | 0.16% | 16.25% |
| E (SFT v0 + memory) | 17.76% | 33.75% |

Full methodology, all numbers, and honest limitations:
[`docs/day4_sft_results.md`](https://github.com/arjhinety/small-mind-companion/blob/main/docs/day4_sft_results.md).

## Limitations

Small data scale (202 examples, 4 personas) — superseded by `sft-v1`'s 10x-larger, rebalanced dataset. Known false-abstention issues at this scale, root-caused and fixed in the v1 pass (`docs/proper_scale_results.md`, `docs/model_quirks.md` #16-17).

This project reports negative/inconclusive results as honestly as positive ones — read the
linked docs before assuming any number here is a clean win.

## Other Checkpoints From This Project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arjhinety/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arjhinety/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arjhinety/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arjhinety/onebee-gf-distill-v1) | SFT+DPO+distillation — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations |

## Citation

```bibtex
@software{small_mind_companion,
  title  = {small-mind-companion: Post-training and cognitive architecture for a small multimodal companion LLM},
  author = {arjhinety},
  year   = {2026},
  url    = {https://github.com/arjhinety/small-mind-companion}
}
```

## License

Apache-2.0, inherited from the base model (`google/gemma-4-E2B-it`).
