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
  - distillation
  - companion
  - multimodal
  - lora
---

# onebee-gf-distill-v1

> Post-trained companion LLM: LoRA SFT → DPO → on-policy distillation from an 8B-class teacher, on top of Gemma 4 E2B — current best checkpoint in this project.

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)](https://github.com/arrogance231/small-mind-companion)

## Model Overview

`onebee-gf-distill-v1` is the current best checkpoint from **small-mind-companion**, an
open-source research project studying how much apparent capability a small (~2B effective
parameter), vision-capable language model can recover through post-training and external
memory, rather than raw parameter scale.

- **What it is**: `gemma-4-E2B-it`, post-trained via LoRA SFT → DPO → on-policy distillation,
  merged into full weights.
- **What it does**: acts as a memory-augmented companion — retrieves relevant facts from an
  external memory store and answers questions about the user grounded in that memory, rather
  than relying on raw context length or parametric recall.
- **What makes it different**: the on-policy distillation stage (student generates its own
  completions, matched to a larger local teacher's distribution via generalized JSD) is
  applied *after* DPO, not instead of it — and measured against a persona-consistency risk the
  hypothesis explicitly flagged before training (a generic, non-persona-tuned teacher could
  pull the student's style/consistency down). It didn't.
- **Base model**: `google/gemma-4-E2B-it`.
- **Teacher (distillation only)**: `google/gemma-4-E4B-it` (8B, same tokenizer/vocab as the
  E2B student — required for `trl.DistillationTrainer`).
- **Training method**: LoRA SFT (2232 examples) → LoRA DPO (2049 preference pairs) → on-policy
  distillation (2008 prompts, 125 steps), each stage chained off the previous checkpoint.

> **GGUF quantizations available**: [12-level GGUF quantizations of this checkpoint](https://huggingface.co/arrochi112/onebee-gf-distill-v1-gguf) (F16 through Q2_K, plus vision projector) for `llama.cpp`-based on-device inference. **Note: Q2_K was found broken on real generation testing — use Q3_K_S or above.**

## Model Details

| Property | Details |
|---|---|
| Model | `onebee-gf-distill-v1` |
| Parameters | ~2B effective (base) + LoRA rank 16 adapter |
| Architecture | Gemma4 (multimodal, text + vision) |
| Base Model | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Teacher Model (distillation stage) | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) (8B) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | LoRA SFT → LoRA DPO → on-policy distillation (generalized-JSD, `trl.DistillationTrainer`) |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

As a companion-persona conversational model within a memory/retrieval pipeline (this checkpoint
does not carry its own memory — pair it with the retrieval system in the
[project repo](https://github.com/arrogance231/small-mind-companion) for the evaluated
configuration). Suitable as a reference point for further post-training research (additional
distillation passes, quantization, abliteration research) given the honest limitations below.

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README](https://github.com/arrogance231/small-mind-companion) for the full research
framing before using it in any production context.

## Capabilities

- Companion-persona conversational responses conditioned on retrieved memories
- Strongest measured calibration in this project: UAR 71.25% (correctly abstains on
  unanswerable questions without over-hedging on answerable ones)
- Best measured answer accuracy: `pra_lenient` 18.59%
- Persona consistency held or improved post-distillation, by both an LLM-judge pairwise
  comparison (+7.6pp favoring this checkpoint) and an independent no-API stylometric
  self-consistency measure (0.524 vs 0.509 pre-distillation)

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("arrochi112/onebee-gf-distill-v1")
processor = AutoProcessor.from_pretrained("arrochi112/onebee-gf-distill-v1")

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
| SFT + memory (pre-distillation) | 15.30% | 70.0% |
| **distill-v1 (this checkpoint)** | **18.59%** | **71.25%** |

> This table spans two changes, not one: `distill-v1` was trained *from the DPO output*, so the
> +3.3pp `pra_lenient` gap includes the DPO stage as well as distillation. The distillation-specific
> number is the pairwise comparison below. See `reports/ERRATA.md`.

Pairwise persona-consistency: 38.1% wins for this checkpoint vs. 30.5% for its
pre-distillation predecessor (33 ties, 105 probes, dual-order judge).

Full methodology, training-time anomalies (and why they didn't predict the real-eval outcome),
and honest limitations:
[`docs/distillation_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/distillation_results.md).

## Limitations

Single seed, single data scale (2008 prompts, 125 distillation steps) — not yet a
multi-seed-confirmed result. Training-time metrics (loss, grad norm, completion-clipping rate)
looked concerning in isolation but did not predict the real-eval outcome, which is what this
model card's numbers are based on — see the full writeup for that discrepancy. This project
reports negative/inconclusive results as honestly as positive ones — read the linked docs
before assuming any number here is a clean win.

## Other Checkpoints From This Project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arrochi112/onebee-gf-distill-v1) | **This repo** — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations (of the pre-distillation checkpoint) |
| [onebee-gf-distill-v1-gguf](https://huggingface.co/arrochi112/onebee-gf-distill-v1-gguf) | GGUF quantizations of **this checkpoint** |

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
