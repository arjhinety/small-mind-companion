---
language:
  - en
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model: google/gemma-4-E2B-it
tags:
  - gguf
  - llama.cpp
  - multimodal
  - companion
  - quantized
  - distillation
---

# onebee-gf-distill-v1-gguf

> GGUF quantizations (F16 reference plus 12 quant levels down to Q2_K, plus vision projector) of the current-best post-distillation companion checkpoint, for `llama.cpp`-based on-device inference.

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)](https://github.com/arjhinety/small-mind-companion)

## Model Overview

GGUF conversion and quantization of
[`onebee-gf-distill-v1`](https://huggingface.co/arjhinety/onebee-gf-distill-v1) — the current
best checkpoint from **small-mind-companion** (LoRA SFT → DPO → on-policy distillation from an
8B-class teacher, on top of `gemma-4-E2B-it`) — for `llama.cpp`-based local/on-device inference.

- **What it is**: the same weights as `onebee-gf-distill-v1`, converted to GGUF.
- **What it does**: runs the current-best companion checkpoint on-device without a Python/
  `transformers` runtime.
- **What makes it different from [`onebee-gf-dpo-v1-scale-gguf`](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf)**:
  quantized from the *post-distillation* checkpoint, not the pre-distillation one — this is the
  checkpoint the project's headline H23 results are actually about.
- **Real generation testing found Q2_K broken for this checkpoint** — garbled special-token
  output and a non-terminating repetition loop, not usable. Q3_K_S and above verified coherent
  and in-character. See [Limitations](#limitations).
- **Base model**: `google/gemma-4-E2B-it`.

## Model Details

| Property | Details |
|---|---|
| Model | `onebee-gf-distill-v1-gguf` |
| Parameters | ~2B effective (base) + merged LoRA rank 16 adapter |
| Architecture | Gemma4 (multimodal, text + vision), GGUF format |
| Base Model | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Source checkpoint | [`onebee-gf-distill-v1`](https://huggingface.co/arjhinety/onebee-gf-distill-v1) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | LoRA SFT → LoRA DPO → on-policy distillation (see source checkpoint) |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

Local/on-device inference via `llama.cpp` where a `transformers`/Python runtime isn't
available or desired. **Use Q3_K_S or above** — Q2_K was found broken on real generation
testing for this checkpoint (see Limitations).

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README](https://github.com/arjhinety/small-mind-companion) for the full research
framing before using it in any production context.

## Capabilities

- Text and vision (image) input
- 12 quant levels (F16 → Q2_K), though **Q2_K is not usable — see Limitations**
- Verified in-character companion-style generation at Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M

## Files

| File | Quant | Size | Notes |
|---|---|---|---|
| `distill-v1-f16.gguf` | F16 | 9.27 GB | Full precision, reference quality |
| `distill-v1-Q8_0.gguf` | Q8_0 | 4.95 GB | Near-lossless |
| `distill-v1-Q6_K.gguf` | Q6_K | 3.83 GB | |
| `distill-v1-Q5_K_M.gguf` | Q5_K_M | 3.62 GB | |
| `distill-v1-Q5_K_S.gguf` | Q5_K_S | 3.58 GB | |
| `distill-v1-Q5_0.gguf` | Q5_0 | 3.58 GB | |
| `distill-v1-Q4_K_M.gguf` | Q4_K_M | 3.42 GB | Verified coherent |
| `distill-v1-Q4_K_S.gguf` | Q4_K_S | 3.35 GB | Verified coherent, in-character |
| `distill-v1-Q4_0.gguf` | Q4_0 | 3.35 GB | |
| `distill-v1-Q3_K_L.gguf` | Q3_K_L | 3.27 GB | |
| `distill-v1-Q3_K_M.gguf` | Q3_K_M | 3.19 GB | Verified coherent, in-character |
| `distill-v1-Q3_K_S.gguf` | Q3_K_S | 3.10 GB | Verified coherent, in-character — **smallest verified-coherent level** |
| `distill-v1-Q2_K.gguf` | Q2_K | 2.98 GB | **Broken — do not use, see Limitations** |
| `mmproj-distill-v1-f16.gguf` | F16 | 0.99 GB | Vision projector — needed for image input, use with any of the above |

## Quick Start

### Installation

```bash
# build llama.cpp, or install a prebuilt release: https://github.com/ggml-org/llama.cpp
```

### Usage

Text:
```bash
llama-cli -m distill-v1-Q4_K_M.gguf -sys "You are a warm AI companion in an ongoing relationship with the user." -p "Hello!" -st
```

Vision (needs `--jinja`):
```bash
llama-mtmd-cli -m distill-v1-Q4_K_M.gguf \
  --mmproj mmproj-distill-v1-f16.gguf \
  --image your_image.png -p "What is in this image?" --jinja
```

## Evaluation

Real generation quality checks (companion system prompt, "What is your favorite color?"):

| Quant | Result |
|---|---|
| Q4_K_S | Coherent, in-character: *"soft white... calm and open, without being cold or stark"* |
| Q3_K_M | Coherent, in-character: *"amber—the soft, buttery glow right before sunset"* |
| Q3_K_S | Coherent, in-character: *"emerald... like the color of moss after a spring rain"* |
| Q2_K | Broken: garbled special tokens followed by a non-terminating repetition loop |

Full methodology: [`docs/quantization_results.md`](https://github.com/arjhinety/small-mind-companion/blob/main/docs/quantization_results.md).

## Limitations

- **Q2_K is broken for this checkpoint** — confirmed via real generation testing, not
  file-size/load checks. Do not use it. Confirmed NOT distillation-specific: the pre-distillation
  `dpo-v1-scale` checkpoint's own Q2_K file fails the same way — Q2_K was simply never safe for
  this model family, untested until this pass. See the full writeup.
- No accuracy/quality regression measured against the project's own PMB eval harness at each
  quant level — verified "coherent and on-topic" via real generation tests, not "measurably as
  accurate as F16."
- No importance-matrix (imatrix) calibration used for these quants.
- This is a research checkpoint from an active, in-progress open-source project.

## Other Checkpoints From This Project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arjhinety/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arjhinety/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arjhinety/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arjhinety/onebee-gf-distill-v1) | Source checkpoint for this repo — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf) | GGUF quants of the pre-distillation checkpoint |

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
