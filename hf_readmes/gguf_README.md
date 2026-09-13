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
---

# onebee-gf-dpo-v1-scale-gguf

> GGUF quantizations (F16 reference plus 12 quant levels down to Q2_K, plus vision projector) of the post-trained companion checkpoint, for `llama.cpp`-based on-device inference.

[![Project](https://img.shields.io/badge/GitHub-small--mind--companion-blue)](https://github.com/arjhinety/small-mind-companion)

## Model Overview

GGUF conversion and quantization of
[`onebee-gf-dpo-v1-scale`](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) — a
`gemma-4-E2B-it` base, LoRA-fine-tuned via memory-aware SFT then DPO on a persona-consistency
preference objective — from **small-mind-companion**, an open-source research project exploring
how much apparent capability a small (~2B effective parameter), vision-capable language model
can recover through post-training, external memory, and retrieval, rather than raw parameter
scale.

- **What it is**: the same weights as `onebee-gf-dpo-v1-scale`, converted to GGUF for
  `llama.cpp`-based local/on-device inference (CPU or GPU).
- **What it does**: runs the companion model on-device without a Python/`transformers`
  runtime, at a range of size/quality tradeoffs.
- **What makes it different**: real generation and multimodal (vision) capability were
  verified post-quantization, not just file-size/load checks — four real `llama.cpp` tooling
  bugs were found and fixed along the way (see [Limitations](#limitations)).
- **Base model**: `google/gemma-4-E2B-it`.
- **Training method** (inherited from the source checkpoint): LoRA SFT → LoRA DPO (this repo
  does not include the later distillation stage — see
  [`onebee-gf-distill-v1-gguf`](https://huggingface.co/arjhinety/onebee-gf-distill-v1-gguf) for
  GGUF quants of the current-best, post-distillation checkpoint).

## Model Details

| Property | Details |
|---|---|
| Model | `onebee-gf-dpo-v1-scale-gguf` |
| Parameters | ~2B effective (base) + merged LoRA rank 16 adapter |
| Architecture | Gemma4 (multimodal, text + vision), GGUF format |
| Base Model | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Source checkpoint | [`onebee-gf-dpo-v1-scale`](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) |
| Language | English |
| Context Length | 131,072 tokens (inherited from base model) |
| Training Method | LoRA SFT → LoRA DPO (see source checkpoint) |
| License | Apache-2.0 (inherited from base model) |

## Intended Use

### Intended Use

Local/on-device inference via `llama.cpp` where a `transformers`/Python runtime isn't
available or desired — e.g. CPU-only or mobile deployment. Pick a quant level from the size/
speed tradeoff table below.

### Out-of-Scope Use

Not evaluated or intended for: safety-critical decisions, medical/legal/financial advice, or any
deployment where a wrong or overconfident answer causes real harm. This is a research artifact
from an open-source project studying post-training and memory architecture on small models — see
[the project README](https://github.com/arjhinety/small-mind-companion) for the full research
framing before using it in any production context.

## Capabilities

- Text and vision (image) input, verified with real generation tests post-quantization
- 12 quant levels (F16 → Q2_K) covering the full size/quality/speed tradeoff space
- Q4_K_M: 63% smaller than F16 with 2.2x faster CPU token generation, no observed coherence loss

## Files

| File | Quant | Size | Notes |
|---|---|---|---|
| `onebee-dpo-v1-scale-f16.gguf` | F16 | 8.64 GiB | Full precision, reference quality |
| `onebee-dpo-v1-scale-Q8_0.gguf` | Q8_0 | 4.61 GiB | Near-lossless || `onebee-dpo-v1-scale-Q6_K.gguf` | Q6_K | 3.57 GiB | |
| `onebee-dpo-v1-scale-Q5_K_M.gguf` | Q5_K_M | 3.37 GiB | |
| `onebee-dpo-v1-scale-Q5_K_S.gguf` | Q5_K_S | 3.34 GiB | |
| `onebee-dpo-v1-scale-Q5_0.gguf` | Q5_0 | 3.34 GiB | |
| `onebee-dpo-v1-scale-Q4_K_M.gguf` | Q4_K_M | 3.18 GiB | **Recommended default** — best size/quality tradeoff |
| `onebee-dpo-v1-scale-Q4_K_S.gguf` | Q4_K_S | 3.12 GiB | |
| `onebee-dpo-v1-scale-Q4_0.gguf` | Q4_0 | 3.12 GiB | |
| `onebee-dpo-v1-scale-Q3_K_L.gguf` | Q3_K_L | 3.05 GiB | |
| `onebee-dpo-v1-scale-Q3_K_M.gguf` | Q3_K_M | 2.97 GiB | |
| `onebee-dpo-v1-scale-Q3_K_S.gguf` | Q3_K_S | 2.89 GiB | |
| `onebee-dpo-v1-scale-Q2_K.gguf` | Q2_K | 2.78 GiB | **Broken — confirmed via generation testing, do not use** |
| `mmproj-onebee-dpo-v1-scale-f16.gguf` | F16 | 940 MiB | Vision projector — needed for image input, use with any of the above |

No importance-matrix (imatrix) calibration was used for these quants. An imatrix-calibrated
requantization was produced separately and is held in a private companion repo pending a full
perplexity comparison — see
[`docs/quantization_results.md`](https://github.com/arjhinety/small-mind-companion/blob/main/docs/quantization_results.md).

## Quick Start

### Installation

```bash
# build llama.cpp, or install a prebuilt release: https://github.com/ggml-org/llama.cpp
```

### Usage

Text:
```bash
llama-cli -m onebee-dpo-v1-scale-Q4_K_M.gguf -p "Hello!" -st
```

Vision (needs `--jinja` — this model's chat template isn't supported by the default non-jinja
parser):
```bash
llama-mtmd-cli -m onebee-dpo-v1-scale-Q4_K_M.gguf \
  --mmproj mmproj-onebee-dpo-v1-scale-f16.gguf \
  --image your_image.png -p "What is in this image?" --jinja
```

## Evaluation

Real benchmark numbers, CPU, 30 threads, `llama-bench`:

| Quant | Size | Prompt (pp512) | Generation (tg128) |
|---|---|---|---|
| F16 | 8.62 GiB | 585.07 ± 0.44 t/s | 26.15 ± 0.28 t/s |
| Q8_0 | 4.59 GiB | 492.33 ± 1.21 t/s | 43.07 ± 0.24 t/s |
| Q4_K_M | 3.17 GiB | 633.00 ± 1.22 t/s | 58.00 ± 0.51 t/s |

> The sizes in this table are `llama-bench`'s own report; the **Files** table above lists the
> on-disk sizes, which differ slightly (e.g. F16 8.62 vs 8.64 GiB). No `llama-bench` output or
> GGUF file is committed to the project repository, so these figures are reported measurements
> rather than independently re-derivable ones — see `reports/ERRATA.md` E5 and E6.

Full methodology and generation-quality checks:
[`docs/quantization_results.md`](https://github.com/arjhinety/small-mind-companion/blob/main/docs/quantization_results.md).

## Limitations

- **Q2_K is broken — confirmed via real generation testing (2026-08-17), not usable.** Produces
  garbled special-token output followed by a non-terminating repetition loop. Use Q3_K_S or
  above. See `docs/quantization_results.md` for the full test and comparison against
  `distill-v1`'s Q2_K (same failure — not checkpoint-specific).
- No accuracy/quality regression measured against the project's own PMB eval harness at each
  quant level — verified "coherent and on-topic" via real generation tests, not "measurably as
  accurate as F16." See `docs/quantization_results.md` for the exact checks run.
- Quantized from the pre-distillation checkpoint (`dpo-v1-scale`), not the current-best
  `distill-v1` — see [`onebee-gf-distill-v1-gguf`](https://huggingface.co/arjhinety/onebee-gf-distill-v1-gguf)
  for GGUF quants of the post-distillation checkpoint.
- This is a research checkpoint from an active, in-progress open-source project — expect real,
  documented limitations (see the linked docs) rather than a polished consumer product.

## Other Checkpoints From This Project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arjhinety/onebee-gf-sft-v0) | Day 4 v0 SFT (202 examples) |
| [onebee-gf-sft-v1](https://huggingface.co/arjhinety/onebee-gf-sft-v1) | Proper-scale SFT (2232 examples) |
| [onebee-gf-dpo-v0](https://huggingface.co/arjhinety/onebee-gf-dpo-v0) | Week 2 DPO v0 (200 pairs) |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) | Proper-scale DPO — source checkpoint for this repo |
| [onebee-gf-distill-v1](https://huggingface.co/arjhinety/onebee-gf-distill-v1) | SFT+DPO+distillation — current best overall |
| [onebee-gf-distill-v1-gguf](https://huggingface.co/arjhinety/onebee-gf-distill-v1-gguf) | GGUF quantizations of the current-best checkpoint |

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
