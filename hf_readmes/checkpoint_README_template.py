import sys

TEMPLATES = {
    "sft-v0": {
        "title": "onebee-gf-sft-v0",
        "desc": "Day 4 v0 LoRA SFT checkpoint (202 train examples) — the small proof-of-concept SFT run, superseded by `sft-v1`.",
        "doc": "docs/day4_sft_results.md",
    },
    "sft-v1": {
        "title": "onebee-gf-sft-v1",
        "desc": "Proper-scale, rebalanced LoRA SFT checkpoint (2232 train examples, 40 personas) — **current best SFT**.",
        "doc": "docs/proper_scale_results.md",
    },
    "dpo-v0": {
        "title": "onebee-gf-dpo-v0",
        "desc": "Week 2 DPO v0 (1 epoch, 200 preference pairs) on top of `sft-v0` — real training signal, but not distinguishable from SFT-only at this scale on real eval.",
        "doc": "docs/dpo_results.md",
    },
    "dpo-v1-4epoch": {
        "title": "onebee-gf-dpo-v1-4epoch",
        "desc": "DPO trained 4 epochs on the same 200-pair v0 dataset — an overfitting experiment (near-perfect train-set fit, real qualitative regression on at least one prompt). Not the recommended checkpoint.",
        "doc": "docs/dpo_results.md",
    },
    "dpo-v1-scale": {
        "title": "onebee-gf-dpo-v1-scale",
        "desc": "Proper-scale, rebalanced DPO checkpoint (2049 preference pairs) on top of `sft-v1`. Superseded as current-best by `distill-v1`. Also available as GGUF quantizations at `onebee-gf-dpo-v1-scale-gguf`.",
        "doc": "docs/proper_scale_results.md",
    },
}

BODY = """---
license: apache-2.0
base_model: google/gemma-4-E2B-it
tags:
  - companion
  - multimodal
  - lora
---

# {title}

**Table of contents:** [Overview](#{title}) · [Other checkpoints](#other-checkpoints-from-this-project) ·
[License](#license)

Part of **[small-mind-companion](https://github.com/arjhinety/small-mind-companion)** — an
open-source research project exploring how much apparent capability a small (~2B effective
parameter), vision-capable language model can recover through post-training, external memory, and
retrieval, rather than raw parameter scale. The project treats "the number looked good" as a
signal to investigate, not a result to trust — see the
[README's Results & Analysis](https://github.com/arjhinety/small-mind-companion#results--analysis)
for real bugs found, root-caused, and fixed along the way (not just "it worked").

{desc}

**Full results, methodology, and honest limitations**: see
[`{doc}`](https://github.com/arjhinety/small-mind-companion/blob/main/{doc}) in the project
repo. This project reports negative/inconclusive results as honestly as positive ones — read
the docs before assuming any number here is a clean win.

## Other checkpoints from this project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arjhinety/onebee-gf-sft-v0) | Day 4 v0 SFT |
| [onebee-gf-sft-v1](https://huggingface.co/arjhinety/onebee-gf-sft-v1) | Proper-scale SFT |
| [onebee-gf-dpo-v0](https://huggingface.co/arjhinety/onebee-gf-dpo-v0) | Week 2 DPO v0 |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale) | Proper-scale DPO |
| [onebee-gf-distill-v1](https://huggingface.co/arjhinety/onebee-gf-distill-v1) | SFT+DPO+distillation — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arjhinety/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations of the pre-distillation checkpoint |
| [onebee-gf-distill-v1-gguf](https://huggingface.co/arjhinety/onebee-gf-distill-v1-gguf) | GGUF quantizations of the current-best checkpoint |

## License

Apache-2.0, inherited from the base model (`google/gemma-4-E2B-it`).
"""

repo = sys.argv[1]
t = TEMPLATES[repo]
print(BODY.format(**t))
