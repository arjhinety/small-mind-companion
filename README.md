# small-mind-companion

Post-training and cognitive-architecture research on a small (~2B effective-parameter) multimodal LLM, evaluated on adversarial long-horizon personalized memory.

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Tests](https://img.shields.io/badge/tests-504%20passing-brightgreen)

## Overview

Small on-device language models generally can't sustain a persona across years of conversation the way a much larger model with a huge context window can — they either forget, hallucinate memories, or answer confidently when they shouldn't. This project asks how much of that gap can be closed without scaling parameters: by pairing a ~2B-parameter vision-capable model with an external memory/retrieval system, LoRA post-training (SFT → DPO → on-policy distillation), and quantization for on-device inference, then measuring the result against an adversarial benchmark built specifically to catch abstention failures and false memories, not just recall accuracy.

Every result below links to a doc with full methodology and honest limitations, including negative/inconclusive findings reported as such. Full research-question hierarchy and hypotheses: [`docs/research_questions.md`](docs/research_questions.md).

## Studies

This project is organised into studies. **Study 001 is frozen** (2026-09-13, tag `study-001`): its
evidence is hash-pinned and its write-ups stop changing. Work after the freeze is Study 002.

**→ [Read Study 001 at small-mind.arjhinety.com](https://small-mind.arjhinety.com)** — the write-up,
charts, and limitations.

- [`docs/STUDIES.md`](docs/STUDIES.md) — what Study 001 asked, what it found, and what Study 002 covers
- [`reports/data/study-001-freeze.json`](reports/data/study-001-freeze.json) — every artifact the findings rest on, pinned by SHA-256
- [`reports/ERRATA.md`](reports/ERRATA.md) — 32 claims the committed artifacts did not support, and their corrections
- [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) — the rules derived from those mistakes
- [`docs/AUDIT.md`](docs/AUDIT.md) — how the freeze, the claims matrix (`registry/claims.jsonl`) and `scripts/validate.py` fit together

```
make freeze-check     # fails if any frozen Study 001 artifact has changed
make validate         # freeze + hashes + recomputed metrics + claims matrix
uv run python scripts/recompute_hashes.py   # re-checks all 11 dataset hashes
```

## Results

| Model | Dataset | Method | pra_lenient | UAR |
|---|---|---|---|---|
| `gemma-4-E2B-it` | — | raw model, no memory | 0.16% | 13.75% |
| `gemma-4-E2B-it` | SFT v0 (202 ex) | LoRA SFT, no memory | 0.16% | 16.25% |
| `gemma-4-E2B-it` | — | + hybrid retrieval memory (k=8), no SFT | 15.13% | 8.75% |
| `gemma-4-E2B-it` | SFT v0 + memory | LoRA SFT + memory | 17.76% | 33.75% |
| `gemma-4-E2B-it` | SFT v1 (2232 ex) | proper-scale LoRA SFT + memory | 15.30% | 70.0% |
| `gemma-4-E2B-it` | SFT v1 + DPO v1 (2049 pairs) | + LoRA DPO. Evaluated pairwise only (24.7pp C-vs-E) — no full-PMB metrics for this checkpoint | — | — |
| `gemma-4-E2B-it` | + distill v1 (2008 prompts) | + on-policy distillation from `gemma-4-E4B-it` | 18.59% | 71.25% |

`pra_lenient` and UAR are measured against **PMB** (Personalized Memory Benchmark), 688 adversarial probes across 8 categories (factual, episodic, temporal, preference, continuity, outdated-fact, distractor, unanswerable). Full writeups: [`docs/proper_scale_results.md`](docs/proper_scale_results.md) (its headline table predates the bug fixes below — see its "Rebalancing fix" section for the final numbers), [`docs/day3_memory_results.md`](docs/day3_memory_results.md), [`docs/day4_sft_results.md`](docs/day4_sft_results.md), [`docs/dpo_results.md`](docs/dpo_results.md), [`docs/distillation_results.md`](docs/distillation_results.md).

## Architecture

```
User message + image
        │
        ▼
┌─────────────────────┐      ┌──────────────────────────┐
│  Hybrid Retriever    │◄────►│  Memory Store (SQLite)   │
│  dense (e5-small) +  │      │  short-term / episodic /  │
│  BM25, RRF fusion     │      │  semantic tiers           │
└──────────┬───────────┘      └──────────────────────────┘
           │ top-k memories
           ▼
┌─────────────────────┐
│  Context Builder      │  persona card + memories + recent turns,
│  (token-budgeted)     │  assembled as real system/user role turns
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  gemma-4-E2B-it       │  LoRA SFT → DPO → distillation post-trained,
│  (multimodal)         │  quantized to GGUF for on-device inference
└──────────────────────┘
```

## Quick Start

```bash
git clone https://github.com/arrogance231/small-mind-companion.git
cd small-mind-companion
uv sync             # base install (CPU-only, lint/test)
```

Smallest runnable example — generate a response using retrieved memory:

```python
from onebee.inference.engine import HFEngine, GenerationConfig

engine = HFEngine("arrochi112/onebee-gf-distill-v1")
response = engine.generate([
    {"role": "system", "content": "You are a warm AI companion who remembers this user."},
    {"role": "user", "content": "What conference did I say I was attending?"},
], GenerationConfig(max_new_tokens=128))
print(response)
```

For the full memory-retrieval-augmented pipeline (not just a raw checkpoint), see `run_system_e_distill.py` at the repo root.

## Dataset

Two data families, both versioned in `data/` and pinned by [`reports/data/study-001-freeze.json`](reports/data/study-001-freeze.json):

- **PMB (evaluation)**: `data/benchmarks/pmb_v0_full/` — 688 adversarial probes across 8 personas (86 each), each probe categorized (factual/episodic/temporal/preference/continuity/outdated-fact/distractor/unanswerable) with a gold answer and supporting-memory IDs. Note: the `acceptable_alternatives` field exists in every record but is currently unpopulated, which is why `pra_strict` is ~0 throughout and `pra_lenient` (judge-scored) is the reported metric.
- **SFT / DPO / distillation (training)**: `data/sft/v1/`, `data/dpo/v1_scale/`, `data/distill/v1/` — each with a `DATASHEET.md` describing generation methodology, class balance, and known caveats (e.g. not human-reviewed, not yet contamination-checked against PMB at generation time — verify with `scripts/check_contamination.py` before reusing).

Splits: SFT v1 is 2232 train / 248 val; DPO v1_scale is 2049 train / 228 val; distillation v1 is 2008 train / 224 val prompts (prompt-only — the student generates its own completions on-policy).

Personas used for training data are generated disjoint from the PMB eval personas (separate seed, separate output directory) — see each `DATASHEET.md` for the exact construction method.

## Training

All runs so far were single-GPU LoRA fine-tunes on a rented workstation (see [Hardware](#hardware)) — no multi-GPU or distributed training has been needed at this model/data scale.

### Single GPU

```bash
uv sync --extra gpu --extra dev

# SFT
uv run python -m onebee.training.sft --config configs/training/sft_v1.yaml

# DPO (chains off the SFT output)
uv run python -m onebee.training.dpo --config configs/training/dpo_v1_scale.yaml

# On-policy distillation (chains off the DPO output)
uv run python -m onebee.training.distill --config configs/training/distill_v1.yaml
```

### Multi-GPU / Distributed Training

Not implemented — out of scope at the current ~2B-parameter, LoRA-rank-16 training scale. Revisit if a full-parameter or larger-model training pass is added later.

## Configuration

Every run is a composed YAML config under `configs/training/` — no hardcoded hyperparameters in code. The values that matter most:

| Field | SFT v1 | DPO v1_scale | Distill v1 |
|---|---|---|---|
| `lora_r` / `lora_alpha` | 16 / 32 | 16 / 32 | 16 / 32 |
| `learning_rate` | 1e-4 | (DPO default) | 1e-6 (conservative — refining an already-trained checkpoint) |
| `num_train_epochs` | 2.0 | 1.0 | 1.0 (125 steps) |
| `per_device_train_batch_size` | 8 | — | 2 (on-policy generation is more expensive per step) |
| `max_seq_length` / `max_completion_length` | 2048 | — | 128 |
| `teacher_model` | — | — | `google/gemma-4-E4B-it` (8B, same tokenizer/vocab as the student) |
| `seed` | 1337 | — | — |

Full configs: [`configs/training/sft_v1.yaml`](configs/training/sft_v1.yaml), [`configs/training/dpo_v1_scale.yaml`](configs/training/dpo_v1_scale.yaml), [`configs/training/distill_v1.yaml`](configs/training/distill_v1.yaml).

## Evaluation

```bash
uv sync --extra judge --extra dev
export OPENAI_API_KEY=...  # LLM judge for PRA/UAR scoring and pairwise comparisons

uv run python run_system_e_distill.py          # generate + score the current-best system against PMB
uv run python compare_c_vs_f_distill.py        # pairwise persona-consistency, pre- vs post-distillation
```

Metrics are computed with an LLM judge under dual-order (position-bias-controlled) scoring plus a rule-based abstention detector — see [`src/onebee/evaluation/`](src/onebee/evaluation/) for the scoring implementation and [`docs/proper_scale_results.md`](docs/proper_scale_results.md) for the full methodology.

## Checkpoints

Trained checkpoints are published to Hugging Face Hub, not committed to git (`outputs/` is gitignored). Resume/load any stage from its HF repo:

```bash
hf download arrochi112/onebee-gf-distill-v1 --local-dir outputs/distill/v1/merged
```

| Checkpoint | Description |
|---|---|
| [`onebee-gf-sft-v0`](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT (202 train examples) |
| [`onebee-gf-sft-v1`](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT, rebalanced (2232 train examples) |
| [`onebee-gf-dpo-v0`](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 (1 epoch, 200 pairs) |
| [`onebee-gf-dpo-v1-4epoch`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO v0 data, 4 epochs (overfitting experiment) |
| [`onebee-gf-dpo-v1-scale`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [`onebee-gf-distill-v1`](https://huggingface.co/arrochi112/onebee-gf-distill-v1) | SFT+DPO+distillation (H23) — **current best overall** |
| [`onebee-gf-dpo-v1-scale-gguf`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations (of the pre-distillation checkpoint) — **Q2_K broken, use Q3_K_S or above** |
| [`onebee-gf-distill-v1-gguf`](https://huggingface.co/arrochi112/onebee-gf-distill-v1-gguf) | GGUF quantizations of the current-best checkpoint (F16 through Q2_K + mmproj) — **Q2_K found broken on generation testing, use Q3_K_S or above** |

Repo names still carry the project's earlier `onebee-gf` name (predates a repo rename to `small-mind-companion`) — renaming them would mean recreating and re-uploading tens of GB per repo, not worth it for a naming-only change.

## Experiments

| # | Hypothesis | Result | Detail |
|---|---|---|---|
| H1, H4, H5 | Memory retrieval and SFT each help; combined beats either alone | Confirmed at v0 scale | [`docs/day3_memory_results.md`](docs/day3_memory_results.md) |
| H10 | Retrieved-memory count k has an inverted-U optimum | Peaks at k=8 | [`docs/day3_memory_results.md`](docs/day3_memory_results.md) |
| H6, H7 | DPO improves preference alignment over SFT alone | Confirmed — 24.7pp pairwise win-rate gap at proper scale | [`docs/dpo_results.md`](docs/dpo_results.md), [`docs/proper_scale_results.md`](docs/proper_scale_results.md) |
| H16/H17-adjacent | 10x data scale improves calibration (UAR) | Confirmed after fixing 2 real bugs that initially masked the improvement | [`docs/proper_scale_results.md`](docs/proper_scale_results.md) |
| H23 | On-policy distillation from a larger teacher improves quality without degrading persona consistency | Confirmed — `pra_lenient` +3.3pp, UAR flat, persona-consistency favored the distilled model | [`docs/distillation_results.md`](docs/distillation_results.md) |
| — | GGUF quantization preserves generation quality | Confirmed down to Q3_K_S by manual generation checks (Q2_K broken) | [`docs/quantization_results.md`](docs/quantization_results.md) |
| H22 | Abliteration increases compliance at the cost of judgment quality | Eval harness built, not yet run | [`docs/research_questions.md`](docs/research_questions.md) |
| — | ORPO as an alternative to DPO | Blocked — not supported by the pinned `trl` version | [`docs/model_quirks.md`](docs/model_quirks.md) #15 |

## Hardware

All LoRA training and on-policy distillation runs were done on a single rented workstation GPU (NVIDIA RTX PRO 6000 Blackwell class, ~96GB VRAM) — sufficient headroom for LoRA fine-tuning and on-policy distillation of a ~2B/8B student/teacher pair without offloading. GGUF conversion and quantization were built separately: CPU-only on a box with no system-wide CUDA toolkit, and on Modal (`RTX-PRO-6000`) for the distill-v1 pass. Quantization benchmarks (generation speed table below) were run CPU-only, since the deployment target is on-device/mobile inference, not GPU-served inference.

| Quant | Size | Generation speed (CPU, 30 threads) |
|---|---|---|
| F16 | 8.64 GiB | 26.15 t/s |
| Q8_0 | 4.61 GiB | 43.07 t/s |
| **Q4_K_M** | **3.18 GiB** | **58.00 t/s** (recommended default) |

Full quantization spread (F16 reference plus 12 quant levels down to Q2_K) and methodology: [`docs/quantization_results.md`](docs/quantization_results.md) — note that no `llama-bench`/`llama-imatrix` output is committed to this repo, so the sizes and speeds below are restated from the original run logs rather than re-derivable from a tracked artifact.

## Project Structure

```
src/onebee/           installable package: memory, retrieval, context, state, inference, training, evaluation
configs/training/      composed YAML configs, one per experiment
scripts/                bake-off, benchmark construction, contamination checking, the Study 001 freeze and the audit validator
data/                    versioned benchmarks + SFT/DPO/distillation datasets, each with a DATASHEET.md
results/                 canonical numbers, versioned by pass (results/v0/, results/v1_scale/, ...)
mobile/                  on-device runtime build/convert scripts (llama.cpp/MLC/ExecuTorch)
docs/                    ADRs, results writeups, and the full environment/bug log
tests/                   504 unit tests, run in CI
```

## Results & Analysis

- **Root-caused a calibration regression to two independent bugs, then fixed both and re-verified end-to-end.** Scaling the training data 10x initially appeared to make the model *worse* at abstaining on unanswerable questions. Investigation found: (1) a naive text-based dedup step in the data-generation script was silently collapsing ~227 intended abstention training examples down to 1; (2) after fixing that, the eval harness's own abstention detector didn't recognize the model's newly-correct phrasing, making a genuine improvement look like a further regression. Fixing both exposed a real third issue — over-correction into excessive hedging — resolved by rebalancing training-data ratios. Full trail: [`docs/proper_scale_results.md`](docs/proper_scale_results.md), [`docs/model_quirks.md`](docs/model_quirks.md) #16-17.
- **Found and fixed a rubric-construction bug that inflated a zero-context baseline to ~94% accuracy** — an operator-precedence bug in a string-concatenation expression silently dropped the gold answer from the judge's rubric whenever a probe had no listed alternatives. Caught because a model with no memory access scoring 94% on personalized-recall is *definitionally* impossible. [`docs/model_quirks.md`](docs/model_quirks.md).
- **Distillation's training-time metrics looked unhealthy (flat loss, unstable grad norm, ~95-98% completion clipping) but real evaluation showed a clean positive result** — a case where trusting the eval harness over the training curve mattered. [`docs/distillation_results.md`](docs/distillation_results.md).
- **27 real environment/API/tooling bugs found, fixed, and documented** across the stack, each with a root cause and fix, not just "it broke": [`docs/model_quirks.md`](docs/model_quirks.md).

## Reproducibility

- All training seeds are pinned in their respective config files (e.g. `seed: 1337` in `configs/training/sft_v1.yaml`).
- Base model revision is pinned by commit SHA in the SFT configs (`base_model_revision`). The DPO and distillation configs still reference `"main"` and should be pinned to a SHA before any re-run is treated as reproducible.
- Dataset directories carry a `DATASHEET.md` documenting generation methodology and known caveats, and a `hash.txt`. **All 11 corpora verify** as of 2026-09-13 — the four benchmark hashes did not match under any algorithm tested before this audit (see [`reports/ERRATA.md`](reports/ERRATA.md) E29), and two H22/H24 probe sets had no hash at all, so every hash was recomputed. `uv run python scripts/recompute_hashes.py` re-checks them, and a test enforces it in CI.
- `uv.lock` pins every dependency version.
- Hypotheses and eval design are committed to git *before* results — git history itself is the pre-registration record.

## Citation

```bibtex
@software{small_mind_companion,
  title  = {small-mind-companion: Post-training and cognitive architecture for a small multimodal companion LLM},
  author = {Ty, Arjhine A.},
  year   = {2026},
  url    = {https://github.com/arjhinety/small-mind-companion}
}
```

## License

Code: Apache-2.0 (`LICENSE`). Benchmarks/data: CC-BY-4.0 (`LICENSE-DATA`).
