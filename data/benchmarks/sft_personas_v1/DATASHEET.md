# sft_personas_v1 — training-persona probe set (NOT an evaluation benchmark)

## Overview

This directory is a **training-persona corpus**, not a benchmark. It was generated to build the
SFT v1 dataset (`data/sft/v1/`), and its personas are **disjoint by construction** from the PMB
v0 evaluation benchmark (`data/benchmarks/pmb_v0_full/`, separate generation run, seed 31415).

Do **not** report metrics computed on this corpus as benchmark results, and do not evaluate
trained checkpoints against it — that would measure performance on personas the model was
trained on. All reported `pra_lenient`/`uar` numbers come from `pmb_v0_full`.

It lives under `data/benchmarks/` only because `scripts/build_pmb.py` writes every generated
corpus to a single output root.

## Generation

- **Teacher client**: `openai`
- **Personas**: 40
- **Facts per persona**: 40
- **Total probes**: 3437

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact,
distractor, continuity.

Matching memory stores: `data/stores/sft_personas_v1/` (40 personas).

## Limitations

- This corpus was generated with a **live teacher model** (OpenAI-compatible endpoint), not a
  deterministic fixture. It is **not yet human-reviewed**.
- One earlier README revision described PMB as having "40 personas" by conflating this
  training-persona count with the evaluation benchmark's 8. The evaluation benchmark is
  `pmb_v0_full` with **8** personas; this set with 40 is never evaluated against.
- The `acceptable_alternatives` field is present but unpopulated throughout.
