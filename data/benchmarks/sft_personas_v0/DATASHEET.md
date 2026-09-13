# sft_personas_v0 — training-persona probe set (NOT an evaluation benchmark)

## Overview

This directory is a **training-persona corpus**, not a benchmark. It was generated to build the
SFT v0 dataset (`data/sft/v0/`), and its personas are **disjoint by construction** from the PMB v0
evaluation benchmark (`data/benchmarks/pmb_v0_full/`, separate generation run, separate output
directory).

Do **not** report metrics computed on this corpus as benchmark results, and do not evaluate
trained checkpoints against it — that would measure performance on personas the model was trained
on. All reported `pra_lenient`/`uar` numbers come from `pmb_v0_full`.

It lives under `data/benchmarks/` only because `scripts/build_pmb.py` writes every generated
corpus to a single output root.

## Generation

- **Teacher client**: `openai`
- **Personas**: 4
- **Facts per persona**: 40
- **Total probes**: 344

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact,
distractor, continuity.

Matching memory stores: `data/stores/sft_personas_v0/` (4 personas).

## Limitations

- This corpus was generated with a **live teacher model** (OpenAI-compatible endpoint), not a
  deterministic fixture. It is **not yet human-reviewed**.
- This set (4 personas) is below the scale of the v1 training corpus (40 personas). It exists only
  to build the small v0 SFT dataset (225 examples) that the project later superseded.
- The `acceptable_alternatives` field is present but unpopulated throughout.
