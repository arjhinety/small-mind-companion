# pmb_v0 — PMB v0 fixture-scale smoke test (NOT the evaluation benchmark)

## Overview

This is a **fixture-scale smoke test**, not the evaluation benchmark. Every `pra_lenient`/`uar`
number reported in `README.md` and `docs/` is measured against
`data/benchmarks/pmb_v0_full/` (8 personas, 688 probes), not this corpus.

It exists to exercise `scripts/build_pmb.py` and the eval harness deterministically without a live
teacher model or API spend. Two personas and 36 probes are enough to prove the pipeline wires up
correctly; they are nowhere near enough to measure anything.

## Generation

- **Teacher client**: `fixture`
- **Personas**: 2
- **Facts per persona**: 8
- **Total probes**: 36

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact,
distractor, continuity.

## Limitations

- This corpus was generated with a **fixture teacher** (deterministic templates, no live LLM).
  Conversations and probes are synthetic approximations, not naturally generated.
- A real teacher model (OpenAI-compatible endpoint) is required for production-quality data. Pass
  `--teacher openai --teacher-model <model>`.
- Distractor and continuity probes are lightweight approximations; a real teacher would produce
  richer variants.
- Per-category probe counts are derived proportionally from `--facts-per-persona` rather than
  targeting the absolute reference distribution from the paper.
- Fixture-scale (2 personas) — a wiring smoke test, **not** the benchmark. Do not report metrics
  from this corpus.
