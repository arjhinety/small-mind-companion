# pmb_v0_full — PMB v0 evaluation benchmark

## Overview

This is the **evaluation benchmark** used for every `pra_lenient`/`uar` number reported in
`README.md` and `docs/`. It is *not* a training set, and the `sft_personas_*` directories under
`data/benchmarks/` are *not* benchmarks — see the note at the end.

The PMB-v0 benchmark evaluates a system's ability to recall and reason over personalised
information across multi-session conversations.

## Generation

- **Teacher client**: `openai`
- **Personas**: 8 (p000–p007)
- **Facts per persona**: 40
- **Total probes**: 688 (86 per persona, covering all 8 categories × 8 personas = 64/64 cells)

Composition: 608 answerable / 80 unanswerable. Per category: factual 98, continuity 96,
distractor 96, outdated_fact 96, preference 91, episodic 69, temporal 62, unanswerable 80.

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact,
distractor, continuity.

Supporting memory stores for evaluation live in `data/stores/pmb_v0_full/` (8 personas).

## Limitations

- This corpus was generated with a **live teacher model** (OpenAI-compatible endpoint), not a
  deterministic fixture. It is **not yet human-reviewed** — conversations and probes should be
  spot-checked by a human before being treated as production-quality reference data.
- The `acceptable_alternatives` field is present on every probe but is currently **unpopulated
  (688/688 records)**. Exact-match (`pra_strict`) scoring is therefore ~0 by construction, which
  is why `pra_lenient` (judge-scored, threshold ≥3/5) is the reported metric. Populating this
  field, or dropping it and documenting `pra_lenient` as the sole metric, is an open task.
- Training data was generated from a **disjoint** persona set (`sft_personas_v1`, separate
  generation run, seed 31415). Persona *names* may coincide because both draw on a shared name
  pool; facts and conversations differ.
- Contamination status is tracked in `data/sft/v1/DATASHEET.md`, not restated here — records can
  drift out of sync. Run `scripts/check_contamination.py` to verify.

## Note on the sibling directories

`data/benchmarks/sft_personas_v0/` and `data/benchmarks/sft_personas_v1/` are **training-persona
probe sets** (4 and 40 personas), generated to build SFT data. They are not evaluation benchmarks
and must not be reported as benchmark results. They live under `data/benchmarks/` only because
`build_pmb.py` writes all corpora to a single output root.
