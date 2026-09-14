# Security

## Credentials

**Do not commit `.env`, API keys, or provider tokens.** `.env` is gitignored; keep it that way.

Two things in this repository talk to a paid API and need credentials at run time:

- **`data/` generation uses a paid teacher API.** The corpora under `data/` were generated with a
  live OpenAI-compatible teacher model (see each `DATASHEET.md`). Regenerating them costs money and
  needs `OPENAI_API_KEY`. Generation scripts read it from the environment; never hard-code it, and
  never commit the generation logs.
- **Evaluation scoring uses an LLM judge.** `pra_lenient` and the pairwise comparisons are
  judge-scored, which needs the same key.

The CPU-only checks — `pytest`, `scripts/validate.py`, `scripts/freeze_study_001.py --check`,
`scripts/recompute_hashes.py`, `scripts/check_contamination.py` — need **no** credentials, download
nothing, and are safe to run anywhere. Keep it that way: a check that needs a key cannot run in CI,
and a claim that only the author can verify is not audited.

## Before you open a pull request

- No `.env` file, key, token, or generated credential in the diff.
- No committed model weights or bulk datasets. Checkpoints are published to the Hugging Face Hub, not
  committed here; `*.gguf`, `*.safetensors` and `outputs/` are gitignored.
- No private or personally identifiable data. The corpora are synthetic personas, generated for this
  project — do not add real conversation logs, real names, or real user data to `data/`.

## Reporting

Report a leaked credential or a sensitive-data problem through a private repository channel rather
than a public issue. If a key was ever committed, rotate it: removing the commit does not remove the
key from history or from anyone who has already cloned.

This repository holds no production system and no user data — it is a research artifact — so the
practical risks are credential leakage and the cost of an accidental large regeneration run, not
runtime compromise.
