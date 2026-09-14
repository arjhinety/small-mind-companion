# Contributing

This repository welcomes reproductions, alternative seeds, negative results, corrections, new
probes, quantization measurements, and studies that fail to reproduce something here. A failure to
reproduce an existing finding is a valuable contribution.

Before opening a pull request:

- **Pre-register.** A hypothesis or eval design goes into git *before* the result exists. Write it in
  [`docs/research_questions.md`](docs/research_questions.md) first; git history is the
  pre-registration record, and a hypothesis added afterwards is not one.
- **Never commit a number by hand.** Add the result as an artifact under `results/`, then index it
  with `uv run python scripts/build_result_registry.py`. Prose follows artifacts, never the reverse.
- **Attribute results from the script, not the write-up** (guardrail G1). Name the script and the
  checkpoint path that produced a number. If the artifact's system name and the prose label disagree,
  the artifact wins.
- **Isolate one stage per comparison** (G2). For an "X improved Y" claim, list every training stage
  that differs between the two systems; if more than one does, report it as a combined effect or run
  the isolating comparison.
- **State absence as absence** (G3). An em dash and one sentence for a metric that was never
  computed — never a number borrowed from the neighbouring row.
- **Add a claim for every public number**, in `registry/claims.jsonl`, and run the validator.
- **Never commit checkpoints, `.env` files, API keys, bulk datasets, or fabricated results.**

## Running the audit

```bash
uv sync --extra dev
make lint typecheck test          # ruff, black, mypy, pytest
make freeze-check                 # Study 001 is frozen; fails if a pinned byte changed
make validate                     # freeze + hashes + recomputed metrics + claims matrix
```

`make validate` is the same command CI runs. It exits non-zero whenever a claim does not match its
artifact. [`docs/AUDIT.md`](docs/AUDIT.md) explains what it checks and what has to change when it
fails.

## Adding a dataset

1. Put it under `data/`, with a `DATASHEET.md` recording how it was generated, what it contains, and
   what is known to be wrong with it.
2. Generate a `hash.txt` with the algorithm its generator uses and record the algorithm next to it —
   this repository has had two incompatible, undocumented hashing schemes in use, and four published
   hashes that matched nothing (errata E29).
3. Contamination-check it against the evaluation benchmark and record the command and its output:

   ```bash
   uv run python scripts/check_contamination.py \
     --train-glob data/<your-corpus>/train.jsonl \
     --eval-glob data/benchmarks/pmb_v0_full/probes.jsonl
   ```

4. Add an entry to `registry/datasets.yaml` with the record counts, splits, hash algorithm and a
   `contamination_status`, plus a `file:line` evidence anchor for each documentary claim.
5. Say whether it is a benchmark or training data in `registry/benchmarks.yaml`. The two
   training-persona corpora under `data/benchmarks/` are **not** benchmarks (errata E3) — the
   directory name is an artefact of `scripts/build_pmb.py` writing every corpus to one root.
6. Run `make validate`.

## Adding a run

1. Pre-register the hypothesis.
2. Train with a config under `configs/training/` that pins `base_model_revision` to a 40-character
   commit SHA. The validator reports configs that do not, and a run from an unpinned config is not
   reproducible however good the result.
3. Record the seed in the config, and the evaluation in `results/<pass>/<system>/` as `metrics.json`
   plus `raw.jsonl`. The raw per-probe responses are the evidence `pra_lenient`/`uar` are recomputed
   from; without them a metric is an assertion.
4. Index it: `uv run python scripts/build_result_registry.py`.
5. Add claims, then `make validate`.

## Adding or correcting a claim

1. Add the entry to `registry/claims.jsonl` using the narrowest check that actually tests the claim.
   The schema and every check type are documented in `registry/README.md`.
2. Run `make validate`. If it fails, fix the check or the document — **do not widen the tolerance or
   loosen the expected value to make it pass.** A check that was relaxed to pass is worse than no
   check, because it reads as verified.
3. If a check genuinely cannot be machine-evaluated, use `status: "manual"` and say *why* in the
   reason. "The evidence does not exist in this repository" is a reason; "it is hard to check" is not.

## The freeze rule

Study 001 is frozen. Its artifacts are pinned by SHA-256 in
[`reports/data/study-001-freeze.json`](reports/data/study-001-freeze.json), and `make freeze-check`
fails if any of them changed.

**A changed Study 001 artifact is never a silent edit.** It is exactly one of two things:

- **An errata entry** — a correction to a *claim*. It goes in
  [`reports/ERRATA.md`](reports/ERRATA.md) against the frozen artifact, with the finding recorded in
  `reports/audits/001-claim-audit/`. The frozen artifact is not rewritten; the errata supersedes it.
- **A Study 002 result** — a changed *result*. It is reported as a Study 002 finding, against the
  frozen Study 001 checkpoints rather than in place of them.

If you need to change a frozen public surface (`README.md`, a published write-up), that is a
re-freeze: run `uv run python scripts/freeze_study_001.py` and add a `REFREEZE_LOG` entry recording
what moved and why it does not change a Study 001 finding. The log is part of the artifact. An
unrecorded re-freeze makes the manifest meaningless, and re-freezing to make a test pass defeats the
point of the freeze.

Living documents — this file, `docs/AUDIT.md`, `docs/STUDIES.md`, `docs/GUARDRAILS.md`,
`reports/ERRATA.md`, the audit ledger — are corrected in place and are excluded from the freeze
check.

New work is Study 002. Start from the candidate list in [`docs/STUDIES.md`](docs/STUDIES.md).
