# How this repository is audited

Study 001 shipped 31 claims the committed artifacts did not support
([`reports/ERRATA.md`](ERRATA.md)). Those corrections are history. This document is the mechanism
that keeps the *next* set from going unnoticed: what is frozen, what is machine-checked, what runs
in CI, and what a contributor has to do.

The rules derived from the 31 findings are [`docs/GUARDRAILS.md`](GUARDRAILS.md). Read that if you
are writing up a result; read this if you are changing one.

## The four layers

### 1. The freeze — `reports/data/study-001-freeze.json`

Study 001 is frozen. Every artifact its findings rest on — benchmark corpus, training data, configs,
`metrics.json`, `raw.jsonl`, evidence-generation scripts and the public write-ups — is pinned by
SHA-256 over LF-normalised bytes, in seven groups.

```bash
uv run python scripts/freeze_study_001.py --check   # or: make freeze-check
```

`--check` fails if any pinned byte changed, if the recorded file count disagrees with the recorded
evidence, or if the manifest's `tree_facts` (how many corpora carry a `hash.txt`, which training
configs pin a revision) no longer describe the tree. Because `README.md` is a pinned artifact, a
correction to it is a re-freeze with a `REFREEZE_LOG` entry — the log is part of the artifact, and an
unrecorded re-freeze makes the manifest meaningless.

The freeze deliberately excludes *living documents* (this file, `docs/STUDIES.md`,
`docs/GUARDRAILS.md`, `reports/ERRATA.md`, the audit ledger). They are expected to keep changing as
mistakes are found; hashing them into the enforced set would mean every correction "broke the
freeze".

What the freeze does **not** do: tell you whether an artifact is *behind* a claim. That is layer 2.

### 2. The claims matrix — `registry/claims.jsonl`

One JSON object per public claim, with a machine-evaluable check. Coverage is the point: the six-row
eval ladder, the pairwise gaps, dataset splits, probe and persona counts, the test count, the bug
count, the hash claims, the config table, and the register of things this repository deliberately
does *not* claim.

Each entry is `{id, claim, where, artifact, check, status}`; the full schema, every check type, and
the three statuses are documented in [`registry/README.md`](../registry/README.md).

Three things about it are load-bearing:

- **A claim with no artifact gets `status: manual` and a reason**, and is printed on every run. It
  is never silently passed. There are 14 of them, all of the "the evidence does not exist in this
  repository" kind: quantization figures with no committed `llama-bench` output, model-card state
  that needs the network, training-time step counts that need the GPU stack, and intermediate runs
  whose raw responses were deliberately not committed.
- **`acknowledged_drift` is a watchlist, not a suppression.** It is used only where the quoting
  document is a frozen artifact that the audit rules forbid editing. The entry must carry a
  resolution naming the superseding record, and it is printed on every run. If it starts holding, the
  run prints it as resolved so the entry can be promoted.
- **A check that is wrong gets fixed; the expected value does not get widened.** A tolerance that was
  loosened to make a check pass is how a matrix stops meaning anything.

### 3. The validator — `scripts/validate.py`

```bash
uv run python scripts/validate.py          # exits non-zero on any failed claim
uv run python scripts/validate.py --strict # makes determinism-hygiene findings fatal too
uv run python scripts/validate.py --json   # machine-readable result
```

It is CPU-only, offline, and downloads nothing. Its sections:

| Section | What it does |
|---|---|
| `freeze` | re-derives the freeze check in-process (`scripts/freeze_study_001.py --check` logic) |
| `hashes` | recomputes every dataset `hash.txt` under its declared algorithm (`scripts/recompute_hashes.py` logic) |
| `metrics` | recomputes `uar`, `pra_lenient` and `pra_strict` from every `results/**/raw.jsonl` and compares to that run's `metrics.json` |
| `claims` | evaluates every entry in `registry/claims.jsonl` |
| `registries` | every count in `registry/*.yaml` against the data files, plus `results/registry.jsonl` freshness and digests |
| `crossdoc` | the same metric must not be quoted with two values across `README.md` and `docs/STUDIES.md`, and every referenced guardrail id must exist |
| `hygiene` | report-only determinism hygiene: configs whose `base_model_revision` is not a 40-char SHA, corpora with no `hash.txt`, and frozen write-ups whose quoted percentage disagrees with its artifact |
| `pollution` | no training corpus shares a 13-gram with the PMB benchmark (`scripts/check_contamination.py` logic) |

Two deliberate limits, because a stricter check would be a false one:

- **The cross-document check is curated, not swept.** Superseded values are legitimately quoted
  inline in the frozen per-experiment write-ups ("superseded: 25.0%"), so comparing every number in
  `docs/` against an artifact would flag history as error. `crossdoc` compares the headline metrics
  across `README.md` and `docs/STUDIES.md`; the `hygiene` sweep covers the frozen result write-ups
  and *reports* rather than fails, so a historical run record is not treated as a broken claim.
- **"Not human-reviewed" and "single seed" are `manual`.** A repository can confirm that every
  datasheet *says* it was not reviewed; it cannot prove that no review happened off-repository. They
  are stated as absences, per guardrail G3.

### 4. The derived result index — `results/registry.jsonl`

A view over the evaluated runs, not a source of truth: one line per run (19 of them), with the run
path, system name, probe count, `pra_lenient`/`uar`/`pra_strict`, the sha256 of its `metrics.json`
and `raw.jsonl`, and which training stages the system carries. A DPO-lineage run with no
`metrics.json` gets `"metrics": null` and exposes `pairwise` instead — there is no full-PMB
measurement for any DPO checkpoint, and the index must not imply one.

```bash
uv run python scripts/build_result_registry.py           # regenerate
uv run python scripts/build_result_registry.py --check   # fail if stale
```

`validate.py` fails if it is stale, if any recorded digest disagrees with the file, if it misses a
run that has a `raw.jsonl`, or if it carries a metric for a run whose `metrics.json` does not exist.

## What runs in CI

`.github/workflows/ci.yml`, on every push and pull request to `main`:

```bash
uv sync --extra dev
uv run ruff check
uv run black --check .
uv run mypy src
uv run pytest
uv run python scripts/freeze_study_001.py --check
uv run python scripts/validate.py
```

The validator runs last because it is the broadest: it subsumes the freeze check, every hash, every
recomputed metric, and the claims matrix. The freeze step is kept as its own CI step so a broken
freeze is attributable at a glance (G8: a command named in a document must exist and do what the
document says).

The same set is `make lint`, `make typecheck`, `make test`, `make freeze-check`, `make validate`.
`make validate` and the CI step run the identical command.

## Adding a result

1. **Pre-register.** Write the hypothesis and the evaluation design into `docs/research_questions.md`
   before running it. Git history is the pre-registration record; a hypothesis added after the result
   is not one.
2. **Produce the artifact.** `metrics.json` and `raw.jsonl` under `results/<pass>/<system>/`. Do not
   hand-write a metric into a document.
3. **Index it.** `uv run python scripts/build_result_registry.py`.
4. **Attribute it correctly.** The label comes from the script that produced the result and the
   checkpoint it loaded, never from a table (G1). If the system name and the prose label disagree,
   the artifact filename wins. `SYSTEM_LABELS` in `scripts/build_result_registry.py` is where the
   attribution is recorded.
5. **Check for confounds.** For an "X improved Y" claim, list every stage that differs between the
   two systems (G2). `results/registry.jsonl`'s `training_stages_present` exists to make that
   visible.
6. **Add the claims.** One entry per public number in `registry/claims.jsonl`, then run the
   validator.
7. **Never add a number to a frozen document.** Study 001 is closed. A new result is a Study 002
   result, reported against the frozen checkpoints rather than replacing them.

## Changing or correcting a Study 001 claim

- A **correction** goes in `reports/ERRATA.md` against the frozen artifact, and the audit ledger
  (`reports/audits/001-claim-audit/`) gains the finding. The frozen artifact is not edited; the
  errata supersedes it. This is what happened 31 times.
- A **changed result** is a Study 002 result. Re-running Study 001 with better data does not update
  Study 001; it produces a new finding that Study 001's write-up cannot claim.
- A **living document** — this file, `docs/STUDIES.md`, `docs/GUARDRAILS.md`, `reports/ERRATA.md` —
  is corrected in place. That is why it is not in the freeze's enforced set.
- A correction to a **frozen public surface** requires a re-freeze with a `REFREEZE_LOG` entry
  saying what moved and why it does not change a Study 001 finding.

## Known open items

Recorded here so the validator's reports are not mistaken for surprises:

- **`docs/STUDIES.md` describes the curriculum as "not human-reviewed" and single-seed**; both are
  `manual` in the claims matrix. They are absences, not measurements.
- **Five training configs do not pin `base_model_revision`** (`dpo*.yaml`, `distill_v1.yaml`); the
  validator reports them every run (G6). Pinning them is a Study 002 item.
- **`data/benchmarks/emotional_range/` and `data/benchmarks/h22_judgment/` have no `hash.txt`.** They
  are unrun probe sets referenced by no headline number.
- **The training-persona corpora are not disjoint from PMB at the text level.** 138 of PMB's 266
  `(predicate, object, category)` fact triples also occur in `data/benchmarks/sft_personas_v1/`, and
  3041 training-persona probe rows carry a question that also appears in PMB. The generation runs and
  persona identities are distinct, and the 13-gram contamination check against PMB is clean — which
  is the leakage test that matters — but "generated disjoint" in `README.md` is stronger than the
  evidence supports, so that claim is `manual` and the measured overlap is recorded in
  `registry/benchmarks.yaml`. See [below](#acknowledged-drift).
- **`reports/ERRATA.md` E5 is still PARTIAL**: two mutually inconsistent GGUF size sets are published
  and neither is re-derivable without a committed `llama-bench` artifact.

### Acknowledged drift

Claims whose quoting document is a Study 001 frozen artifact, recorded rather than corrected:

- **`data/distill/v1/DATASHEET.md`** states that 0 of the 224 distill val prompts appear in
  `data/sft/v1/val.jsonl`. The correct count is **2**. The substantive claim — that the whole prompt
  set was carved from `train.jsonl` — is correct and machine-checked (`distill(train+val)` equals the
  SFT v1 train prompt set); the stray 2 is a consequence of duplicate prompt text inside
  `data/sft/v1/` itself (4 SFT val rows duplicate SFT train rows), not of how distill was built. See
  `datasheet-distill-val-count-zero` in the claims matrix.

Everything else the validator reports is either a deliberate `manual` absence or a hygiene note.
