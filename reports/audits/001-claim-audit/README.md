# Claim audit 001

The record of every claim in this repository, its Hugging Face model cards, and its
documentation that the committed artifacts did not support — and how each was resolved.

| File | Contents |
|---|---|
| `findings.json` | The 33 findings of the audit of commit `ae75ca9` (2026-09-13). Fields: `id`, `severity`, `area`, `location`, `claim` (as written), `actual` (what the artifacts show), `category`, `pinned`, `public` |
| `resolutions.json` | One resolution per finding: `status`, `where` it was fixed, and a note where the fix needed one. Statuses are `RESOLVED`, `PARTIAL` or `OPEN` |

The rules derived from these findings are in [`docs/GUARDRAILS.md`](../../../docs/GUARDRAILS.md).
Corrections to artifacts that could not be edited in place are in
[`reports/ERRATA.md`](../../ERRATA.md).

## Severity

- **HIGH** — a reader draws a materially wrong conclusion about what was measured.
- **MED** — a claim is wrong, stale or unsupported, but does not invert a result.
- **LOW** — internal inconsistency, broken link, or a count that is merely out of date.

## Area

- **D** — top-level docs (`README.md`, `Makefile`, `CITATION.cff`, `docs/reproduction.md`)
- **T** — training and evaluation reports (`docs/proper_scale_results.md`, `docs/distillation_results.md`)
- **C** — capability and quantization (`docs/quantization_results.md`, GGUF cards)
- **P** — data and public surfaces (`data/**/DATASHEET.md`, Hugging Face model cards)

## Method

Five independent read-only passes over commit `ae75ca9`, each recomputing values from the
committed artifacts rather than reading prose:

1. **Metrics recomputation.** Every `results/**/metrics.json` was recomputed from its own
   `results/**/raw.jsonl` by re-deriving `uar` (unanswerable probes where `abstained` is true)
   and `pra_lenient` (answerable probes where `lenient_correct` is true) against the probe
   categories in `data/benchmarks/pmb_v0_full/probes.jsonl`. **All seven runs matched
   `metrics.json` exactly.** The evaluation numbers in this repository are sound; the defects
   found by this audit are in the prose and metadata around them.
2. **Benchmark and dataset verification.** Probe counts, persona counts, category
   distributions, per-persona balance, answerable/unanswerable split, dataset splits and
   `By kind` balances were counted directly from the JSONL files. All matched, except the
   claims listed in the ledger.
3. **Provenance check.** The script that produced each artifact was read to confirm *which
   checkpoint* a result describes, rather than trusting the label in the write-up. This is what
   surfaced findings 1 and 2 — the highest-severity items in this audit.
4. **Integrity check.** All six committed `hash.txt` files were recomputed. **Four verify and four
   do not.** The four SFT/DPO hashes match under `sha256` over concatenated, LF-normalised,
   sorted `*.jsonl`. The four benchmark hashes match under **none of thirteen tested algorithms**,
   on a clean checkout of the commit that introduced them (finding #29). The two hashing
   algorithms in use are undocumented in the files themselves and are now recorded in
   `docs/reproduction.md`. The first pass of this audit reported "all six match" from the four
   that passed — that sentence is itself corrected in `reports/ERRATA.md`.
5. **Contamination check.** `scripts/check_contamination.py` was actually run against both SFT
   corpora: `No contamination found.` (exit 0). The "not yet checked" datasheet text was stale,
   not the data.

Claims that could not be verified from the repository at all — the teacher model identity
(`gpt-5.6-luna`), the "not human-reviewed" status, all quantization benchmark numbers, and the
private imatrix repository — are recorded as `MISSING_EVIDENCE` rather than silently accepted.

## What this audit did not do

- It did not re-run any training, evaluation, or quantization. No GPU was used.
- It did not verify the private imatrix repository, the Hugging Face Hub state, or any external
  service.
- It did not evaluate whether the reported results are *scientifically* correct — only whether
  the repository's claims about them match the repository's own artifacts.
