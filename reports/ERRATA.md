# Errata

Corrections to claims in this repository, its Hugging Face model cards, and its published
results that the committed artifacts did not support.

## How to read this file

Two classes of correction:

- **Adopted in the repository.** The file is not published or hash-pinned, so it was corrected
  in place. The audit ledger (`reports/audits/001-claim-audit/resolutions.json`) records where.
- **Errata only.** The artifact is already published (the Hugging Face model cards) or is a
  committed measurement record that must not be rewritten (`results/**/metrics.json`). Those are
  **not** edited here; this document supersedes them until they are re-published.

**If a number on a Hugging Face model card disagrees with this file, this file is correct.**

The full findings ledger is `reports/audits/001-claim-audit/findings.json`. The rules derived from
these mistakes are `docs/GUARDRAILS.md`.

## Provenance

Audited at commit `ae75ca9` (2026-09-04). Method and verification results are in
`reports/audits/001-claim-audit/README.md`. All seven evaluated runs' `metrics.json` files were
independently recomputed from their own saved `raw.jsonl` responses and matched exactly — the
evaluation numbers in this repository are sound. Everything below is about the claims made
*around* those numbers.

## Published artifacts — re-pushed

All corrections below are now live on the Hugging Face Hub. The cards were published before this
audit and carried the superseded figures until 2026-09-13.

| Card | Corrected claim | Superseded text that was live |
|---|---|---|
| `onebee-gf-sft-v1` | SFT v1 + memory is **15.30% / 70.0%**; DPO is evaluated pairwise only | Table showed DPO as 70.0% UAR |
| `onebee-gf-dpo-v1-scale` | Preference alignment **45.7% vs 21.0% (24.7pp gap)**; no UAR measurement exists | Capability line and eval table claimed **70.0% UAR** |
| `onebee-gf-distill-v1` | Pre-distillation row is **SFT + memory**, not `dpo-v1-scale`; the +3.3pp spans two stages | Row labelled `dpo-v1-scale (pre-distillation)` |
| `onebee-gf-dpo-v1-scale-gguf`, `onebee-gf-distill-v1-gguf` | Q3_K_S is the smallest verified-coherent level | One card called Q3_K_M the recommended smallest |
| all eight cards | Owner references normalised to `arjhinety` | 99 links to `arrochi112`, 44 to `arghance231`-era URLs |
| both GGUF cards | "F16 reference plus 12 quant levels" | "12 levels, F16 through Q2_K" (ambiguous count) |

Re-push commits: `onebee-gf-dpo-v1-scale` `e152aa7`, `onebee-gf-sft-v1` `4025a09`,
`onebee-gf-distill-v1` `da4a14f`, `onebee-gf-dpo-v1-scale-gguf` `ba48df0`,
`onebee-gf-distill-v1-gguf` `368f83a`, `onebee-gf-sft-v0` `9c4f607`, `onebee-gf-dpo-v0` `663123e`,
`onebee-gf-dpo-v1-4epoch` `36e21cd`. Verified by fetching each card back from
`huggingface.co/arjhinety/<repo>/raw/main/README.md` and asserting the superseded strings are gone
and the corrected ones present.

The cards still restate quantization figures that no committed artifact backs (E6); each GGUF card
now says so explicitly rather than presenting them as independently checkable.

## Corrections by finding

### E1 — UAR 70.0% and pra_lenient 15.30% were attributed to the DPO checkpoint (HIGH)

**As written:** `docs/distillation_results.md` and four model cards described the system achieving
70.0% UAR / 15.30% `pra_lenient` as "SFT+DPO+memory" or as `dpo-v1-scale`.

**Correction:** those figures belong to **SFT-v1 + memory**. The tracked artifact
`results/v1_scale/E_sft_memory/metrics.json` was produced by `run_system_e_v1.py`, which loads
`HFEngine("outputs/sft/v1/merged")` — the SFT-only checkpoint — and names the system
`E_sft_memory_v1`. `docs/proper_scale_results.md` defines E as "SFT v1 + memory".

**No full-PMB measurement of the DPO checkpoint exists anywhere in `results/`.** `dpo-v1-scale`
appears only in the pairwise C-vs-E comparison. Do not cite any `pra_lenient` or UAR figure as a
DPO result.

### E2 — The distillation gain was measured across two training stages (HIGH)

**As written:** on-policy distillation improved `pra_lenient` by 3.3pp (15.30% → 18.59%).

**Correction:** the two compared systems are `outputs/sft/v1/merged` and
`outputs/distill/v1/merged`, and the latter was trained *from the DPO output*. The +3.3pp
therefore spans the DPO stage as well as distillation.

The distillation-specific measurement is the pairwise **C-vs-F** run (`C` = DPO+memory,
`F` = distill+memory), which holds the DPO stage constant on both sides: **F wins 38.1% vs C's
30.5%, a +7.6pp gap**. Cite that for H23.

### E3 — PMB has 8 personas, not 40 (HIGH)

**As written:** "688 adversarial probes across 40 personas".

**Correction:** 8 personas (`p000`–`p007`), 86 probes each. 40 is the count of SFT *training*
personas in `data/benchmarks/sft_personas_v1/`. The same README correctly said "8 categories"
two sections earlier.

### E4 — Revision pinning was claimed for configs that use `"main"` (HIGH)

**As written:** "Base model revision is pinned by commit SHA, not a moving tag
(`base_model_revision` in each config)."

**Correction:** only `sft.yaml` and `sft_v1.yaml` pin a SHA. `dpo.yaml`, `dpo_v1_scale.yaml`,
`dpo_v1_more_epochs.yaml` and `distill_v1.yaml` all use `base_model_revision: "main"`, and
`distill_v1.yaml` also uses `teacher_model_revision: "main"`. Runs launched from those configs are
**not** revision-pinned and should not be described as reproducible until the SHAs are recorded.

### E5 — Two mutually inconsistent GGUF size sets are published (MED)

**As written:** `README.md` and the GGUF cards' files tables state F16 8.64 GiB / Q8_0 4.61 GiB /
Q4_K_M 3.18 GiB.

**Correction:** `hf_readmes/gguf_README.md` contains *both* this set and 8.62 / 4.59 / 3.17 GiB in
its own evaluation table; `docs/quantization_results.md` states the F16 file as "9.27GB"
(8.63 GiB) while its table says 8.62 GiB. **No `llama-bench` output or GGUF file is committed**
(`*.gguf` is gitignored, `results/` holds no quantization artifact), so neither set is
re-derivable from a tracked artifact. One set must be chosen and the other corrected once a
benchmark artifact is committed.

### E6 — All quantization numbers lack committed evidence (MED)

Every size, throughput and perplexity figure in `docs/quantization_results.md` and both GGUF cards
is restated from run logs that are not in this repository: `llama-bench` throughput
(585.07/492.33/633.00 pp512, 26.15/43.07/58.00 tg128), the imatrix run (5,328 chunks, n_ctx=512,
PPL 2.7258 ± 0.0046), "model reports as 4.63B params", and every per-file size. There are no
`llama-*` outputs in `git ls-files`, and `results/` contains no quantization directory.

These are reported measurements, not independently verifiable ones. `docs/quantization_results.md`
and `README.md` now say so explicitly. Re-run `llama-bench` if you need to rely on them.

### E7 — The reported quantization floor understated the verified result (MED)

**As written:** "Confirmed down to Q4_K_M by manual + automated checks".

**Correction:** `docs/quantization_results.md` verifies coherent, in-character output at Q3_K_S and
Q3_K_M, and names **Q3_K_S as the smallest verified-coherent level** for both checkpoints. The
checks described are manual `llama-cli` / `llama-mtmd-cli` runs; no automated quality check is
evidenced.

### E8 — imatrix quantization was described as undone after it was done (MED)

**As written:** `docs/research_questions.md` stated the GGUF quants "used no imatrix calibration
data — a real next step".

**Correction:** `docs/quantization_results.md` in the same repository documents the calibration
corpus, the `llama-imatrix` run, and 6 levels requantized with `--imatrix`. What genuinely remains
open is the **imatrix-vs-non-imatrix generation-quality comparison** — that was never completed,
and `results/imatrix_perplexity_comparison.md` does not exist.

### E9 — `CITATION.cff` was stale on four of its fields (MED)

**As written:** title "onebee-gf: Stretching a 1B Parameter LLM…", version 0.0.0, author
"Arrochi", url `github.com/arrogance231/onebee-gf`.

**Correction:** the project is `small-mind-companion`, version 0.1.0 (`pyproject.toml`), the model
is ~2B effective-parameter (4.63B as reported by `llama-bench`), the repository is
`github.com/arjhinety/small-mind-companion` (the cited URL does not exist), and the author string
matched neither the README's BibTeX entry nor the Hugging Face owner.

### E10 — The card template contradicted every card it produced (MED)

`hf_readmes/checkpoint_README_template.py` declared `license: gemma` and "Inherits Gemma's license
terms", while all seven cards it generates declare `license: apache-2.0` and "Apache-2.0, inherited
from the base model". `CITATION.cff` and the README also say Apache-2.0. The template was wrong.
Note that the underlying claim — that Apache-2.0 is inherited from `google/gemma-4-E2B-it` — is
not verifiable from this repository.

### E11 — Contamination status was asserted in both directions at once (MED)

The SFT datasheets said the corpora were **not** checked; `data/distill/v1/DATASHEET.md` said they
**were** checked clean, citing `data/sft/v1/DATASHEET.md` — a file that said the opposite.

**Correction:** the check was actually run:

```
python scripts/check_contamination.py \
  --train-glob "data/sft/v1/train.jsonl" \
  --eval-glob "data/benchmarks/pmb_v0_full/probes.jsonl"
→ No contamination found.   (exit 0)
```

Same result for `data/sft/v0/train.jsonl`. **The data is clean**; the "not yet checked" text was
stale. Both SFT datasheets and the distill datasheet now record the command and its result.

### E12 — The SFT datasheets cited seeds that do not exist (MED)

`data/sft/v0/DATASHEET.md` cited "seed 9999" and `data/sft/v1/DATASHEET.md` cited "seed 31415".
Neither number appears anywhere in the repository. The persona corpora were generated by
`scripts/build_pmb.py`, whose `--seed` default is 1337; `generate_sft_data.py` uses
`random.Random(4242)`. As documented, persona generation was not reproducible. Both datasheets now
state that the seed is not recorded.

### E13 — DPO v1_scale was not built from the final SFT v1 (MED)

`data/dpo/v1_scale/DATASHEET.md` said the pairs used "the same pipeline as `data/sft/v1`".
Measured against the current `data/sft/v1/train.jsonl`: **2263/2277 user turns match but 0/2277
system texts match.** Both datasets call `ContextBuilder.build(..., recent_turns[-6:])`, so
identical inputs would produce identical system text — the retrieval contexts diverged. Git
timestamps show the DPO set predates SFT v1's dedup fix and ratio rebalance. The cause was not
pinned down. "Same pipeline" holds as generator *code*, not as prompt construction.

### E14 — Stale counts in the README (MED)

- "451 passing" tests → **473** `def test_` functions in `tests/**`; `docs/research_questions.md`
  already said 473.
- "21 real environment/API/tooling bugs" → `docs/model_quirks.md` numbers **1 through 27**.

### E15 — The hardware claim conflated two machines (MED)

**As written:** "All training and quantization runs were done on a single rented workstation GPU
(NVIDIA RTX PRO 6000 Blackwell class, ~96GB VRAM)".

**Correction:** `docs/quantization_results.md` records that the quantization box had **no
system-wide CUDA toolkit**, that `llama.cpp` was built CPU-only, that conversion and quantization
are CPU-only tools regardless, and that the source checkpoint was downloaded because "the previous
GPU box was deleted". The distill-v1 GGUF build ran on **Modal**. The README now separates LoRA
training from GGUF conversion/quantization.

### E16 — The reproducibility chain contained commands that cannot run (MED)

`docs/reproduction.md` presented `make figures` as a step in the chain that "should be run from a
clean clone, in order, to earn the reproducibility claim". `scripts/make_figures.py` does not
exist, so the target always failed. The `paper` target referenced a non-existent `paper/build.sh`.
Both dead targets are removed and the reproduction doc now states that figure regeneration is
unavailable rather than presenting a command that fails.

### E17–E28 — Lower-severity corrections

| # | Correction |
|---|---|
| E17 | "12 levels (F16 → Q2_K)" was ambiguous — it is the F16 reference **plus** 12 quant levels (13 text files; 14 with the vision projector) |
| E18 | One GGUF card called **Q3_K_M** the "recommended smallest safe level" while its own table and `docs/quantization_results.md` name **Q3_K_S**. **Corrected and re-pushed** (`368f83a`) |
| E19 | `sft_personas_v0`, `sft_personas_v1` and `pmb_v0` datasheets were all titled "PMB v0 — Personalised Memory Benchmark". Only `pmb_v0_full` is the benchmark; the others are training-persona corpora and a fixture smoke test. Root cause was a hardcoded title in `scripts/build_pmb.py` |
| E20 | `pmb_v0_full/DATASHEET.md` said "This run (8 personas) may not be the full v0 benchmark (target: 8 personas)" — self-contradictory, and stale relative to the generator that was fixed 65 s earlier |
| E21 | `data/distill/v1/` has **no `hash.txt`**, contradicting "Every dataset directory has a hash.txt" |
| E22 | Probes were said to ship "acceptable alternatives"; the field is present on all 688 probes and **empty in 688/688**, which is why `pra_strict` is ~0 throughout and `pra_lenient` is the reported metric |
| E23 | `make lint` / `make typecheck` assume the `dev` extra that the documented bare `uv sync` does not install |
| E24 | A card template linked to `README.md#engineering-highlights`; that heading does not exist (it is now "Results & Analysis") |
| E25 | `dpo-v1-scale` was still called "current best checkpoint overall" in the card template, the card generator and the quantization doc header, contradicting `distill-v1`'s status in those same files |
| E26 | `results/imatrix_perplexity_comparison.md` was referenced as a possible outcome file; **it does not exist** |
| E27 | The imatrix calibration corpus was described as "~11MB / 15k lines"; it is 11,086,366 bytes and **66,383 lines** |
| E28 | `docs/proper_scale_results.md`'s headline table reports the superseded pre-fix run, while the README linked to it as "current authoritative results". The table now carries superseded values inline; the corrected figures are 25.0% and 70.0% |

### E29 — Four benchmark `hash.txt` files do not verify, and never did (HIGH)

**As written:** `README.md` stated "Every dataset directory has a `hash.txt` and `DATASHEET.md`",
and this document's own first pass claimed that "all six `hash.txt` files match" once CRLF was
normalised to LF.

**Correction:** the six split into two groups, and that first sentence generalised from one of them.

- **The four SFT/DPO hashes verify.** `data/sft/v0`, `data/sft/v1`, `data/dpo/v0`,
  `data/dpo/v1_scale` all match `sha256` over the concatenated, LF-normalised, sorted `*.jsonl` —
  the algorithm in `generate_sft_data.py` / `generate_dpo_data.py`. Normalising CRLF is required
  because this repo has no `.gitattributes` and the hashes were computed on an LF checkout.
  **The training data is intact.**
- **The four benchmark hashes do not verify under any algorithm tested.**
  `data/benchmarks/{pmb_v0_full,sft_personas_v1,pmb_v0,sft_personas_v0}` mismatch on **thirteen**
  variants: path encoded as `str()` vs `as_posix()`; data read as raw bytes, as text, or
  LF-normalised; `sha256` over path+data, data-only, or `probes.jsonl`-only; walk order vs name
  order; and with `DATASHEET.md` included or excluded.

This is not caused by the audit's edits. The mismatch is present on a clean `git worktree` of
`ae75ca9` — the commit that introduced the files, where `hash.txt`, `probes.jsonl` and
`DATASHEET.md` were committed together and the `DATASHEET` has not changed since. Whatever produced
those four values is not the algorithm in `scripts/build_pmb.py` today, or the corpus was written
by a different revision of the script than the one committed beside it. **The cause is not
determined.**

**Consequence:** for the four benchmark corpora, `hash.txt` is not a valid integrity pin and must
not be cited as one. [`reports/data/study-001-freeze.json`](../data/study-001-freeze.json) is the
authoritative pin instead — it covers all four corpora and was verified against a clean checkout at
the `study-001` tag. `README.md` now says this.

**Resolved (2026-09-13).** All four benchmark hashes were regenerated, and
`data/distill/v1/hash.txt` was created for the first time, using the algorithm now documented in
`docs/reproduction.md`. All 11 corpora verify:
`uv run scripts/recompute_hashes.py` reports `11/11 corpora hash-clean`. A regression test
(`tests/unit/test_dataset_hashes.py`) now recomputes every hash on each test run, which is the
check whose absence let this go unnoticed — guardrail G9c is the rule, that test is its
enforcement. The *cause* of the four original values remains undetermined and is not needed:
the corpora were edited during this audit anyway (datasheet corrections, findings #19–#22), so the
old values were stale regardless of what produced them.

### E30 — Three docs quoted a superseded SFT v1 size as current (MED)

**As written:** `docs/day4_sft_v1_results.md`, `docs/dpo_results.md` and
`docs/proper_scale_results.md` all described the SFT v1 dataset as **2242 examples (2017 train /
225 val)**.

**Correction:** 2242 was the **first** v1 generation. That corpus was regenerated twice the same
day — once to fix the dedup collapse (E11), once to rebalance the abstention ratios — and the
committed `data/sft/v1/` is **2480 examples (2232 train / 248 val)**. `README.md` and
`data/sft/v1/DATASHEET.md` already had the correct figure; the three result docs still quoted the
superseded one. This is the same failure mode as E14 (a hand-typed count drifting) recurring in a
different file after the count changed.

All four occurrences now state that 2242 was the original generation and that the committed corpus
is 2480. The historical run rows keep the 2242 label, marked as the original generation, because
that *is* what those runs trained on — silently overwriting it would misrepresent the experiment.

**Lesson, folded into [`docs/GUARDRAILS.md`](GUARDRAILS.md) G4:** when a dataset is regenerated,
grep for its old size. A count that changes in the data does not change itself in the prose.

### E31 — `pra_lenient` for System D was quoted off by 0.03pp (LOW)

**As written:** `README.md`, `docs/STUDIES.md`, `docs/proper_scale_results.md` and
`docs/day4_sft_results.md` all reported System D (raw model + hybrid retrieval memory, k=8) as
`pra_lenient` **15.10%**, and derived two deltas from it — "+14.9pp" for the retrieval effect and
"+2.7pp" for the D→E gap at v0 scale.

**Correction:** `results/v0.1/D_memory/metrics.json` holds `0.1513157894736842`, i.e. 92/608 =
**15.13%**. 15.10% is that value rounded to one decimal and then written with two. Corrected in all
seven places, and the deltas follow: the retrieval effect is **+15.0pp** (0.16% → 15.13%) and the
D→E gap is **+2.6pp**.

**Why it survived the original audit.** The first pass checked that the documents agreed *with each
other* and that the v0 rows were backed by committed artifacts — both true — rather than that each
quoted figure rounded from its artifact. A self-consistent set of documents can all be wrong
together. `scripts/validate.py` now recomputes each quoted percentage from the artifact and compares
it at the document's own precision, which is what caught this.

**Lesson, folded into [`docs/GUARDRAILS.md`](GUARDRAILS.md) G9b and the claims matrix:** a
cross-document agreement check is not an artifact check. Both are needed, and
`registry/claims.jsonl` carries them as separate entry types (`cross_doc` and `json_field`).

### E32 — Two committed probe sets had no hash and were outside every integrity check (MED)

**As written:** `README.md` stated that dataset directories carry a `hash.txt`, and every document
described the integrity-pinning story as covering the committed corpora.

**Correction:** `data/benchmarks/h22_judgment` (the H22 abliteration judgment-quality probes) and
`data/benchmarks/emotional_range` (H24, nine registers) appeared in **neither**
`scripts/recompute_hashes.py` nor `scripts/validate.py`. Neither had a `hash.txt`, neither was
covered by the Study 001 freeze, and neither would have been touched by any consistency check in
this repository. They are committed research corpora — the H22 and H24 probes that Study 002 is
meant to run — and they were the only two unpinned probe sets in the tree.

Found by `scripts/validate.py`'s hygiene check (`unpinned-probe-sets`), not by reading. The corpus
count is **11, not the nine** that every document claimed.

**Fix:** both added to the hash-recompute list and hashed for the first time, and added to
`validate.py`'s mirrored list — whose sync assertion is what forced both files to be edited at once
rather than one silently drifting from the other. All "nine corpora" references updated to 11, and
the two claims-matrix entries renamed `readme-eleven-corpora-*` with `== 11` expressions.
`11/11 corpora hash-clean`.

**Lesson, folded into G9c:** a corpus that no list mentions is not covered by a check that iterates
a list. The hygiene check exists precisely to find corpora that the *other* checks do not know
about, and it should be run against the tree, not against the registry.

### E33 — Three validator checks compared raw bytes and failed only in CI (MED)

**As written:** `scripts/validate.py` reported 143 checks and 0 failures, and the result registry it
verifies was described as current.

**Correction:** it passed on Windows and failed on GitHub Actions — **4 tests failed in CI while 510
passed locally.** Three checks compared platform-dependent bytes:

- `result_registry_hashes_match()` used `sha256_file()`, hashing un-normalised bytes.
- `build_result_registry.sha256_file()` did the same when generating the index.
- `imatrix_calibration_shape()` compared `path.stat().st_size == 11086366`.

This repository has no `.gitattributes` and sets `core.autocrlf=true`, so a Windows working tree is
CRLF while the CI checkout is LF. The imatrix corpus is 11,086,366 bytes on Windows and 11,004,368
on Linux — the check was measuring the checkout, not the corpus.

**Fix:** all three now compare LF-normalised bytes, matching what `scripts/freeze_study_001.py` and
`scripts/recompute_hashes.py` already did, and `results/registry.jsonl` was regenerated. Verified on
both a CRLF tree and a simulated LF checkout (CR stripped across `.jsonl`, `.json`, `.txt`, `.md`,
`.py`): 143 checks, 0 failed, on both.

**This is the fourth instance of one root cause in this repository** — E29's four benchmark hashes,
the freeze manifest's recorded byte counts, the imatrix size, and now the result registry. Each was
found separately, each by a different check, and each had the same fix. That is the argument for
normalising at the single point where files are hashed rather than at each call site.

**Why local verification did not catch it.** Every check passed on the machine that wrote it. A
check that has only ever run on one platform has not been tested against the thing it is
vulnerable to; the CI run was the first real test, and it is why the CI gate exists.

### E34 — System B's UAR was quoted from the wrong run in the distillation table (MED)

**As written:** `docs/distillation_results.md`'s results table listed
`B (v1, rebalanced) | SFT alone, no memory | 0.16% | 25.0%`.

**Correction:** `results/v1_scale/B_sft/metrics.json` holds `uar = 0.2125` = **21.25%** (17/80).
**25.0%** (20/80) belongs to the *fixed-dedup, pre-rebalance* run — which
`docs/proper_scale_results.md` labels correctly as "B (SFT-v1 alone, fixed data)". The distillation
table labelled the row "rebalanced" while quoting the pre-rebalance figure, so the one document
presenting B next to the distillation result attributed the wrong calibration number to B.

**This is the third occurrence of one failure mode.** E2 quoted a confounded comparison; E31
mis-rounded D's `pra_lenient`; this quotes the right *family* of runs but the wrong *member*. In
each case the number was plausible, internally consistent with its neighbours, and drawn from a
real measurement — just not the one the label named. The general form is that "which run" is part
of the claim, not a detail behind it.

**Found while regenerating the site's discrepancy table**, not by `scripts/validate.py`, which
does not track this metric. It is now listed as a live discrepancy that the page would surface if
it recurred.

### E35 — The README marked H6 and H7 "Confirmed"; H7 was never tested (MED)

**As written:** the README's Experiments table had one row,
`H6, H7 | DPO improves preference alignment over SFT alone | Confirmed — 24.7pp pairwise win-rate gap at proper scale`.

**Correction:** the two hypotheses are different claims and neither is confirmed.
- **H6** (preference optimization improves persona consistency more than SFT alone,
  `docs/research_questions.md:68`): the 24.7pp C-vs-E pairwise gap is real and reconciles with
  `results/v1_scale/C_vs_E_pairwise/summary.json`, but `docs/proper_scale_results.md:65` calls it
  "not a fully confirmed H6 verdict, but the strongest evidence yet" — one seed, one data scale,
  an ad-hoc pairwise judge rather than the PCS metric H6 was specified against.
- **H7** (preference optimization incurs an alignment tax on general capability,
  `docs/research_questions.md:69`): **not tested.** `docs/dpo_results.md:101` says so — "Untested,
  not refuted" — because no general-capability benchmark was run against any DPO checkpoint.

The README row now splits them: H6 "Consistent, not confirmed — 24.7pp … single seed", H7 "Not
tested". The row also renamed H6 as "preference alignment", which is the DPO training objective,
not the persona-consistency outcome H6 predicts.

**Why it survived the validator.** `scripts/validate.py` and `registry/claims.jsonl` checked the
*number* in that row (the 24.7pp gap, which is correct) and treated the row as verified. Neither
checks the *verdict* next to it. A correct figure under a wrong conclusion is the same shape as
E2: the measurement was real, and the claim built on it was stronger than the measurement.

**Found from outside the repository**, by a reader comparing the README against the study site
(which already said "H7 was never tested").

### E36 — H5 was judged on a different comparison and a different metric than it was pre-registered on (HIGH)

**As written:** `docs/day4_sft_results.md:32-33`: "H5 (memory-aware SFT beats generic SFT when
evaluated with memory): **confirmed.** D→E: +2.6pp `pra_lenient` (15.13%→17.76%)." The README row
`H1, H4, H5 | … | Confirmed at v0 scale` (`README.md:181`) carried the same verdict.

**Correction:** H5 is **not tested**. `docs/research_questions.md:67` pre-registered it as Exp B1 vs
B2 (memory-aware SFT against generic SFT, both evaluated with memory), predicting +8 to +15pp on
Memory Utilization Rate. The verdict changed both:
- **The comparison.** D is the raw base model with memory; E is memory-aware SFT with memory. No
  generic-SFT model was trained, so the B2 arm does not exist, and D→E cannot separate
  memory-aware SFT from SFT in general, which is what H5 asks.
- **The metric.** The verdict used `pra_lenient`, not MUR, and +2.6pp is below the pre-registered
  range even on that metric.

**What D→E does show** *(post-hoc reanalysis of the frozen responses, not a new result)*: 108 vs 92
of 608 answerable probes, 27 correct only under E and 11 only under D (exact McNemar p = 0.014;
paired bootstrap 95% CI +0.7 to +4.6pp resampling probes, −0.2 to +5.3pp resampling whole
personas; single seed). The probe-level test separates the two response sets; the persona-level
interval, over only 8 personas, does not exclude zero, so the gap may depend on which personas the
benchmark contains. With 8 personas any interval is coarse; error bars on small evaluation sets are
easily too narrow, as Bowyer et al. (arXiv 2503.01747) show for CLT-based intervals. The D→E gap is
graded by the model that wrote E's training answers, so it may partly reflect the judge favouring
its own style: `run_system_d.py:38` and `run_system_e.py:38` request `gpt-5.6-luna` as judge, and
`generate_sft_data.py:55` requested SFT v0's targets from the same model, a relationship Li et al.
(arXiv 2502.01534) call preference leakage and report biases judges toward the related student
model. The same caveat applies to every judge-scored training gain in this repository; see E38.
Neither `pra_strict` (near 0 by construction) nor UAR (scored on the 80 unanswerable probes, not
the 608 behind the gap) can confirm or refute it. Figures recomputed by
`scripts/paired_tests_v0_1.py` (`results/v0.1/paired_tests/paired_tests.json`).

**MUR could not have tested H5 as implemented.** `mur`
(`src/onebee/evaluation/metrics/memory_quality.py:7`) is called by the harness
(`src/onebee/evaluation/harness.py:66`) and is 0.0 in every saved run because the IDs it compares
never match: probes cite benchmark-authored IDs (`p000_s003_t008`), while the memory store assigns
a random UUID at insert (`src/onebee/memory/store.py:179-180`). Of the 39 gold IDs for persona
p000, 0 appear in `data/stores/pmb_v0_full/p000.db`. The implemented proxy also only asks whether a
gold memory was *retrieved*, which depends on the retriever, not on the answer; two SFT variants
sharing a retriever score the same by construction. Wiring it up would not test H5; a
response-level measure is needed.

Corrected alongside: `docs/STUDIES.md` Study 002 item 12 ("computed but not called": it is called;
the IDs never match) and `registry/benchmarks.yaml` (`mur` was described as "Misattribution rate":
it is Memory Utilisation Rate).

**Why it survived the validator.** No claim in `registry/claims.jsonl` covered the README's
H1/H4/H5 row. Same shape as E35: the verdict column was unchecked.

### E37 — H10's "peaks at k=8" rests on a 3-probe decline (MED)

**As written:** `docs/day3_memory_results.md:48-49`: "H10 … **confirmed closely.** Clear
inverted-U, peak exactly at k=8, decline at k=16." README: `H10 | … | Peaks at k=8`
(`README.md:182`).

**Correction:** H10 is **not confirmed**. *(The figures below are a post-hoc reanalysis of the
frozen responses, not a new result.)* The k-sweep ran on a 120-probe stratified subsample (105
answerable). From k=8 to k=16, **3 probes were lost and 0 gained** (18 → 15 correct; exact McNemar
p = 0.25), so the decline that makes the curve an inverted U is within noise. The paired bootstrap
95% interval for that change is −6.7 to 0.0pp. Its upper bound is 0.0 because every disagreeing
probe went the same way, so it is a by-product of the method, not evidence. Resampling whole
personas instead gives −5.6 to −0.8pp. That interval excludes zero only because the 3 lost probes
belong to 3 different personas and none was gained: a resample avoids all three with probability
(5/8)⁸ ≈ 2.3%, below the 2.5% tail, so it too is a by-product of the method, not evidence. The rise
before the peak is real: from k=4 to k=8, 10 probes were gained and 0 lost (8 → 18 correct;
p = 0.002; 95% interval +4.8 to +15.2pp resampling probes, +4.8 to +14.9pp resampling personas).
The data cannot be told apart from H10's own falsifier, "monotonic non-decreasing"
(`docs/research_questions.md:73`), and k=12, where degradation was predicted to begin, was never
run.

Every figure above is recomputed from the committed responses by `scripts/paired_tests_v0_1.py`
(`results/v0.1/paired_tests/paired_tests.json`) and checked by `registry/claims.jsonl`. The k-sweep
`raw.jsonl` files are validated against their `metrics.json` but are not pinned by the Study 001
freeze.

**Why it survived the validator.** The k-sweep metrics are pinned by the freeze; no claim checked
the verdict built on them.

### E38 — Every judge-scored training gain is graded by the model that wrote the training targets (MED)

**As written:** the README's results table (`README.md:35-43`) and its Experiments rows for H5, H6
and H23 (`README.md:181`, `:183`, `:186`) report gains of trained systems over untrained or
less-trained ones, measured by `pra_lenient` or by a pairwise judge. None says who scored them.

**Correction:** no figure changes, but each of these comparisons is graded by the model that wrote
the training data it rewards. Every evaluation and comparison script constructs
`OpenAIJudge(model="gpt-5.6-luna")`, for example `run_system_e_v1.py:38` and
`compare_c_vs_e_v1_scale.py:47`. The same model wrote the targets for every SFT and DPO corpus
(`generate_sft_data.py:55`, `generate_dpo_data.py:55`; the `sft/v0`, `sft/v1`, `dpo/v0` and
`dpo/v1_scale` datasheets). The distillation teacher is different (`google/gemma-4-E4B-it`,
`configs/training/distill_v1.yaml:9`), but the distilled model starts from the DPO checkpoint. Li et
al. (arXiv 2502.01534) call this relationship between generator and judge preference leakage and
report that it biases judges toward the related student model. A gain measured this way may partly
reflect the judge favouring its own style.

**Affected:** the `pra_lenient` of every trained row against an untrained one in the results table;
D→E (E36); the 24.7pp C-vs-E pairwise gap, where C additionally trained on gpt-5.6-luna-written
chosen responses (H6); and the H23 uplift and its persona-consistency comparison.

**UAR is free of judge bias, but its grader was extended after seeing outputs.** UAR is scored by a
rule-based phrase matcher (`src/onebee/evaluation/metrics/personalized.py:76`,
`src/onebee/evaluation/graders/rule.py:86-88`), not by `gpt-5.6-luna`. Its phrase list was extended
twice on 2026-08-14 to include phrasing the SFT data teaches:
- **After outputs were seen.** Five literal training-template phrases (`rule.py:32-36`, commit
  `2615598`) were added after the dedup-fixed v1 outputs were seen (`docs/model_quirks.md:195-199`).
- **With the new training phrasing.** Seven paraphrases (`rule.py:40-46`, commit `2156e20`) were
  added in the same commit that wrote the new training templates, before the rebalanced run the
  README reports.

Both facts hold. The extension corrected a real mismatch: a model that had learned the trained
abstention phrasing was scored as not abstaining, so the dedup-fixed run read 2.5% where its saved
responses give 77/80 = 96.25% under the extended list (that run is superseded; its metrics were
never committed). And the measurement was changed after seeing the outputs it scores, which is the
adaptive reuse of data that standard inference does not assume (Dwork et al., arXiv 1411.2664).

**Which README numbers rest on which list.** The v0 rows (`README.md:36-39`) were scored with the
original 17 phrases (`_ABSTENTION_PHRASES[:17]`), the v1 SFT+memory and distilled rows (`:40`,
`:42`) with the full list. Re-scoring every committed run under both lists
(`scripts/uar_phrase_lists.py`) changes only those two: 70.0% becomes 55.0% and 71.25% becomes
58.75% under the original list. The H23 row ("UAR flat") compares two runs scored with the same
full list. The data-scale row compares across list versions and is corrected in E39. The results
table places the v0 SFT+memory row (33.75%, original list) next to the v1 row (70.0%, full list)
without saying the scoring changed.

Because the added phrases are the SFT data's own wording, gaps between raw and trained models' UAR
may partly measure wording rather than abstention behaviour; whether raw models abstain in
phrasings no version of the list covers is untested. Comparisons between untrained systems that
differ only in retrieval (A vs D, the k-sweep) are outside both the judge-bias and phrase-list
concerns.

**Size unknown.** The judge sees a gold answer and scores correctness on 1–5, which may limit a
style preference; this has not been tested. The responses are saved, so re-scoring them with an
unrelated judge would measure it (Study 002 item 15, `docs/STUDIES.md`).

**Why it survived the validator.** The validator checks numbers against artifacts. Which model
produced an artifact, which grader version scored it, and how either relates to the training data
were recorded only in prose.

### E39 — The "10x data scale improves calibration" row attributes to scale a change made of several (MED)

**As written:** `README.md:185`: `H16/H17-adjacent | 10x data scale improves calibration (UAR) |
Confirmed after fixing 2 real bugs that initially masked the improvement`.

**Correction:** the row compares System E trained on SFT v0 (`results/v0.1/E_sft_memory`) with
System E trained on the final SFT v1 corpus (`results/v1_scale/E_sft_memory`, `4fcf740`):
abstention on unanswerable probes 33.75% → 70.0% and answerable probes flagged as abstaining
151 → 195 of 608, under the full phrase list. Between those runs, everything below changed.
*Training data:*
1. Size: 225 → 2480 examples (`data/sft/v0/DATASHEET.md:9`, `data/sft/v1/DATASHEET.md:9`).
2. Personas: 4 → 40, each with its own memory store (`DATASHEET.md:5-6`).
3. Dedup fix (`bd7e556`, `docs/model_quirks.md` #16): abstention and irrelevant-retrieval
   examples stopped collapsing to one each.
4. Intended ratios: 15%/10% → 6%/5% (`2156e20`). Because of change 3, the actual shares went *up*:
   1 + 1 of 225 (0.4% each) → 136 irrelevant-retrieval (5.5%) and 105 abstention (4.2%) of 2480.
5. Templates: one fixed target per kind → four paraphrases per kind (`2156e20`).

*Training config:*

6. Effective batch 8 → 32: `per_device_train_batch_size` 4 → 8 and `gradient_accumulation_steps`
   2 → 4 (`configs/training/sft.yaml:19-20`, `configs/training/sft_v1.yaml:19-20`); with 2 epochs,
   52 → about 140 optimizer steps (v1 computed, not recorded).

*Evaluation:*

7. UAR phrase list: 17 → 29 phrases (E38). Under one list the change is 33.75% → 70.0% with
   answerable flagged 151 → 195 of 608 (full), or 33.75% → 55.0% with 151 → 137 of 608 (original;
   that 151 → 137 difference is within noise: exact McNemar p = 0.343, paired bootstrap 95% CI
   −6.7 to +2.1pp, one training seed per run).

Unchanged: base model and revision, LoRA settings, learning rate, schedule, warmup, epochs, max
length and seed (`sft.yaml:2-15,21-24`, `sft_v1.yaml:5-18,21-24`); the teacher; retrieval k=8; the
judge; the 688 probes; the evaluation script apart from the checkpoint path.

**The trajectory, one phrase list at a time.** *(Post-hoc reanalysis of saved responses, not a new
result.)* System E was trained and evaluated four times between v0 and the reported v1 run. Each
step is one training run with one seed; the corpus behind each v1 run is inferred from commit order
(`scripts/uar_trajectory.py`, `results/audit/uar_trajectory/uar_trajectory.json`).

| Step | SFT corpus (abstention + irrelevant-retrieval of total) | Changes from the previous step | Full list: UAR / answerable flagged | Original 17: UAR / answerable flagged |
|---|---|---|---|---|
| v0 | `data/sft/v0`: 1 + 1 of 225 | — | 33.75% / 151 of 608 | 33.75% / 151 of 608 |
| `e9e5ca8` | `data/sft/v1` @ `2b4c308`: 1 + 1 of 2242 | 1, 2, 6: 10x size, 40 personas, effective batch 32 | 16.25% / 60 | 16.25% / 60 |
| `5a9fe1b` | @ `bc2f409`: 129 + 341 of 2698 | 3: the dedup fix (the corpus grew because examples stopped collapsing) | 96.25% / 421 | 2.50% / 32 |
| `4fcf740` | @ `799390e`: 105 + 136 of 2480 | 4, 5: lower ratios, four templates per kind | 70.0% / 195 | 55.0% / 137 |

Under the full list, abstention on unanswerable probes runs 33.75% → 16.25% → 96.25% → 70.0%, and
on answerable probes 151 → 60 → 421 → 195 of 608 (24.8% → 9.9% → 69.2% → 32.1%). Under the original
list the dedup-fix step reverses direction on both, because that model answers in template wording
the original list does not contain (E38): 33.75% → 16.25% → 2.50% → 55.0%, and
151 → 60 → 32 → 137. Neither measure alone describes the change: the reported run abstains more on
unanswerable probes than v0 under both lists; on answerable probes, under the full list,
151 → 195 (exact McNemar p = 0.002; 95% CI +2.6 to +11.8pp resampling probes, but −1.3 to +15.8pp
resampling whole personas; see E36 on intervals over 8 personas), and under the original list
151 → 137, a difference within noise (exact McNemar p = 0.343; 95% CI −6.7 to +2.1pp resampling
probes, −10.7 to +6.1pp resampling personas; one training seed per run). On the 80 unanswerable
probes, every adjacent step differs by more than chance at the probe level under both lists (exact
McNemar p ≤ 0.013; the probe-level and persona-level intervals both exclude zero). No step changes
one thing: the first carries changes 1, 2 and 6 together, and the dedup fix also changed the corpus
size.

**Each step is one training run with one seed.** The McNemar tests and bootstrap intervals resample
probes, not training runs. They show that two saved sets of responses are unlikely to abstain
equally often on these probes. They cannot measure how much UAR moves between training runs of the
same configuration, which is the variation a claim about a configuration needs. Comparing training
configurations needs several seeds per configuration (Colas et al., arXiv 1806.08295, relate the
number of seeds to the probability of statistical errors in such comparisons). So the table shows
what four single runs did. It does not show that any one change produces its step, and the step
sizes carry no seed-level uncertainty. `docs/proper_scale_results.md:26-27` calls 16.25% "wrong"
with a "corrected value" of 70.0%: it is a correct measurement of a different model, two steps
earlier in this table, superseded rather than corrected.

Counts are recomputed by `scripts/sft_abstention_counts.py`, UAR by `scripts/uar_phrase_lists.py`
and the trajectory by `scripts/uar_trajectory.py`.

**Why it survived the validator.** The row's numbers were checked; the attribution in its verdict
was not.

### E40 — ORPO was recorded as removed from the pinned trl; it had moved to `trl.experimental` (MED)

**As written:** `README.md:190`: `— | ORPO as an alternative to DPO | Blocked — not supported by the
pinned trl version`. `docs/model_quirks.md` #15: "`trl.ORPOTrainer`/`ORPOConfig` do not exist in this
environment's trl (1.10.0) at all … ORPO support was dropped entirely". `docs/research_questions.md:116-121`
moved H3.1 to Week 3 on that basis.

**Correction:** trl 1.10.0 ships ORPO as `trl.experimental.orpo`: `trl/experimental/orpo/__init__.py`
exports `ORPOConfig` and `ORPOTrainer` (checked in `trl-1.10.0-py3-none-any.whl` from PyPI, sha256
`f4bdad4a10dffa558399177938267b0aee51a5052768027cbe807a97d1916576`). Only the top-level
`from trl import ORPOConfig` fails. trl moved ORPO into its experimental namespace and later removed the
deprecated top-level alias (release notes for v0.29.0, PR #5050); the experimental namespace carries no
stability guarantee. H3.1 was shelved on a misread `ImportError`, not on a missing feature. Whether the
experimental trainer works with this repository's training code is untested. H3.1 returns to Study 002
(`docs/STUDIES.md` item 16), and the `trl-orpo-removed` entry was dropped from
`registry/upstream.jsonl`, since there was nothing to report upstream.

**Why it survived the validator.** The README row's verdict was unchecked. The quirks entry recorded a
true symptom (the `ImportError`) and a false conclusion without inspecting the installed package.

## What was verified correct
So this file is not read as blanket scepticism:

- **All seven evaluated runs' metrics** recomputed from their own saved `raw.jsonl` responses match
  `metrics.json` exactly (`uar`, `pra_lenient`, `pra_strict`).
- **The four SFT/DPO `hash.txt` files match**, under `sha256` over concatenated LF-normalised
  `*.jsonl`. **The four benchmark `hash.txt` files do not match under any of thirteen tested
  algorithms — see E29.** The original audit pass reported "all six match" from a partial check;
  that sentence was wrong and is corrected here. The hashing algorithms in use are undocumented in
  the files and are now recorded in `docs/reproduction.md`.
- **The contamination check passes**: `No contamination found.` for both SFT corpora.
- **Every dataset split and count** is exact: 202/23, 2232/248, 2049/228, 2008/224, 688 probes,
  3437 training-persona probes; `By kind` balances re-derived by template matching; 40 facts per
  persona everywhere; the DPO rejected pool is exactly 5 distinct sentences; `distill/v1` is a
  genuine strict subset of `sft/v1/train.jsonl`.
- **PMB's design is sound**: 86 probes per persona × 8 personas × 8 categories = a fully populated
  64/64 cell grid, 608 answerable / 80 unanswerable, and all 80 unanswerable correctly carry no
  gold answer.
- The v0 result rows (0.16%/13.75%, **15.13%**/8.75%, 17.76%/33.75%), the 24.7pp C-vs-E gap, the
  38.1%/30.5% distillation win rates, the 0.524/0.509 stylometric figures, and the Q2_K-breakage
  narrative (which matches git history) all reconcile with committed artifacts.
  *(System D's `pra_lenient` was quoted as 15.10% until E31 — the artifact rounds to 15.13%. The
  first audit pass missed it by checking that the documents agreed with each other rather than that
  they matched the artifact; `scripts/validate.py` recomputes it and caught the difference.)*

## What remains open

- No full-PMB `pra_lenient`/UAR measurement exists for any DPO checkpoint.
- No committed evidence for any quantization number.
- `data/distill/v1/hash.txt` was never generated, and the four benchmark `hash.txt` files did not
  verify (E29) — **both resolved 2026-09-13**: all 11 corpora now verify.
- The `acceptable_alternatives` field is unpopulated in all 688 probes.
- The DPO v1_scale prompt-context divergence from SFT v1 was never root-caused.
- No human evaluation exists anywhere; no reviewer log or teacher transcript was retained, so
  "not human-reviewed" and the `gpt-5.6-luna` teacher identity are self-reported and unverifiable
  from the repository.
- Single seed throughout. Every headline number in this project is one run.
- H5 has no test: no generic-SFT model was trained, and MUR cannot measure it as implemented (E36).
- Every judge-scored training gain is scored by the model that wrote the training targets (E38); re-scoring the saved responses with an unrelated judge is Study 002 item 15.
