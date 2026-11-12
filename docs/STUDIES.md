# Studies

This project is organised into studies. A study is **frozen** when its findings are final: its
evidence is hash-pinned, its write-ups stop changing, and its repository links resolve at a git
tag. Work done after a freeze belongs to the next study.

The rule exists because the alternative — a repository whose numbers quietly drift as the
interesting result changes — is exactly what the audit found 40 instances of. See
[`reports/ERRATA.md`](../reports/ERRATA.md) for what went wrong, and
[`docs/GUARDRAILS.md`](GUARDRAILS.md) for the rules derived from it.

---

## Study 001 — Post-training a small multimodal model for long-horizon personalised memory

**Status: FROZEN** · Frozen on **2026-09-13** · Evidence tag **`study-001`**
· Freeze manifest: [`reports/data/study-001-freeze.json`](../reports/data/study-001-freeze.json)
· Write-up: **[small-mind.arjhinety.com](https://small-mind.arjhinety.com)**

**Frozen artifact.** [`reports/data/study-001-freeze.json`](../reports/data/study-001-freeze.json)
pins 76 artifacts across 7 groups. The write-up above is generated from the same committed
artifacts; if the two ever disagree, the artifacts and this document win, and the site is the bug.

**Question.** How much of the long-horizon memory gap can a ~2B vision-capable model close without
scaling parameters — by pairing it with an external memory system and LoRA post-training — and how
does that interact with knowing when *not* to answer?

**Base model.** `google/gemma-4-E2B-it`, pinned at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
(4.63B parameters as reported by `llama-bench`; ~2B effective).

**What was run.**

| Stage | Intervention | Data | Result |
|---|---|---|---|
| A | raw model, no memory | — | pra_lenient 0.16% / UAR 13.75% |
| B (v0) | LoRA SFT, no memory | 225 ex | 0.16% / 16.25% |
| D | raw + hybrid retrieval memory (k=8) | — | 15.13% / **8.75%** |
| E (v0) | LoRA SFT + memory | 225 ex | 17.76% / 33.75% |
| E (v1) | LoRA SFT + memory, rebalanced | 2,480 ex | 15.30% / 70.0% |
| E-distill | + on-policy distillation | 2,008 prompts | **18.59% / 71.25%** |
| C-vs-E | DPO, pairwise only | 2,277 pairs | 45.7% vs 21.0% (+24.7pp) |

Metrics are on PMB (Personalized Memory Benchmark): 688 probes, 8 personas × 8 categories, 608
answerable / 80 unanswerable. `pra_lenient` = judge-scored recall on answerable probes;
UAR = unanswerable-abstention rate.

**Headline findings.**

1. **Retrieval augmentation alone made abstention worse.** Adding memory moved `pra_lenient` from
   0.16% to 15.13% (+15.0pp) while UAR fell from 13.75% to **8.75%** (−5pp). Better recall and more
   confident fabrication arrived together.
2. **Abstention had to be supervised explicitly, and the ratio had to be tuned.** Restoring the
   abstention training signal took UAR from 16.25% to **96.25%** — and drove false-abstention on
   answerable questions to **69.2%**. Rebalancing (ratios 15%→6% and 10%→5%, four paraphrases per
   template instead of one) landed at 70.0% UAR with false-abstention cut to 32.1%.
3. **The apparent 10×-scale regression was two bugs and one real tradeoff.** A dedup step was
   collapsing ~227 intended abstention examples to **1**, and the eval harness's own abstention
   detector did not recognise the model's corrected phrasing. Both are documented in
   `docs/model_quirks.md` #16–17.
4. **On-policy distillation improved `pra_lenient` by 3.3pp with UAR flat**, confirmed by a
   pairwise comparison that holds the DPO stage constant (+7.6pp) and by two independent
   persona-consistency measures.
5. **Q2_K quantization is broken for this model family**, reproduced on two independently trained
   checkpoints; Q3_K_S is the smallest verified-coherent level.

**What Study 001 does not claim.** No full-PMB measurement exists for any DPO checkpoint. No
quantization figure is backed by a committed artifact. `pra_strict` (whole-response exact match) is ~0
because responses are sentences, not because the `acceptable_alternatives` field is empty (E41), so
`pra_lenient` is reported instead. Everything is a single seed and a single run. There is no human evaluation.
**Freeze.** `scripts/freeze_study_001.py` pins 76 artifacts across 7 groups by SHA-256 over
LF-normalised bytes. `--check` fails if any of them changes:

```
uv run python scripts/freeze_study_001.py --check
```

Living documents (this file, the errata, the guardrails, the audit ledger) are recorded for
provenance but excluded from the check, because they are expected to keep changing as mistakes are
found.

---

## Study 002 — Not started

**Status: IN PROGRESS · no results yet**

Everything after the Study 001 freeze. Each experiment is pre-registered before it runs, and
reported against the frozen Study 001 checkpoints rather than in place of them.

Study 002 also inherits the audit items Study 001 could not close. `reports/ERRATA.md` records what
was corrected; these are the ones that need new work rather than a correction, and each names the
finding it came from.

### Needs a GPU

1. **The crossover experiment (H16/H17).** `run_crossover_baseline.py` is written and unrun: 2B +
   scaffold vs the 8B `gemma-4-E4B-it` with no memory. This is Study 001's own thesis question and
   the one experiment that answers it.
2. **Multi-seed replication.** Every Study 001 number is a single seed and a single run. The
   retrieval finding in particular — recall up 0.16% → 15.13%, abstention *down* 13.75% → 8.75% —
   is the kind of claim that needs ≥3 seeds and a paired bootstrap before it should be relied on.
   The repo already has `bootstrap_ci`, `paired_bootstrap_diff`, `holm_bonferroni` and
   `minimum_detectable_effect`; nothing new is needed but compute.
3. **A full-PMB measurement of the DPO stage** (E1). No `pra_lenient` or UAR figure exists for any
   DPO checkpoint; `dpo-v1-scale` appears only in pairwise comparisons.
4. **Fix the DPO lineage before measuring it** (E13). The preference pairs were generated *before*
   SFT v1's dedup fix and ratio rebalance: 2263/2277 user turns match the current `sft/v1/train.jsonl`
   but **0/2277 system texts** do, so the retrieval context differs from the corpus the SFT stage
   trained on. Regenerate the pairs, then measure the stage. The cause of the divergence was never
   pinned down and should be.
5. **H22 abliteration** (24 probes built, unrun) and **H24 emotional range** (27 probes across 9
   registers, unrun).
6. **Image-derived memory tiers** (RQ13, second half) — the base model is a VLM, but nothing in the
   memory tier comes from an image.
7. **Score the quantized models on PMB** (E6). The quantization work was `llama-bench` plus a
   coherence check; no quality benchmark was ever run against a quantized checkpoint, so "Q3_K_S is
   the smallest verified-coherent level" rests on manual generation inspection.
8. **Resolve the two GGUF size sets** (E5). `README.md` and the cards' files tables say 8.64 / 4.61
   / 3.18 GiB; `hf_readmes/gguf_README.md`'s own evaluation table says 8.62 / 4.59 / 3.17 GiB. No
   artifact backs either. Re-run `llama-bench` against the GGUF files, commit the output, and
   correct whichever set is wrong.

### No GPU needed

9. **Commit the quantization evidence** (E6). No `llama-bench`, `llama-imatrix` or `llama-quantize`
   output is committed (`*.gguf` is gitignored and `results/` holds no quantization directory), so
   every size, throughput and perplexity figure is restated from run logs a reader cannot check.
   Even without re-running the benchmarks, a `results/quantization/` directory holding the original
   stdout would close most of this.
10. **Populate `acceptable_alternatives`, or drop the field** (E22). **Closed 2026-11-11 without
    populating it** (E41). The item's premise was wrong: `pra_strict` is ~0 because it compares the
    whole response with the gold answer (3 of 4816 responses match across all 7 full-harness runs),
    and alternatives are compared the same way, so populating them would not change it. It would
    change `pra_lenient`, whose judge rubric includes the alternatives, and the probe file is frozen.
    The field stays, documented as unpopulated. A judge-free cross-check (does the response contain
    the answer) agrees with the judge on Study 001's runs, but that rule was written after seeing
    the responses. It is pre-registered for Study 002, unchanged, as the secondary cross-check
    `pra_contain` in `docs/preregistrations/study002-answer-containment.md` (2026-11-11), with its
    known failure modes and the report every Study 002 run owes it.
11. **Widen the k-sweep.** It covers 120 of 688 probes (15 per category, seed 1337) and is reported
    as an inverted U peaking at k=8. At n=120 the curve's shoulders are not distinguishable from
    noise; a full-688 run costs inference, not training. Widening k alone will not help: with the
    default tier quota the k=16 run retrieved the same list as k=8 for 118 of 120 probes (E42), so
    the search ordering and the quota have to be fixed first.
12. **Replace `mur`** (E36). The harness calls it, and it is **0.0 in every saved run** because the
    IDs it compares never match: probes cite benchmark-authored memory IDs, while the store assigns a
    random UUID at insert. Its Week-1 proxy also measures retrieval, not the answer, so even with the
    IDs reconciled it could not test H5. A response-level memory-utilisation measure is needed, or
    the metric should be dropped from the reported set.
13. **Pin `base_model_revision` in the DPO and distillation configs** (E4). `distill_v1.yaml`
    (base and teacher), `dpo.yaml`, `dpo_v1_more_epochs.yaml` and `dpo_v1_scale.yaml` all reference
    `"main"`. Until they pin a SHA, no run launched from them can be called reproducible.

    **Done 2026-11-11.** The frozen configs stay as Study 001's record. Pinned equivalents are in
    `configs/training/study002/`, and `scripts/validate.py` now fails any config there whose model
    is not a Hub repo id or whose revision is not a 40-char SHA. Two findings shaped them:
    - The frozen bases are local folders (`outputs/*/merged`), for which `revision` is ignored, so
      a SHA there would not have pinned anything. Each Study 002 config loads the published copy
      instead, at the commit that added its weights (2026-08-14; unchanged as of 2026-11-11).
      That each copy is byte-identical to the local folder Study 001 trained from is likely but
      unverified: compare `sha256sum outputs/*/merged/model.safetensors` on the GPU box with the
      values in `results/audit/hub_revisions/hub_revisions.json`.
    - `main` has moved. On 2026-11-03 both Gemma models changed `tokenizer.json` (nothing else). The
      distillation teacher is pinned to `google/gemma-4-E4B-it` at `ee0ef60`, the commit `main`
      resolved to when Study 001's distillation ran (resolved from Hub history, not logged at the
      time; the teacher reads the student's token ids, so the pin is for exactness).
      `run_crossover_baseline.py`, which does use E4B's own tokenizer, is pinned to the same commit.

    `src/onebee/training/distill.py` also now passes the student's revision to trl, which reads it
    only from `model_init_kwargs`; before, a Hub student would have loaded from `main` whatever the
    config said. `tests/unit/test_study002_pins.py` checks every pin reaches its load.

    Outside this item and still unpinned: the evaluation and comparison scripts that load local
    `outputs/*/merged` folders (`run_system_b*.py`, `run_system_e*.py`, `compare_*.py`,
    `run_emotional_range_eval.py`), and `scripts/model_bakeoff.py`, which defaults to `main` but
    records the commits it resolved. A Study 002 run of any of them should load the published copy
    at its pinned commit instead.
14. **A small blinded human evaluation.** No human review exists anywhere: every datasheet says "not
    human-reviewed", and no reviewer log or teacher transcript was retained, so the
    `gpt-5.6-luna` teacher identity is self-reported and unverifiable from the repository. A
    few dozen probes scored by a human would bound how much the judge-based metric can be trusted.
15. **Re-score with an unrelated judge** (E36, E38). Every `pra_lenient` and pairwise verdict was
    scored by `gpt-5.6-luna`, which also wrote the SFT and DPO training targets. Re-judge the saved
    responses (every `results/**/raw.jsonl`, including the two pairwise sets, which store both
    responses per probe, and the PCS analysis where it can be re-run from saved responses) with a
    judge from a different model family, and report, per system, the change in `pra_lenient` and
    the agreement between the two judges. A gain that shrinks under the second judge measured the
    judge, not the model. API cost only; nothing is regenerated.
16. **H3.1: ORPO vs DPO** (E40, needs a GPU). ORPO is available as `trl.experimental.orpo` in the
    pinned trl. Train it on the same preference pairs as DPO v1_scale (after item 4's lineage fix) and
    compare at matched compute. The experimental API may change between releases, so record the exact
    trl version in the run manifest.

**Not Study 002:** correcting a Study 001 claim. Those go in `reports/ERRATA.md` against the frozen
artifact. If a *result* changes, it is a Study 002 result and is reported as one.
