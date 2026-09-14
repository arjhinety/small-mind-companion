# Studies

This project is organised into studies. A study is **frozen** when its findings are final: its
evidence is hash-pinned, its write-ups stop changing, and its repository links resolve at a git
tag. Work done after a freeze belongs to the next study.

The rule exists because the alternative — a repository whose numbers quietly drift as the
interesting result changes — is exactly what the audit found 33 instances of. See
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
quantization figure is backed by a committed artifact. The `acceptable_alternatives` field is
unpopulated in all 688 probes, which is why `pra_strict` is ~0 and `pra_lenient` is reported
instead. Everything is a single seed and a single run. There is no human evaluation.
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
10. **Populate `acceptable_alternatives`, or drop the field** (E22). It is present on all 688 probes
    and empty in 688/688, which is why `pra_strict` is ~0 throughout and `pra_lenient` is the
    reported metric. Populating it makes `pra_strict` meaningful; dropping it removes the need to
    explain it away.
11. **Widen the k-sweep.** It covers 120 of 688 probes (15 per category, seed 1337) and is reported
    as an inverted U peaking at k=8. At n=120 the curve's shoulders are not distinguishable from
    noise; a full-688 run costs inference, not training.
12. **Measure `mur`.** It is defined and implemented in the harness and is **0.0 in every saved
    run** — because it is computed but not called, not because the models score zero. Either wire it
    up or remove it from the reported metric set.
13. **Pin `base_model_revision` in the DPO and distillation configs** (E4). `distill_v1.yaml`
    (base and teacher), `dpo.yaml`, `dpo_v1_more_epochs.yaml` and `dpo_v1_scale.yaml` all reference
    `"main"`. Until they pin a SHA, no run launched from them can be called reproducible.
14. **A small blinded human evaluation.** No human review exists anywhere: every datasheet says "not
    human-reviewed", and no reviewer log or teacher transcript was retained, so the
    `gpt-5.6-luna` teacher identity is self-reported and unverifiable from the repository. A
    few dozen probes scored by a human would bound how much the judge-based metric can be trusted.

**Not Study 002:** correcting a Study 001 claim. Those go in `reports/ERRATA.md` against the frozen
artifact. If a *result* changes, it is a Study 002 result and is reported as one.
