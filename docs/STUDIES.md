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

Candidates, in rough order of how much of Study 001 they unblock:

1. **The crossover experiment (H16/H17).** `run_crossover_baseline.py` is written and unrun: 2B +
   scaffold vs the 8B `gemma-4-E4B-it` with no memory. This is Study 001's own thesis question and
   the one experiment that answers it. Needs a GPU.
2. **Multi-seed replication.** Every Study 001 number is one seed. Finding 1 in particular (the
   −5pp UAR drop from adding memory) is the kind of claim that needs ≥3 seeds and a paired
   bootstrap before it should be relied on.
3. **Populate `acceptable_alternatives`, or drop the field.** This is what makes `pra_strict`
   meaningful and removes the need to explain it away.
4. **Score the quantized models on PMB.** The quantization work was `llama-bench` plus a
   coherence check; no quality benchmark was ever run against a quantized checkpoint.
5. **Fix the DPO lineage.** No full-PMB measurement of the DPO checkpoint exists, and the DPO
   preference pairs were generated *before* SFT v1's final regeneration (0/2277 system texts
   match). Regenerate the pairs and measure the stage properly.
6. **H22 abliteration** (24 probes built, unrun) and **H24 emotional range** (27 probes across 9
   registers, unrun).
7. **Image-derived memory tiers** (RQ13, second half) — the base model is a VLM but nothing in the
   memory tier comes from an image.
8. **Pin `base_model_revision` in the DPO and distillation configs** before any re-run is described
   as reproducible.

**Not Study 002:** correcting a Study 001 claim. Those go in `reports/ERRATA.md` against the
frozen artifact. If a *result* changes, it is a Study 002 result and is reported as one.
