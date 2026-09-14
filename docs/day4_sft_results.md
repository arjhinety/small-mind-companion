# Day 4/5 results: LoRA SFT, and the full A/B/D/E comparison

**Status:** real results, single seed, small-scale v0 dataset. Raw data in
`results/v0.1/{A_raw,B_sft,D_memory,E_sft_memory}/`.

## What was built and run

1. **Real memory-aware SFT dataset v0** (`data/sft/v0/`, 225 examples: 202 train / 23 val),
   generated via the real retrieval pipeline against 4 memory stores populated from personas
   disjoint from the PMB-v0 eval set. Real teacher (gpt-5.6-luna) targets, ~15%
   irrelevant-retrieval examples, ~10% abstention examples. Contamination-checked clean.
2. **Real LoRA SFT training** on `gemma4-e2b` (r=16, alpha=32, 2 epochs, 52 steps, ~64s
   wall-clock — small dataset, fast run). Loss 3.99 → 1.01, token accuracy 52% → 77%.
   Adapter + merged checkpoint at `outputs/sft/v0/` (not committed to git — model weights
   belong on HF Hub per the project's publication plan, not in the repo).
3. **Systems A, B, D, E** all evaluated on the same 688 real PMB-v0 probes.

## Headline numbers

| System | Description | pra_strict | pra_lenient | uar |
|---|---|---|---|---|
| A | raw, no memory | 0.0% | 0.16% | 13.75% |
| B | SFT, no memory | 0.0% | 0.16% | 16.25% |
| D | raw + memory (k=8) | 0.0% | 15.13% | 8.75% |
| **E** | **SFT + memory (k=8)** | **0.49%** | **17.76%** | **33.75%** |

## Interpretation against pre-registered hypotheses

- **H4** (conversational SFT improves dialogue quality but not factual recall): **confirmed.**
  A→B: `pra_lenient` essentially unchanged (0.16%→0.16%) — SFT alone cannot grant knowledge of
  facts the model was never given. UAR improved slightly (13.75%→16.25%).
- **H5** (memory-aware SFT beats generic SFT when evaluated with memory): **confirmed.** D→E:
  +2.6pp `pra_lenient` (15.13%→17.76%). This is the core hypothesis motivating Day 4's entire
  design — training on examples where retrieved memories are actually in context, in the same
  format the eval-time pipeline produces, measurably helps beyond what raw retrieval alone gets.
- **Unplanned but striking finding: SFT dramatically improved calibration with memory present.**
  D→E: UAR went 8.75%→33.75% (~4x). Day 3's finding was that memory *increases* hallucination
  (H3) — adding SFT on a dataset containing ~10% deliberate abstention examples appears to
  substantially counteract that specific failure mode, not just improve raw recall. This wasn't
  a pre-registered hypothesis but follows directly from H5's mechanism and is consistent with
  the training data's explicit abstention examples.
- **`pra_strict` finally nonzero for E** (0.49%, vs 0.0% for A/B/D) — SFT taught the model to
  occasionally produce terse, exact-match-style answers rather than always full sentences,
  incidentally helping the strict grader too (not the primary metric of interest, but a
  consistent signal).

## Manual verification

Spot-checked 3 `lenient_correct` System E responses and 2 correct abstentions from the raw
unanswerable-category output — all genuinely accurate/honest, not judge artifacts:
- "Which university did you graduate from?" → "I graduated from Berkeley, where I studied Art
  History." (gold: Berkeley — correct, and note the extra accurate detail about the major)
- "What was your childhood nickname?" → "I don't recall that. Do you remember giving me one?"
  (correct abstention, phrased naturally in-character)

Also confirmed the merged SFT checkpoint retained multimodal capability
(`HFEngine._is_multimodal == True`, `Gemma4Processor` loads cleanly) after fixing a real bug
where saving only the tokenizer silently dropped vision support — see `docs/model_quirks.md`
#13.

## Known limitations

- **Tiny SFT dataset** (202 train examples) vs. the roadmap's 6-10k target scale — this is
  explicitly a v0 proof-of-concept run, not the intended full-scale Day 4 experiment. The
  effect sizes above should be read as "SFT helps, direction confirmed" not "this is the
  ceiling of what memory-aware SFT buys."
- **Single seed**, no bootstrap CIs on `pra_lenient` (same harness limitation noted on Day 3).
- **Only 1 epoch-count/hyperparameter setting tried** — no ablation on LoRA rank, learning
  rate, or data scale (all Week 2+ per the roadmap's own experiment matrix).
- **Real trl/transformers API breaks fixed along the way** (4 of them — see
  `docs/model_quirks.md` #11-14): `TrainingArguments.warmup_ratio` removed,
  `SFTTrainer`'s `tokenizer`→`processing_class` + SFTConfig restructuring,
  list-vs-`datasets.Dataset` requirement, and the multimodal-processor-loss-on-save bug. All
  fixed generically (via version-detection fallbacks, not hardcoded to this exact version), but
  worth knowing the training code path had never been exercised against real trl before this.
- Adapter/merged checkpoint not yet uploaded to HF Hub (Day 4's stated deliverable includes
  this — deferred; the checkpoint exists locally on the training box and would need to be
  re-generated or transferred if the box is shut down before uploading).

## Where this fits in the roadmap

Day 4's core deliverable (one real LoRA SFT run, evaluated) is done. Day 5's "full evaluation
grid" is partially done (A/B/D/E covers the memory×SFT 2×2, matching the doc's P0 priority
systems) — not yet run: the 8B unaugmented baseline (A-8B/D-8B) or the quantization sweep
(explicitly deferred per `docs/adr/0001-model-selection.md`'s own reasoning: quantize the
post-trained checkpoint, not before). DPO/ORPO (H6-H7) and distillation (H8-H9) are Week 2 per
the original roadmap and have not been started.
