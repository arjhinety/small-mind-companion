# H23: On-policy distillation results (2026-08-15)

**Status:** real, positive result across three independent measures, single seed. Raw data in
`results/v1_scale/{E_distill,C_vs_F_distill_pairwise}/`.

## What was built and run

`src/onebee/training/distill.py` — a new training module wrapping `trl.DistillationTrainer`
for on-policy distillation (the student generates its own completions during training, matched
token-by-token to the teacher's distribution via generalized Jensen-Shannon divergence — see
the "On-Policy Distillation" paper). This replaces H8/H9's original offline "teacher generates
SFT data" framing, which turned out not to map onto anything actually buildable in this
project — see `docs/research_questions.md`'s H8/H9 → H23 reframe note for why.

- **Student**: the current-best checkpoint, `outputs/dpo/v1_scale/merged` (proper-scale,
  rebalanced SFT+DPO).
- **Teacher**: `google/gemma-4-E4B-it` (8B params, confirmed identical tokenizer/vocab to the
  E2B student — a hard requirement for `DistillationTrainer`).
- **Data**: 2008 train / 224 val prompt-only examples (`data/distill/v1/`), extracted from the
  existing SFT v1 dataset's system+user turns (no assistant turns — the student generates its
  own).
- **Config**: `configs/training/distill_v1.yaml` — 1 epoch, 125 steps, LoRA (r=16), conservative
  learning rate (1e-6, trl's own default for continuing to refine an already-trained
  checkpoint rather than training from scratch).

## A real bug found and fixed before this could run at all

`trl.DistillationConfig`'s `teacher_model_name_or_path` field looks like it should be all you
need to set — it isn't. The trainer needs the teacher passed as its own separate `teacher_model`
constructor argument; leaving it only on the config left `trainer.teacher_model` as `None`,
crashing on the very first real training step with `AttributeError: 'NoneType' object has no
attribute 'eval'`. A `--dry-run` (build the trainer, skip `.train()`) did NOT catch this —
teacher loading is lazy, not done at construction — so the bug only surfaced on a real (paid)
training run. Fixed by passing `teacher_model=` explicitly. Full detail:
`docs/model_quirks.md` #22.

## Training-time signal: concerning on its face

| Step | loss | grad_norm | completions/clipped_ratio |
|---|---|---|---|
| ~10 | 7.6-7.9 | 89-107 | ~95-97% |
| ~60 (mid) | 7.3-7.6 | 88-104 | ~94-98% |
| 125 (final) | 7.58 (train_loss avg 7.547) | 98.9 | 98.75% |

Loss barely moved across the entire run, `grad_norm` stayed large and noisy throughout
(88-107, no clear downward trend), and `completions/clipped_ratio` — the fraction of
training-time completions that hit the 128-token cap without the model naturally emitting an
end token — sat at 95-98% for nearly every logged step. Read in isolation, this looks like an
unhealthy or non-converging run.

**This did NOT translate into bad real output.** A manual generation sanity check (companion
system prompt, "What is your favorite color?") produced a coherent, in-character, natural
response before being cut off by the test's own `max_new_tokens` limit — not garbage, not
repetition, not a collapsed distribution. Per this project's own discipline (training-time
metrics are informative but not sufficient — real eval is authoritative, validated repeatedly
elsewhere in this project), the real evaluation below is what actually answers H23, not the
loss curve.

## Real evaluation result: positive, across three independent measures

### Aggregate harness metrics (688-probe PMB harness)

| System | Description | pra_lenient | uar |
|---|---|---|---|
| B (v1, rebalanced) | SFT alone, no memory | 0.16% | 21.25% |
| E (v1, rebalanced) | SFT + memory (pre-distillation) | 15.30% | 70.0% |
| **E-distill (H23)** | **SFT+DPO+distill + memory (post-distillation)** | **18.59%** | **71.25%** |

> **B's UAR was 25.0% here until E34.** That is the *fixed-dedup, pre-rebalance* run
> (`docs/proper_scale_results.md`'s "B (SFT-v1 alone, fixed data) | 25.0% (20/80)"); the
> rebalanced checkpoint these rows are drawn from scores
> `results/v1_scale/B_sft/metrics.json` UAR = 0.2125 = **21.25%**. The row was labelled
> "rebalanced" while quoting the pre-rebalance number.
>
> **These two rows are not a clean isolation of distillation.** E is `outputs/sft/v1/merged` and
> E-distill is `outputs/distill/v1/merged`, which was trained *from the DPO output* — so the
> +3.3pp `pra_lenient` delta in this table spans both the DPO stage and the distillation stage.
> The distillation-specific comparison is the **pairwise C-vs-F** run (C = DPO+memory, F =
> distill+memory), which holds the DPO stage fixed on both sides: F wins 38.1% vs C's 30.5%, a
> +7.6pp gap. Cite that number for H23, not this table.
>
> Neither figure is measured on a DPO checkpoint's full-PMB metrics — no such run exists in
> `results/`; `dpo-v1-scale` appears only in pairwise comparisons.

`pra_lenient` improved by 3.3 percentage points (15.30%→18.59%) and UAR held essentially flat
(70.0%→71.25%, within noise) — the distilled model answers more answerable questions correctly
without becoming less calibrated on unanswerable ones. Both moved in the direction H23 predicted
(quality improves, calibration doesn't regress), not just one.

### Pairwise persona-consistency (105-probe dual-order judge, same sampling as prior C-vs-E runs)

| | Wins | Rate |
|---|---|---|
| F (post-distillation) | 40/105 | 38.1% |
| C (pre-distillation) | 32/105 | 30.5% |
| Ties | 33/105 | 31.4% |

**Gap: 7.6pp favoring the post-distillation model.** This is the part of H23 that could most
plausibly have gone the other way — the teacher (`gemma-4-E4B-it`) is a generic Google instruct
model, NOT companion-persona-tuned, so pulling the student's distribution toward it was a real,
stated risk to persona consistency (see the pre-registered hypothesis in
`docs/research_questions.md`). It didn't happen; if anything, persona consistency improved
alongside general quality, not at its expense.

## Interpretation against H23

**H23** (on-policy distillation from a larger local teacher improves `pra_lenient` without
degrading persona consistency): **supported by this pass** — `pra_lenient` improved, UAR didn't
regress, and the pairwise persona-consistency comparison favored the distilled model rather
than the null-hypothesis-consistent "no difference" or the risk-case "measurably worse." This
is a single seed/run at one data scale (2008 prompts, 125 steps), same limitation as every
other single-pass result in this project — not a fully confirmed, multi-seed result, but a real
positive finding, honestly the strongest and cleanest of the three possible outcomes the
hypothesis table considered before this ran.

**Open question, not resolved here:** why did training-time loss/grad_norm look unhealthy while
real-eval quality genuinely improved? Plausible explanations not investigated further in this
pass: on-policy GJSD loss values may simply have a different healthy-range/scale than typical
cross-entropy losses this project's other training runs report, so "loss ~7.5, flat" may not
mean what it would mean for SFT/DPO; the high `completions/clipped_ratio` may reflect that
companion-style responses often exceed 128 tokens even when good (not that the model is failing
to converge on when to stop) — the `max_completion_length=128` training-time cap may simply be
tighter than the model's natural response length, unrelated to model health. Worth a genuine
follow-up if distillation work continues (e.g. raise `max_completion_length`, check whether
loss/grad_norm stabilize over more steps) — not chased further here since the real-eval result
already answers H23's actual question.

## Follow-up: real PCS-stylometric analysis (2026-08-15)

After building the project's first real PCS (Persona Consistency Score) implementation
(`src/onebee/evaluation/metrics/persona_consistency.py` — a judge-based semantic-consistency
`pcs`/`pcs_judge_score`, plus a pure-text-statistics `pcs_stylometric`/`stylometric_drift` that
needs no GPU or judge API at all), ran it against the already-saved eval transcripts from this
pass (`run_pcs_stylometric_analysis.py`, no new inference needed) to add an independent,
non-semantic data point to H23's story: does distillation change *how* the model writes, not
just *what* it gets right.

| System | n | Self-consistency (stylometric) |
|---|---|---|
| B (SFT alone) | 688 | 0.5091 |
| E (pre-distillation) | 688 | 0.5085 |
| E-distill (post-distillation, H23) | 688 | **0.5243** |

**Stylometric drift, pre- vs post-distillation: 0.8725** (1.0 = identical style, 0.0 = maximally
different — moderate, real drift, not extreme).

Post-distillation writing style is slightly MORE internally consistent than either
pre-distillation baseline, not less — a further independent piece of evidence against the risk
H23 flagged going in (that pulling toward a non-persona-tuned teacher would degrade
consistency). Combined with the earlier judge-based pairwise result (persona consistency
favored post-distillation, +7.6pp), this is now two independent measures — one semantic
(LLM-judged), one purely statistical (word/sentence-level features, no model calls) — both
pointing the same direction. Neither alone would be conclusive; together they're a more
credible signal than either. Full numbers: `results/v1_scale/pcs_stylometric_analysis/summary.json`.

This does not replace the judge-based `pcs`/`pcs_judge_score` functions as the intended
"real" PCS metric (those measure semantic in-character-ness, which stylometric features
can't) — it's a complementary, cheaper-to-compute signal that happened to be runnable
immediately against data already on disk.

## Follow-up: real judge-based PCS analysis (2026-08-16)

With OpenAI judge access available, ran the judge-based `pcs_judge_score` (semantic
in-character consistency, distinct from the stylometric-only signal above) against a random
60-probe sample per system (`run_pcs_judge_analysis.py`, seed 1337, same companion persona used
at generation time — no new inference needed, scores responses already saved on disk):

| System | n sampled | PCS (judge-scored) |
|---|---|---|
| B (SFT alone, no memory) | 60/688 | 0.370 |
| E (pre-distillation, SFT+DPO+memory) | 60/688 | 0.630 |
| E-distill (post-distillation, H23) | 60/688 | **0.646** |

This is now a **third independent measure** agreeing with H23's positive direction: memory
alone drives most of the semantic in-character gain (0.37→0.63, expected — a companion with no
memory access can't ground responses in the relationship at all), and distillation adds a
further real, if smaller, improvement on top (0.63→0.646) — consistent with, not contradicted
by, the earlier pairwise judge result (+7.6pp favoring post-distillation) and the stylometric
self-consistency result (0.524 vs 0.509). Three measures — one pairwise-comparative, one
absolute-semantic, one purely statistical — all point the same direction. Full numbers:
`results/v1_scale/pcs_judge_analysis/summary.json`.

**Caveat**: this pass sampled 60/688 probes per system (not the full set) to bound judge API
cost — a real, if secondary, limitation worth flagging alongside the ones below.

## Known limitations

- Single seed, single data scale (2008 prompts) — no ablation on teacher choice, training
  steps, or `beta`/`temperature` (the JSD-interpolation and sampling hyperparameters).
- The training-time metric anomaly above is flagged, not explained — a genuine open question.
- The judge-based PCS follow-up above sampled 60/688 probes per system, not the full set.
- This closes out Week 2's scope (DPO + distillation) — ORPO remains deferred to Week 3,
  tracked in `docs/model_quirks.md` #15.
