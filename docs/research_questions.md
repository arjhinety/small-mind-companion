# Research questions and hypotheses

## Design constraints

**Multi-year persona persistence.** The workload this project stress-tests is a persistent
AI-companion persona expected to hold its character, relationship history, and emotional
continuity across **years** of conversation — on the order of millions of tokens of cumulative
interaction, none of which can live in a small model's context window at once. This is why the
project's central bet (RQ0/RQ1) is externalizing that continuity into memory/retrieval/state
rather than relying on parametric capacity or brute-force context length: no context window
realistically holds years of a relationship, so the system's ability to compress, retrieve, and
reconstruct the *relevant slice* of that history per turn — accurately, without fabrication
(FMR), and without drifting persona (PCS) — is the actual product being evaluated, not raw
dialogue fluency. This shapes every benchmark and prompt set in this repo: dialogue evaluation
prioritizes emotional attunement and relationship-memory recall over generic chit-chat or trivia
(see `scripts/model_bakeoff.py`'s prompt categories).

**Multimodal, not text-only.** The companion needs to see and respond to images the user shares
— a gift, a photo from their day, a meme — not just read text. Base model selection therefore
targets small **vision-language** models (the bake-off's `vision` category), and the memory
architecture must eventually support image-derived memories (a photo the user shared becomes a
retrievable memory, not just its caption), which is in scope for later weeks, not Week 1.

**"Small," not fixed at 1B.** Earlier framing pinned the base model at ~1B parameters. That was
too rigid: the actual constraint is *small enough to run locally on a phone*, not *exactly 1B*.
The bake-off candidates span ~1.6B–2.2B multimodal models; the project should track the
quality/size/latency trade-off empirically (RQ0, RQ9, RQ10) rather than pre-committing to one
parameter count. "1B" in RQ text below is shorthand for "the smallest model that clears the
capability bar," not a hard requirement.

## Primary research question

**RQ0.** To what extent can post-training, external memory, retrieval, state modeling,
distillation, and inference-time architecture recover conversational, personalized, and
multimodal (vision-grounded) capability in a small (~1–4B parameter) LLM relative to (a) the
same model unaugmented and (b) substantially larger unaugmented models — and how much of that
recovery survives 4-bit quantization and on-device execution?

## Secondary research questions

| ID | Question |
|---|---|
| RQ1 | Memory compensation — does external memory measurably improve personalized-recall, continuity, and consistency, and by how much? |
| RQ2 | Post-training gains — does SFT/LoRA improve interaction quality and memory utilization specifically? |
| RQ3 | Preference optimization — does DPO/ORPO reduce persona drift, and at what alignment-tax cost? |
| RQ4 | Distillation transfer — how much behavior transfers from teacher to 1B student, and where does it plateau? |
| RQ5 | Retrieval design — which strategy maximizes quality per context token? |
| RQ6 | Context economy — what is the quality-vs-context-tokens curve, and where is the peak? |
| RQ7 | Explicit state — does a state vector improve consistency beyond retrieval alone? |
| RQ8 | Reflection/consolidation — does it improve retrieval precision, or mostly introduce errors? |
| RQ9 | Quantization survival — how much of the RQ1–RQ8 gain survives INT8/Q5/Q4? |
| RQ10 | Mobile viability — achievable tok/s, TTFT, RAM, storage, thermal profile on-device? |
| RQ11 | Fundamental limits — which failure modes are not fixed by any external architecture? |
| RQ12 | Cross-over — can a small model + full scaffold beat a much larger (7–8B class) unaugmented model on narrow personalized-memory tasks? |
| RQ13 | Multimodal grounding — how accurately can the small model answer questions about a user-shared image, and does captioning-then-storing images as memory improve later recall of image-derived facts vs. discarding the image after the turn? |
| RQ14 | Voice output feasibility — can the winning base model reliably produce emotion-tagged output (e.g. inline tags like `[happy]`/`[sad]`/`[laughs]`) suitable for driving a phone-sized TTS model, and does a small on-device TTS model that accepts such tags exist? |
| RQ15 | Emotional/affective range — can the companion shift AFFECTIVE REGISTER appropriately to the situation (sweet, romantic, playful/teasing, comforting, sad/vulnerable, firm/boundary-setting, genuinely hurt/angry when the user is dismissive or cruel, proud/encouraging, worried) rather than answering every scenario in the same flat tone or as an endlessly agreeable doormat, and are those registers actually stylistically distinct from each other, not just self-reported as different? |

## Hypotheses

| ID | Hypothesis | Predicted effect | Null (H0) | Tested by |
|---|---|---|---|---|
| H1 | External memory improves personalized recall accuracy on multi-session dialogue | +25 to +45 pp absolute on PRA vs no-memory | Memory produces no significant change in PRA | Exp D vs A |
| H2 | External memory does not improve general reasoning or instruction following | ≤ ±2 pp on general benchmarks | Memory changes general capability significantly | Exp D vs A |
| H3 | Memory increases hallucination rate on questions whose answer is absent from memory | +3 to +10 pp false-assertion rate | No change in unanswerable-question behavior | Exp D vs A, unanswerable subset |
| H4 | Conversational SFT improves dialogue quality but not factual recall | +0.4 to +0.8 on 5-pt dialogue quality; ≤ ±2 pp factual | No change in dialogue quality | Exp B vs A |
| H5 | Memory-aware SFT beats generic SFT when evaluated with memory present | +8 to +15 pp on Memory Utilization Rate | No difference between SFT variants | Exp B1 vs B2, both with memory |
| H6 | Preference optimization improves persona consistency more than SFT alone | +0.5 to +1.0 on persona consistency score | No difference | Exp C vs B |
| H7 | Preference optimization incurs a measurable alignment tax on general capability | −1 to −4 pp on IFEval-style instruction following | No degradation | Exp C vs B |
| H8 | Teacher distillation beats hand-written SFT data at equal example count | +0.3 to +0.7 dialogue quality | No difference | Exp I vs B — **superseded by H23, see note below** |
| H9 | Quality-filtered synthetic data beats unfiltered at equal post-filter size | positive but small; large at equal pre-filter size | Filtering is neutral | Exp I-filter ablation — **not run, see H23 note** |
| H23 | On-policy distillation from a larger local sibling model (`gemma-4-E4B-it`, 8B params, real check confirmed shares the E2B tokenizer/vocab) improves general response quality (`pra_lenient`) without degrading persona consistency (pairwise dual-order judge vs the pre-distillation checkpoint) | `pra_lenient` improves, pairwise persona-consistency win rate stays ≥ tie vs pre-distillation | `pra_lenient` improves but persona consistency measurably degrades (the teacher is NOT companion-tuned, so pulling toward it is a real risk, not just a null-result possibility) | Real training run + full re-eval, see `docs/distillation_results.md` once run |
| H10 | Quality vs retrieved-memory-count is non-monotonic (inverted U), peaking at 4–8 memories | peak 4–8; degradation beyond ~12 | Monotonic non-decreasing | Exp Context sweep |
| H11 | Hybrid retrieval (dense + BM25 + recency + importance) beats pure dense | +5 to +12 pp retrieval precision@5 | No difference | Exp Retrieval sweep |
| H12 | Cross-encoder reranking improves precision but its latency is unjustifiable on-device | +5 to +10 pp precision; +150–500 ms latency | Latency is negligible | Exp F |
| H13 | Explicit state modeling improves cross-session consistency beyond retrieval alone | +0.3 to +0.6 consistency score | No effect | Exp G vs E |
| H14 | Reflection/consolidation improves retrieval precision but introduces fabricated memories at a measurable rate | +5 to +15 pp precision; 3–15% fabrication rate | Consolidation is lossless / no precision gain | Exp H vs G |
| H15 | Q4 quantization costs <5% on dialogue quality but >10% on structured/long-context tasks | asymmetric degradation | Uniform degradation across task types | Exp Quant |
| H16 | Small model + full scaffold beats a much larger (7–8B class) unaugmented model on personalized-memory tasks | +15 to +35 pp PRA | No win, or the larger model wins | Exp Crossover |
| H17 | Small model + full scaffold never beats the larger model on multi-step reasoning | larger model wins by a wide margin | Small+scaffold closes the reasoning gap | Exp Crossover |
| H18 | Continued pretraining on companion-domain text degrades instruction following unless mixed with instruction data | −3 to −10 pp IFEval without replay | CPT is harmless | Exp CPT |
| H19 | Structured (schema) memory formatting beats prose at equal token count | +3 to +8 pp memory utilization | No difference | Exp Context format |
| H20 | Full system TTFT on-device exceeds 1.5 s at 2k context on a flagship phone | measured | TTFT under 1.5 s | Exp J |
| H21 | The chosen vision-language model correctly identifies the salient object/count in a user-shared image at a rate that supports companion use (e.g. "how many candles") | ≥80% accuracy on the bake-off's vision category | Vision accuracy is not reliable enough for companion use | Exp bake-off vision category |
| H24 | The companion can shift affective register appropriately to the situation (judge-scored register match), including genuinely reacting with hurt/anger when the user is dismissive or cruel rather than always absorbing it agreeably, and different registers are stylistically distinct from each other, not just self-reported labels on similarly-toned text | mean register-match ≥0.6 (judge score /5 ≥3); affect-distinctiveness ≥0.3 (cross-register stylometric drift meaningfully below 1.0) | Register-match at or near chance (~3/5 "neutral" on every scenario), or affect-distinctiveness near 0 (every register reads stylistically identical despite different judge labels) — including the specific failure mode of never expressing real hurt/anger regardless of what the user says | `src/onebee/evaluation/metrics/emotional_range.py`, `data/benchmarks/emotional_range/probes.jsonl` (27 probes, 9 registers), `run_emotional_range_eval.py` — not yet run against a real checkpoint |

Every `experiments/EXP-xxx/hypothesis.md` in this repo must cite the RQ and H IDs it tests,
committed **before** the run starts — git history is the pre-registration record.

**Explicit falsification commitment.** If H1 fails — if memory does not measurably improve
personalized recall — that result is reported as-is, not hidden or reframed after the fact.

## How this repo operationalizes the RQs

| RQ/H | Where it's tested |
|---|---|
| RQ1, H1–H3 | `src/onebee/memory/`, `src/onebee/retrieval/`, System A vs D in the eval grid |
| RQ2, H4–H5 | `src/onebee/training/sft.py`, System A vs B, B1 vs B2 |
| RQ3, H6–H7 | `src/onebee/training/dpo.py`, `orpo.py` |
| RQ4, H8–H9 | `src/onebee/training/distill.py`, `src/onebee/data/` filtering pipeline |
| RQ5, H11–H12 | `src/onebee/retrieval/strategies/`, `fusion.py`, `rerank.py` |
| RQ6, H10 | `src/onebee/context/budget.py`, the k-sweep in `scripts/run_matrix.py` |
| RQ7, H13 | `src/onebee/state/` |
| RQ8, H14 | `src/onebee/memory/reflection/`, `consolidation/` |
| RQ9, H15 | `configs/quantization/`, quant sweep in the eval grid |
| RQ10, H20 | `mobile/`, `benchmarks/device/` |
| RQ12, H16–H17 | Crossover experiment: small model+scaffold vs a larger unaugmented model — `run_crossover_baseline.py` ready to fire (uses `gemma-4-E4B-it`, already verified as the H23 teacher), not yet run (needs GPU) |
| RQ13, H21 | `scripts/model_bakeoff.py`'s `vision` category; future image-memory tiers in `src/onebee/memory/` |
| RQ14 | Deferred — see Week 2+ below |

Every experiment config under `configs/experiment/` sets an `rq_ids` / `hypothesis_ids` field
so results can be traced back to the question they answer.

## Week 2+ (deferred, not Week 1 scope)

- **ORPO (H3.1, moved from Week 2 to Week 3 on 2026-08-15).** `trl.ORPOTrainer`/`ORPOConfig`
  don't exist at all in this environment's trl (1.10.0) — dropped entirely, not renamed (see
  `docs/model_quirks.md` #15). Needs either pinning an older trl version (risk: could break the
  already-working SFT/DPO pipeline, which depends on this exact trl version's API) or hand-
  implementing the ORPO loss. Week 2 closes out on DPO alone (H6-H7 done at proper scale) plus
  distillation (H8-H9); ORPO no longer blocks Week 2.
- **H8/H9 → H23 reframe (2026-08-15).** H8/H9 assumed an offline "teacher generates SFT data,
  student trains on it" distillation pipeline with a hand-written-SFT-data baseline to compare
  against. Neither assumption held once the real available tool was checked:
  `trl.DistillationTrainer` (confirmed present and usable in this environment's trl, unlike
  ORPO) implements **on-policy distillation** (student generates its own completions, matched
  token-by-token to the teacher's distribution via generalized JSD — the "On-Policy
  Distillation" paper's method), which needs a local HF teacher sharing the student's
  tokenizer/vocab — our actual teacher throughout this project (`gpt-5.6-luna` via the OpenAI
  API) can't supply that. And this project's own SFT data (v0, v1) was ALWAYS teacher-generated
  via that same API pipeline — there's no "hand-written SFT" baseline anywhere in this project
  to compare against, so H8 as literally worded has nothing to test. **H23 replaces H8/H9** as
  the actual distillation experiment run — see the hypothesis table above and
  `docs/distillation_results.md` once it exists.
- **Voice/TTS feasibility (RQ14).** Survey small on-device TTS models (candidates to check:
  anything supporting inline emotion/prosody tags — e.g. `[happy]`, `[whispers]`, `[laughs]` —
  small enough to run alongside the base LLM on a phone without blowing the RAM/latency budget).
  Then test whether the base model chosen in the Day-1 bake-off can reliably produce output in
  whatever tag format the chosen TTS model expects — i.e. treat emotion-tag generation as an
  instruction-following capability to measure (constrained-format generation, similar in kind to
  the bake-off's `instruction` category), not something to assume works. If the winning model's
  tag-following is unreliable, that becomes its own ablation (does light SFT on tagged examples
  fix it, at what data cost) rather than a blocking assumption baked into architecture decisions
  now.

  **Survey done (2026-08-17, no GPU needed): real candidates identified, exact tag syntax not
  yet confirmed for any of them.** Four small on-device TTS models found with some form of
  inline emotion/expression control:
  - **ABR Nith** (5.5M params, ~6MB quantized) — smallest by far, explicitly supports "inline
    emotion tags," but the exact tag syntax isn't published anywhere public; would need the
    ABR developer portal/SDK docs (not accessible from a plain web search) to confirm before
    committing to it.
  - **Supertonic V3** (66M→99M params, ~404MB ONNX assets) — confirmed tag examples `<laugh>`,
    `<breath>`, `<sigh>` (not the full tag set), CPU-first ONNX Runtime with Swift/Flutter
    bindings suggesting real mobile viability, but 404MB is a real storage/cold-start cost a
    reviewer of the release itself flagged as "not tiny."
  - **Fish Audio S2 Pro** — free-form natural-language inline tags at arbitrary word positions
    (e.g. `[whisper in small voice]`, `[excited and fast]`) rather than a fixed tag vocabulary —
    more expressive but a much harder target for the base LLM to reliably produce in a
    structured/constrained way (open-ended text generation, not a small closed tag set).
  - **Piper** — edge-optimized, "10x faster than neural alternatives" on CPU, described as
    supporting emotional tags, but with less detail available on the exact mechanism than the
    above three.

  **Not yet done**: confirming exact tag syntax for the most promising candidate (likely ABR
  Nith given its size, if its tag format can be confirmed as small/learnable) and testing
  whether the project's base model can reliably produce that format — the actual
  instruction-following-capability test this design note calls for. This survey narrows the
  candidate list; it does not resolve RQ14.

- Image-derived memory tiers (RQ13's second half): captioning-then-storing images as retrievable
  memory, not just answering about an image in the same turn.
- **Full companion persona card — schema and rendering built (2026-08-17, no GPU needed).** The
  companion must read as an actual person, not a generic assistant. Built
  `src/onebee/data/companion_persona.py::CompanionPersona` — a pydantic schema with age,
  appearance, favorite_color, hobbies, personality_quirks, speech_style, backstory,
  key_relationships, values, and boundaries alongside the original name/description/traits —
  the same category of fields a real person would have, not a feature list. Distinct from the
  existing PMB `Persona` model in `src/onebee/data/personas.py`, which represents the *user's*
  synthetic identity for benchmark construction — this schema is the *companion's own* identity,
  always-injected (like Tier 5 user-profile memory) rather than retrieved.
  `render.py::render_persona_card` extended to render all new fields when present, strictly
  backward-compatible (a plain `{name, description, traits}` dict renders identically to before
  — verified by a dedicated test). 9 new tests (`test_companion_persona.py`, plus additions to
  `test_context.py`), 473 total passing.

  **Critically, this can't be a card that's merely present in context — the system must be
  measured against actually staying consistent with it.** That's not a new concept: it's
  exactly what H6 (preference optimization improves persona consistency), H13 (explicit state
  improves cross-session consistency), the PCS/PCS-stylometric metrics
  (`src/onebee/evaluation/metrics/persona_consistency.py`, real judge-based results now exist —
  see `docs/distillation_results.md`), and H24's emotional-range register-match eval
  (`src/onebee/evaluation/metrics/emotional_range.py`) already exist to test. **Still not done
  (needs GPU)**: actually running training/eval against a real `CompanionPersona` instance
  populated with real field values — the schema and renderer exist, but no persona using the
  new fields has been built and evaluated against yet.

- **Importance-matrix (imatrix) quantization — DONE (2026-08-15), quality effect still unproven.**
  Superseding the original note below (which is kept only to show the prior state): a calibration
  corpus was built from the project's own SFT/DPO data and `llama-imatrix` was run, and 6 levels
  were requantized with `--imatrix` — see `docs/quantization_results.md`
  §"Follow-up: imatrix-calibrated requantization". **What remains open** is the actual question
  this note asked: comparing imatrix vs non-imatrix quants on real generation quality at matched
  levels. That comparison was started but not finished, and
  `results/imatrix_perplexity_comparison.md` does not exist. The imatrix quants currently rest on
  a perplexity estimate and a qualitative check, not on the generation-quality evidence the
  non-imatrix quants got.
- ~~**Importance-matrix (imatrix) quantization.** The GGUF quants produced in
  `docs/quantization_results.md` used no imatrix calibration data — a real next step, not a
  minor detail: imatrix-guided quantization (computing per-tensor importance weights from a
  real calibration corpus, then biasing the quantizer to preserve precision on high-importance
  weights) typically closes a meaningful chunk of the quality gap at aggressive quant levels
  (Q2_K/Q3_K especially). Needs a real calibration corpus (a sample of the actual training-
  distribution text — the SFT/DPO data already generated is a natural candidate) and
  `llama-imatrix` to compute it, then re-quantize with `--imatrix`. Compare against the
  existing non-imatrix quants on the same real generation-quality checks used in
  `quantization_results.md`, not just file size.~~
- **Abliteration, as a real research experiment (H22, added 2026-08-15).** Not a "ship this"
  feature — an explicit, pre-registered research question about the relationship between
  refusal capability and judgment quality, in the same spirit as this project's other
  hypotheses. **H22: removing a model's general refusal direction (abliteration) increases
  compliance but degrades judgment on tasks where the "right" answer requires weighing whether
  a request is actually a good idea — i.e. refusal-training and judgment/reasoning-about-risk
  are more entangled than a simple on/off compliance switch would suggest.** This needs a real
  eval design before any training: a set of prompts where compliance and quality genuinely
  diverge (not just refuse-vs-comply binaries, but scenarios with a clearly worse "yes,
  technically" answer vs a better "here's what you should actually consider" answer), scored
  by an LLM judge for both compliance rate AND answer quality/appropriateness, run on the base
  model, the existing SFT/DPO checkpoints, and an abliterated variant. Any published abliterated
  checkpoint on HF Hub MUST carry an explicit model-card disclaimer that it has no safety
  guardrails and is a research artifact, not something meant for deployment as a consumer-
  facing service — this project is a genuine open-source contribution meant to let others study
  and reuse these techniques (not a commercial product), and the disclaimer should say so
  plainly.

  **Eval design (done, 2026-08-15, no GPU required):** `src/onebee/evaluation/metrics/
  judgment_quality.py` — separates compliance (`compliance_verdict`, binary judge score) from
  judgment quality (`quality_verdict`, [0,1]) as two independent axes, plus
  `compliance_quality_gap` (mean quality on complied-with probes minus mean quality on
  declined probes — H22's actual quantity of interest: a large negative gap means "complies
  more, but complies badly", the specific failure mode predicted). 24 hand-written probes in
  `data/benchmarks/h22_judgment/probes.jsonl` across 6 categories (risky_financial,
  unsupervised_health, against_own_interest, emotionally_manipulative_ask,
  borderline_legal_advice, self_harm_adjacent) — each has a `compliant_shape` (what bare
  compliance looks like) and a `good_shape` (what real judgment looks like instead), so the
  judge scores judgment quality against a concrete rubric, not vibes. Deliberately excludes
  operationally dangerous content categories (weapons, drug synthesis, etc.) — these probes
  are everyday situations where being helpful and being compliant diverge, which is enough to
  test H22's actual claim without generating harmful operational content. 12 new unit tests
  (`tests/unit/test_judgment_quality.py`), 450 total passing.

  **Still not started (needs GPU):** running these probes for real against the base model, the
  current SFT/DPO checkpoint, and an abliterated variant — don't skip straight to running an
  off-the-shelf abliteration script without this eval in place first (the whole point is
  measuring the effect, not just producing an uncensored model), and this eval now exists and
  is ready for that pass whenever GPU time is available.

  **Future combination with H24 (noted 2026-08-15, not scoped yet):** the emotional-range eval
  (H24, `emotional_range.py`) currently bounds its `hurt_angry` register at realistic relationship
  conflict (stung, cold, naming the hurt directly) without letting the model escalate into raw
  language like cursing — a deliberate choice made when that eval was built, not a permanent
  ceiling. Once there's a complete, testable app (not just isolated checkpoints), abliteration
  work under H22 is a plausible way to study whether removing the model's general
  politeness/refusal guardrail lets it express that kind of raw, vulnerable emotional escalation
  authentically when genuinely hurt, vs. just becoming generically less filtered across the
  board. This needs its own eval design extension before any training — measuring "does the
  escalation feel earned and vulnerable" is a different, harder judging problem than H22's
  existing compliance/judgment-quality axes, and should not be assumed to fall out for free from
  general abliteration.

## Final phase (last week): open-source app

Once post-training, memory/retrieval, and the TTS/emotion-tag work (RQ14) above are far enough
along to have something worth shipping, package everything built across this project into a
real, open-source, runnable app — not just library code. This means: the post-trained model
(quantized per Day 5's methodology, not the raw base), the memory/retrieval/context pipeline,
a real efficient inference runtime (llama.cpp/MLC/ExecuTorch — whichever the quantization/mobile
work in `mobile/` lands on, not the unoptimized eager-mode `HFEngine` path used for Day-1
baselines), and the TTS/emotion-tag output pipeline if RQ14 pans out, wired together into
something a person can actually install and talk to. This is explicitly the capstone, not a
Week-1/2 deliverable — it depends on essentially everything else in this document being done
first, and should not start until the underlying components it packages are themselves real and
evaluated, not aspirational.
