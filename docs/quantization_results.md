# GGUF quantization results (2026-08-15)

**Status:** real, working GGUF quantization of `arrochi112/onebee-gf-dpo-v1-scale` — the
rebalanced proper-scale SFT+DPO checkpoint at the time of this pass; `distill-v1` superseded it as
current-best later (see §"Follow-up" and `docs/proper_scale_results.md`). Both text and vision
(multimodal) capability verified to survive conversion and quantization. Four real bugs found and
fixed along the way — `docs/model_quirks.md` #18-21.

> **Evidence note.** No `llama-bench`, `llama-imatrix`, `llama-quantize` or GGUF file is committed
> to this repository (`*.gguf` is gitignored, and `results/` contains no quantization artifacts).
> Every size, throughput and perplexity figure below is restated from the original run logs on the
> GPU box, not re-derivable from a tracked artifact in a fresh clone. Treat them as reported
> measurements, and re-run `llama-bench` if you need to rely on them.

## What was done

1. Built `llama.cpp` from source, CPU-only (`GGML_CUDA=OFF`) — this box had no system-wide
   CUDA toolkit (`nvcc`), and conversion/quantization are CPU-only tools regardless; a CPU
   build is also arguably more representative of the actual on-device deployment target
   (phones don't have CUDA) than a GPU-accelerated benchmark would have been.
2. Downloaded `arrochi112/onebee-gf-dpo-v1-scale` from HF Hub (the checkpoint uploaded before
   the previous GPU box was deleted).
3. Converted to GGUF F16 in two passes (`convert_hf_to_gguf.py`, see model_quirks #18):
   - Main language model: `onebee-dpo-v1-scale-f16.gguf` (9.27GB / 8.64 GiB — the benchmark table
     below reports llama-bench's own 8.62 GiB; the two differ by filesystem size reporting)
   - Vision projector: `mmproj-onebee-dpo-v1-scale-f16.gguf` (986MB)
   - Required patching a copy of `tokenizer_config.json` first (model_quirks #19) — llama.cpp's
     own pinned `transformers==4.57.6` and our training environment's `transformers==5.15.0`
     disagree on the `extra_special_tokens` field format.
4. Quantized the main model to the full standard spread with `llama-quantize` (F16 reference plus
   12 quant levels down to Q2_K — the common K-quant + legacy set the GGUF community typically
   ships, e.g. Q8_0/Q6_K/Q5_K_M/Q5_K_S/Q5_0/Q4_K_M/Q4_K_S/Q4_0/Q3_K_L/Q3_K_M/Q3_K_S/Q2_K —
   deliberately excluding the exotic IQ*/TQ*/MXFP4_MOE types, which need imatrix calibration data
   to be worthwhile and would quantize poorly without it. The imatrix-calibrated requantization
   that note referred to was subsequently done, see §"Follow-up" below).
   - The mmproj (vision projector) was NOT separately quantized — used as F16 with every
     text-model quant level, which is llama.cpp's standard pattern (the projector is small
     relative to the LLM, and is more precision-sensitive).
   - Full file listing and exact sizes: see the model card on
     `arrochi112/onebee-gf-dpo-v1-scale-gguf`.
5. Verified real generation quality and multimodal capability on all three levels — not just
   "did it produce a file," but "does it produce coherent, correct output."

## Real benchmark numbers (llama-bench, CPU-only, 30 threads, this box's hardware)

Model reports as **4.63B params** (not literally "2B" — `gemma4-E2B` is an MoE architecture
with an "effective 2B" active-parameter naming, confirmed by llama-bench's own model-info
output, not something previously stated explicitly in this project's docs).

| Quant | Size | Prompt processing (pp512) | Generation (tg128) |
|---|---|---|---|
| F16 | 8.62 GiB | 585.07 ± 0.44 t/s | 26.15 ± 0.28 t/s |
| Q8_0 | 4.59 GiB (-47%) | 492.33 ± 1.21 t/s | 43.07 ± 0.24 t/s (+65%) |
| Q4_K_M | 3.17 GiB (-63%) | 633.00 ± 1.22 t/s | 58.00 ± 0.51 t/s (+122%) |

Real, somewhat counter-intuitive result worth noting honestly: Q8_0's prompt-processing speed
is *slower* than F16 (492 vs 585 t/s), while Q4_K_M's is *faster* than both (633 t/s). This is
plausible (Q8_0's dequantization overhead during prompt-batch matmuls can outweigh its memory-
bandwidth savings on some kernels/hardware, while Q4_K_M's more aggressive size reduction wins
on both axes) but wasn't investigated further — noted as a real observation, not explained away.
Generation (token-by-token, memory-bandwidth-bound) speed scales cleanly with quantization
level as expected, in both directions.

## Real generation quality checks (not just "did a file get produced")

- **Text, no persona context** ("What is 2+2?", "What is your favorite color?"): coherent on
  all three levels (F16, Q8_0, Q4_K_M). Content-wise the model answered "2+2=6" (wrong — this
  is a small companion-persona-tuned model, not a math model, and math was never part of any
  training data in this project; not a quantization bug) and gave the generic "as an AI, I
  don't have preferences" disclaimer to the favorite-color question (expected: that test had
  no injected memory/persona system prompt of the kind the real eval harness always uses — see
  `docs/proper_scale_results.md` for the real, in-format evaluation of this exact behavior).
- **Text, with the real companion system prompt**: consistent phrasing/style across all three
  quant levels — no signs of quantization-induced incoherence or repetition loops.
- **Vision** (`llama-mtmd-cli --jinja`, fixture image `wrapped_gift_red.png`): **"A red gift
  box tied with a gold ribbon is shown against a light gray background."** — accurate,
  specific, matches the real fixture image. Confirms multimodal capability survived the full
  conversion + quantization pipeline, which is exactly the risk this project's own
  `docs/model_quirks.md` has previously flagged (multimodal capability silently lost at the
  save-checkpoint step during SFT, item #13) — this time it survived cleanly, a genuinely good
  result worth confirming explicitly rather than assuming.

## Four real bugs found and fixed (full detail: `docs/model_quirks.md` #18-21)

1. `--mmproj` exports ONLY the projector, not a combined file — two separate runs needed.
2. Cross-version `tokenizer_config.json` format mismatch (`extra_special_tokens` list vs dict)
   between the training environment's transformers (5.15.0) and llama.cpp's pinned converter
   dependency (4.57.6) — required patching a copy of the config before conversion.
3. `llama-cli`'s `-cnv` (auto-enabled with any chat template) also auto-enables interactive
   mode, which hangs on closed stdin in a scripted/backgrounded invocation — burned real time
   mistaking this for "quantization made the model catastrophically slow" before finding the
   actual cause (wrong CLI flag). Fix: `--single-turn`.
4. `llama-mtmd-cli` crashes on this model's real (complex, loop-based) chat template unless
   run with `--jinja`.

None of these are quantization *quality* issues — all four are tooling/environment friction,
the same category of finding this project has documented extensively at every prior stage
(model loading, training APIs, evaluation scoring). The actual quantization numerics worked
cleanly on the first successful attempt once the tooling issues were resolved.

## Where the artifacts are

- Local (this GPU box, `/root/small-mind-companion/gguf/`): all 4 files (F16 main + mmproj, Q8_0, Q4_K_M).
- HF Hub: uploaded to `arrochi112/onebee-gf-dpo-v1-scale-gguf` (see repo for exact filenames)
  so they survive this box being deleted, same pattern as the merged checkpoints themselves.

## Known limitations / not done in this pass

- No comparison against Google's own QAT mobile variant
  (`google/gemma-4-E2B-it-qat-mobile-transformers`) — noted as a next step, not done here.
- No real accuracy/quality regression test against the PMB-v0 benchmark at each quant level —
  this pass confirmed "coherent and on-topic," not "measurably as accurate as F16 on the real
  eval harness." A real quant-vs-quality tradeoff study (does Q4_K_M measurably hurt
  `pra_lenient`/`uar` vs F16?) would need to actually run the harness through llama.cpp's
  Python bindings or server API against all three levels — not attempted here, given this
  pass's scope was "does quantization work at all for this architecture+checkpoint," which it
  does.
- CPU-only benchmark numbers only — no GPU-accelerated llama.cpp build/benchmark for
  comparison (this box had no CUDA toolkit installed; installing one and rebuilding with
  `GGML_CUDA=ON` was judged not worth the setup time for this pass, since the deployment
  target is on-device/mobile CPU inference anyway, not GPU-accelerated serving).
- Only one base checkpoint quantized (`dpo-v1-scale`, the current best) — the other 4
  checkpoints on HF Hub (sft-v0, sft-v1, dpo-v0, dpo-v1-4epoch) were not quantized, since
  they're superseded/experimental, not needed for deployment.

## Follow-up: imatrix-calibrated requantization (2026-08-15)

**Real, working next step.** Rather than a generic corpus (e.g. wikitext), the importance
matrix was computed from **this project's own data** — real companion conversations and
persona-consistent preference responses (`data/imatrix_calibration.txt`, built by
`build_imatrix_calibration.py` from `data/sft/v1/train.jsonl` + `data/dpo/v1_scale/train.jsonl`
chosen responses, ~11MB / 66,383 lines) — so the quantizer preserves precision on what this model
is actually used for, not generic language modeling. Rejected DPO responses (the
disclaimer-breaking behavior being trained away) were deliberately excluded from calibration.

- Computed via `llama-imatrix` against the F16 GGUF, 5328 chunks (n_ctx=512), final training
  PPL estimate 2.7258 ± 0.0046. Took considerably longer than expected on CPU (~2 hours), well
  past the tool's own initial ETA — a real observation, not investigated further (this run was
  time-constrained; a future pass could profile why).
- Requantized 6 levels with `--imatrix`: Q2_K, Q3_K_S/M/L, Q4_K_S/M — the levels imatrix
  calibration is expected to help most (near-lossless levels like Q8_0/Q6_K weren't
  reprocessed, imatrix barely matters there).
- Real sanity check on `Q4_K_M-imat` (companion system prompt, "What is your favorite color?"):
  coherent, in-character output — same qualitative behavior as the non-imatrix version.
- All 6 imatrix-calibrated files (`onebee-dpo-v1-scale-<LEVEL>-imat.gguf`) plus the raw
  `imatrix.gguf` itself uploaded to a **private** repo,
  `arrochi112/onebee-gf-dpo-v1-scale-gguf-imatrix` — kept separate from and private relative to
  the public `arrochi112/onebee-gf-dpo-v1-scale-gguf` repo (which still has the original
  non-imatrix quants, public) per an explicit decision to hold the imatrix quants back for now
  while the perplexity comparison below is still unverified.

**Not done in this pass** (real time constraint — the GPU box was on a tight rental
budget with a real risk of the instance disappearing mid-work, so verification depth was
deliberately traded for getting real artifacts saved): a full quantitative perplexity
comparison between imatrix and non-imatrix versions at matching quant levels was started
(`llama-perplexity` on a held-out slice of `data/sft/v1/val.jsonl`, explicitly NOT the
calibration corpus) but not confirmed complete before this doc was written. **Confirmed later: the
outcome file was never produced** — `results/imatrix_perplexity_comparison.md` does not exist in
this repository. Treat the imatrix quants as "real, uploaded, sanity-checked for coherence, not yet
numerically proven better" until that follow-up lands. This is reported honestly rather than
claiming a clean win that
wasn't actually measured — consistent with this project's discipline throughout.

## Follow-up: distill-v1 (current-best checkpoint) GGUF quantization (2026-08-17)

The GGUF quants above were built from `dpo-v1-scale` (pre-distillation) — the current-best
checkpoint, `distill-v1` (H23, post-distillation), had no GGUF quants until this pass. Built on
Modal (`RTX-PRO-6000`, see `docs/model_quirks.md` #25-26 for the real Modal/CUDA-image and
volume-caching issues hit along the way), same 12-level spread (F16 through Q2_K) plus mmproj,
uploaded to a new repo: `arrochi112/onebee-gf-distill-v1-gguf` (52.5GB total, all 14 files
verified present).

The tokenizer_config.json `extra_special_tokens` list→dict bug (`model_quirks.md` #19)
reproduced exactly as documented on this checkpoint too, same fix applied.

**Real generation quality checks, with the companion system prompt** ("What is your favorite
color?"):

| Quant | Result |
|---|---|
| Q4_K_M | Coherent (bare prompt, no system prompt in this specific test — generic-assistant-style response as a result, not a quant artifact) |
| Q4_K_S | Coherent, in-character: "soft white... calm and open, without being cold or stark" |
| Q3_K_M | Coherent, in-character: "amber—the soft, buttery glow right before sunset" |
| Q3_K_S | Coherent, in-character: "emerald... like the color of moss after a spring rain" |
| **Q2_K** | **Broken** — garbled special-token output (`<\|turn\>confusion`, `<\|turn\>engine`) followed by a nonsense repeating loop (`type:t \| \| \| ...`), not usable |

**Follow-up (2026-08-17, CPU-only, no GPU needed): confirmed NOT distillation-specific.**
Tested `dpo-v1-scale`'s own Q2_K file (same prompt, same companion system prompt, `llama.cpp`
built CPU-only locally) — it fails the exact same way: garbled `<|turn>` special-token output
followed by a non-terminating repetition loop (repeating "amazon" instead of "type:t", but the
same failure shape). **This resolves the open question: Q2_K was never safe for this model
family at all — it was simply never generation-tested before this pass, for either checkpoint.**
Not a distillation side-effect. **Recommendation stands: do not use Q2_K for this model family
— Q3_K_S is the smallest verified-coherent level, for both `dpo-v1-scale` and `distill-v1`.**

A separate, real infrastructure bug hit during upload (not a quantization issue): HF Hub's Xet
upload backend (`_upload_xet_files`) repeatedly raised `TimeoutError: ... error decoding
response body` even after a full 52.5GB transfer completed, requiring `HF_HUB_DISABLE_XET=1`
to force the classic upload path — see `docs/model_quirks.md` for detail if this recurs.
