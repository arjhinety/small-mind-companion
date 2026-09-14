# Distillation v1 prompt set (H23)

Prompt-only (system+user) examples extracted from `data/sft/v1/train.jsonl` for on-policy
distillation via `trl.DistillationTrainer` (student generates its own completions during
training, scored against the teacher's token distribution -- no assistant turns needed or
used from the source data).

- Total: 2232 (2008 train / 224 val)
- Source: `data/sft/v1/train.jsonl`. All 2008 train and all 224 val prompts are present in the SFT
  v1 **train** split — the prompt set was carved entirely out of `train.jsonl`, and these 224 val
  records are a slice *of it*, not of `data/sft/v1/val.jsonl` (0 of them appear there). Stated
  explicitly because the reverse reading would be wrong in a way that matters: `data/sft/v1/val.jsonl`
  remains held out from distillation, and prompt-match counts must be taken against `train.jsonl`.
  No new content was introduced, so no separate contamination check was run; the parent corpus is
  contamination-checked clean against `pmb_v0_full` — see `data/sft/v1/DATASHEET.md`.
- A `hash.txt` now exists (it did not originally); see `docs/reproduction.md` for the algorithm and
  `scripts/recompute_hashes.py` to verify it.
- Not human-reviewed.
