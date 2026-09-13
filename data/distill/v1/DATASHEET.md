# Distillation v1 prompt set (H23)

Prompt-only (system+user) examples extracted from `data/sft/v1/train.jsonl` for on-policy
distillation via `trl.DistillationTrainer` (student generates its own completions during
training, scored against the teacher's token distribution -- no assistant turns needed or
used from the source data).

- Total: 2232 (2008 train / 224 val)
- Source: `data/sft/v1/train.jsonl`. This is a strict subset (2008/2232 train and 224/248 val
  prompts match exactly) with no new content, so no separate contamination check was run; the
  parent corpus is contamination-checked clean against `pmb_v0_full` — see
  `data/sft/v1/DATASHEET.md`.
- **No `hash.txt` is committed for this directory** (unlike the SFT/DPO datasets), so its
  contents are not integrity-pinned. Generate one before relying on the subset-equality claim.
