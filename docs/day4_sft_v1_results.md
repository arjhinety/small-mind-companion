# SFT v1: proper-scale training run (follow-up to Day 4's v0 proof-of-concept)

**Status:** real training run, 10x the data scale of the v0 run documented in
`docs/day4_sft_results.md`. Not yet re-evaluated against the A/B/D/E harness (that's task #46,
after DPO v1 also completes, so both checkpoints get evaluated together).

## What changed vs v0

| | v0 (Day 4) | v1 (this run) |
|---|---|---|
| Personas | 4 | 40 |
| SFT examples | 225 (202 train / 23 val) | 2242 generated (2017 train / 225 val) — **superseded**; the committed `data/sft/v1/` is the rebalanced 2480 (2232 train / 248 val) |
| `per_device_train_batch_size` | 4 | 8 (doc-recommended) |
| `gradient_accumulation_steps` | 2 | 4 (doc-recommended) |
| Effective batch size | 8 | 32 |
| Training steps | 52 | 128 |
| Wall-clock | ~64s | ~623s (~10.4 min) |

Same base model (`gemma4-e2b`), same LoRA config (r=16, alpha=32, dropout=0.05,
all-linear target modules), same 2 epochs, same learning rate (1e-4, cosine schedule).
Config: `configs/training/sft_v1.yaml`. Data: `data/sft/v1/` (contamination-checked clean
against `pmb_v0_full`, see `data/sft/v1/DATASHEET.md`). Data generation: `docs` — persona set
`data/benchmarks/sft_personas_v1/` (40 personas, 3437 probes), memory stores
`data/stores/sft_personas_v1/` (2463 accepted claims from 2277 real user turns via
`OpenAITeacherExtractor`).

## Training curve

| Step (of 128) | loss | token accuracy | epoch |
|---|---|---|---|
| ~10 | 4.229 | 51.1% | 0.16 |
| ~20 | 1.881 | 62.6% | 0.32 |
| ~40 | 1.093 | 74.9% | 0.63 |
| ~60 | 0.918 | 77.8% | 0.95 |
| ~90 | 0.852 | 78.8% | 1.41 |
| ~120 | 0.789 | 80.0% | 1.73 |
| final (128) | 0.797 (train_loss avg 1.258) | 80.1% | 2.0 |

Smooth, well-behaved convergence — no signs of the near-perfect-fit overfitting seen in the
DPO v1 (4-epoch) run's training curve. Final token accuracy (80.1%) is a real improvement over
v0's final 77% on a dataset 10x larger, consistent with SFT continuing to improve with more
data rather than saturating at v0's scale.

## Checkpoint

`outputs/sft/v1/merged` (LoRA adapter merged into the base model). `processor_config.json`
present alongside the model weights — multimodal capability preserved (the fix from
`docs/model_quirks.md` #13 applies cleanly here too, no regression).

## Not yet done

- Full A/B/D/E-style re-evaluation on the 688-probe PMB harness with this checkpoint — deferred
  to task #46, run together with the properly-scaled DPO checkpoint so both get evaluated in
  one pass.
- Upload to HF Hub — same deferred status as v0.
