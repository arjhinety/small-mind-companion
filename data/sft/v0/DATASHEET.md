# SFT v0 dataset

Memory-aware conversational SFT data: [persona card + retrieved memories + recent turns + user
turn] -> response, generated via the real retrieval pipeline (HybridRetriever, k=8) against
populated memory stores built from 4 personas disjoint from the PMB-v0 eval set
(`data/benchmarks/sft_personas_v0/`, `data/stores/sft_personas_v0/`), with target responses from
a live teacher model (gpt-5.6-luna).

- Total examples: 225 (202 train / 23 val)
- By kind: {"irrelevant_retrieval": 1, "memory_relevant": 223, "abstention": 1}
- **Contamination-checked clean** against the PMB-v0 evaluation benchmark:
  `python scripts/check_contamination.py --train-glob "data/sft/v0/train.jsonl"
  --eval-glob "data/benchmarks/pmb_v0_full/probes.jsonl"` → `No contamination found.`
- Personas are disjoint from the PMB-v0 eval set by construction: a separate generation run via
  `scripts/build_pmb.py` written to a separate output directory. The exact `--seed` used for the
  training-persona corpus is **not recorded** (the script default is 1337) — earlier revisions of
  this file cited "seed 9999", which appears nowhere in the code.
- Persona NAMES may coincidentally overlap with the eval set (shared name pool); IDs and facts
  differ, so this is not data leakage.
- Not human-reviewed.
