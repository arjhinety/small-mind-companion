# Reproduction

## Install

```bash
uv sync                # base install: CPU-only, sufficient for tests and CI
uv sync --extra gpu     # + torch/transformers/vllm/trl/peft/etc. for training and inference
```

## Run the test suite

```bash
pytest
```

## Build the PMB benchmark

```bash
python scripts/build_pmb.py --out-dir data/benchmarks/pmb_v0 \
  --n-personas 8 --sessions-per-persona 6 --turns-per-session 14 --facts-per-persona 40
```

The repo ships a small fixture-scale run (2 personas) under `data/benchmarks/pmb_v0/` for
offline smoke-testing; the command above regenerates it at full v0 scale. A live teacher
endpoint (`--teacher`) is required for non-fixture generation — not yet wired up (see
`scripts/build_pmb.py --help`).

## Run the evaluation harness

```python
from onebee.evaluation.harness import run_harness, save_harness_result
# see tests/integration/test_smoke.py for a minimal end-to-end example
```

## Populate a memory store

```python
from onebee.memory.store import MemoryStore
store = MemoryStore("data/stores/example.db")
```

## Check for train/eval contamination

```bash
python scripts/check_contamination.py \
  --train-glob "data/**/train*.jsonl" --eval-glob "data/benchmarks/**/probes*.jsonl"
```

## Train the SFT adapter

Requires the `gpu` extra and a GPU workstation — not run in CI.

```bash
python -m onebee.training.sft --config configs/training/sft.yaml
```

## Regenerate figures

Not available. `scripts/make_figures.py` and the `make figures` target do not exist; the figures in
`results/figures/` and `assets/` were produced ad hoc outside the repo. Either implement the script
or drop this step — see `README.md`'s reproducibility list, which does not claim it.

## Verifying and regenerating `hash.txt`

Verify every dataset hash in one step:

```bash
uv run python scripts/recompute_hashes.py          # report drift, change nothing
uv run python scripts/recompute_hashes.py --write  # rewrite the hash.txt files
```

Two algorithms are in use, and they were undocumented in the files themselves:

- **SFT / DPO / distillation datasets**: `sha256` over the concatenated contents of every `*.jsonl`
  in sorted order (`generate_sft_data.py`, `generate_dpo_data.py`).
- **Benchmark corpora**: sorted directory walk, hashing each file's forward-slash relative path
  followed by its bytes, skipping `hash.txt` itself (`scripts/build_pmb.py`).

Both hash **LF-normalised** bytes. This repo sets `core.autocrlf=true` and has no `.gitattributes`,
so a Windows working tree is CRLF while the committed hashes were computed on LF — a naive
`sha256sum` will not match, and the hash would otherwise depend on the checkout platform rather
than on the content.

**History.** The four benchmark `hash.txt` files committed with Study 001 matched their corpora
under none of thirteen tested algorithms and may never have matched; `data/distill/v1/` had no hash
at all. All nine were recomputed on 2026-09-13 with the algorithm above and now verify. The
benchmark corpora were also edited (datasheets corrected), so the old values were stale regardless.
See `reports/ERRATA.md` E29.

Every command above should be run from a clean clone, in order, to earn the reproducibility
claim. If a step breaks, that is a bug in this file or the code — file it.
