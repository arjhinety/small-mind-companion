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

## Generating a `hash.txt`

`hash.txt` files pin dataset integrity, but the two algorithms in use are different and
undocumented in the files themselves:

- **SFT / DPO datasets**: `sha256` over the concatenated contents of all `*.jsonl` in sorted order
  (`generate_sft_data.py`, `generate_dpo_data.py`).
- **Benchmark corpora**: sorted directory walk, hashing each file's forward-slash relative path
  followed by its bytes, skipping `hash.txt` itself (`scripts/build_pmb.py`).

Both assume **LF line endings**. This repo sets `core.autocrlf=true`, so a working tree on Windows
is CRLF and a naive `sha256sum` will not match — normalise `\r\n` → `\n` before hashing.

`data/distill/v1/` has no `hash.txt`.

Every command above should be run from a clean clone, in order, to earn the reproducibility
claim. If a step breaks, that is a bug in this file or the code — file it.
