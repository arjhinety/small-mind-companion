.PHONY: install install-gpu install-dev lint format typecheck test test-unit baseline eval freeze-check validate result-registry

install:
	uv sync

install-gpu:
	uv sync --extra gpu

# ruff, black, mypy and pytest live in the `dev` extra; the bare sync above does not install them.
install-dev:
	uv sync --extra dev

# Study 001 is frozen. Fails if any hash-pinned artifact changed — see docs/STUDIES.md.
freeze-check:
	uv run python scripts/freeze_study_001.py --check

# The whole audit: the freeze, every dataset hash, every metric recomputed from its own raw.jsonl,
# the claims matrix (registry/claims.jsonl), registry counts against the data files,
# cross-document consistency, and the 13-gram contamination check. CPU-only and offline. This is
# the same command the CI workflow runs as its last step — see docs/AUDIT.md.
validate:
	uv run python scripts/validate.py

# Regenerate the derived index of evaluated runs. `make validate` fails if it is stale.
result-registry:
	uv run python scripts/build_result_registry.py

lint:
	uv run ruff check
	uv run black --check .

format:
	uv run ruff check --fix
	uv run black .

typecheck:
	uv run mypy src

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

baseline:
	echo "run scripts/model_bakeoff.py — requires GPU, not run in CI"

eval:
	echo "placeholder for eval targets"

# `figures` and `paper` targets were removed: scripts/make_figures.py and paper/build.sh do not
# exist in this repo, so both targets always failed. See docs/reproduction.md.
