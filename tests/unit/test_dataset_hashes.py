"""Every dataset ``hash.txt`` must match its corpus.

This is guardrail G9c in executable form: a hash that has never been verified from a clean
checkout is not a pin. Study 001 shipped four benchmark hashes that matched under none of thirteen
tested algorithms, and one dataset with no hash at all — both went unnoticed because nothing ever
recomputed them (``reports/ERRATA.md`` E29).

Runs on CPU with no model download or network access.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "recompute_hashes.py"


@pytest.mark.parametrize(
    "rel",
    [
        "data/sft/v0",
        "data/sft/v1",
        "data/dpo/v0",
        "data/dpo/v1_scale",
        "data/distill/v1",
        "data/benchmarks/pmb_v0_full",
        "data/benchmarks/sft_personas_v0",
        "data/benchmarks/sft_personas_v1",
        "data/benchmarks/pmb_v0",
    ],
)
def test_corpus_has_a_hash(rel: str) -> None:
    assert (ROOT / rel / "hash.txt").is_file(), (
        f"{rel} has no hash.txt. Study 001 shipped data/distill/v1/ without one; add it with "
        "`uv run python scripts/recompute_hashes.py --write`."
    )


def test_all_corpora_match_their_hashes() -> None:
    """The single check that would have caught E29."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "a dataset hash does not match its corpus:\n"
        f"{result.stdout}\n{result.stderr}\n"
        "If the data legitimately changed, that is a Study 002 result — see docs/STUDIES.md. "
        "If only the hash is wrong, regenerate it and say so in reports/ERRATA.md."
    )
