"""The paired tests behind ERRATA E36-E39 (``scripts/paired_tests_v0_1.py``).

Known values for the exact McNemar test, and a check that resampling uses only
``Random.random()``, the one generator method whose sequence Python keeps stable across versions.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def stats() -> Any:
    path = ROOT / "scripts" / "paired_tests_v0_1.py"
    spec = importlib.util.spec_from_file_location("smc_paired_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["smc_paired_tests"] = module
    spec.loader.exec_module(module)
    return module


def test_mcnemar_exact_known_values(stats: Any) -> None:
    # Two-sided exact: min(1, 2 * P(Binomial(b + c, 0.5) <= min(b, c))).
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 0) == pytest.approx(0.25)  # E37: 3 lost, 0 gained
    assert stats.mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10)  # E37: 10 gained, 0 lost
    assert stats.mcnemar_exact(5, 5) == 1.0


def test_draw_with_replacement_uses_only_random(stats: Any) -> None:
    population = list(range(37))
    drawn = stats.draw_with_replacement(random.Random(1337), population, 500)
    reference = random.Random(1337)
    expected = [population[math.floor(reference.random() * 37)] for _ in range(500)]
    assert drawn == expected
