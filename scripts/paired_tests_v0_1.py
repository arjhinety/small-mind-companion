"""Paired significance tests over the committed v0.1 per-probe responses.

Recomputes, from the saved ``raw.jsonl`` files only, two families of paired comparisons on
``pra_lenient`` (judge-scored recall on answerable probes):

* ``d_vs_e`` -- System D (raw model + memory, k=8) against System E (memory-aware SFT + memory,
  k=8) on the 608 answerable PMB probes. This is the comparison ``docs/day4_sft_results.md``
  reported as H5's evidence.
* ``k*_vs_k*`` -- adjacent steps of the k-sweep (k = 0, 2, 4, 8, 16) on its 120-probe stratified
  subsample (105 answerable). This is the curve ``docs/day3_memory_results.md`` reported as H10's
  evidence.

Both families compare two systems on the *same* probes, so the tests are paired:

* **Exact McNemar test** (two-sided). Only the probes where the two systems disagree carry
  information; under the null each disagreement is equally likely to go either way, so the count
  in one direction is Binomial(disagreements, 0.5). p = min(1, 2 * P(X <= min(b, c))).
* **Paired bootstrap 95% CI** on the difference in recall (second system minus first, in
  percentage points): resample probes with replacement, keeping each probe's pair together.
  Percentile interval, nearest-rank: the values at sorted positions floor(0.025 * N) and
  ceil(0.975 * N) - 1 of N resamples.
* **Holm-adjusted p-values** across the four k-sweep steps, since they are read together as one
  curve. ``d_vs_e`` is a single comparison and is not adjusted.

Standard library only, on purpose: ``scripts/validate.py`` imports this module to recompute every
quoted p-value and interval, and the validator does not depend on numpy. ``random.Random`` with a
fixed seed is also stable across Python versions for ``choices``, which numpy's Generator streams
do not promise.

Usage::

    uv run python scripts/paired_tests_v0_1.py          # (re)write the output JSON
    uv run python scripts/paired_tests_v0_1.py --check  # exit 1 if the committed JSON is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "v0.1" / "paired_tests" / "paired_tests.json"

SEED = 1337
N_RESAMPLES = 10_000
CI_LEVEL = 0.95
METRIC = "lenient_correct"

# (id, first run, second run). The difference is always second minus first.
D_VS_E = ("d_vs_e", "results/v0.1/D_memory/raw.jsonl", "results/v0.1/E_sft_memory/raw.jsonl")
KSWEEP_KS = (0, 2, 4, 8, 16)
KSWEEP = tuple(
    (f"k{a}_vs_k{b}", f"results/v0.1/ksweep/k{a}/raw.jsonl", f"results/v0.1/ksweep/k{b}/raw.jsonl")
    for a, b in zip(KSWEEP_KS, KSWEEP_KS[1:], strict=False)
)


def sha256_lf_normalised(path: Path) -> str:
    """Same hash method as the Study 001 freeze manifest."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_answerable(rel: str) -> dict[str, int]:
    """probe_id -> 1/0 lenient correctness, over answerable probes only."""
    outcomes: dict[str, int] = {}
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probe = row["probe"]
        if not probe["answerable"]:
            continue
        probe_id = probe["probe_id"]
        if probe_id in outcomes:
            raise ValueError(f"{rel}: duplicate probe_id {probe_id}")
        outcomes[probe_id] = 1 if row.get(METRIC) else 0
    return outcomes


def mcnemar_exact(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = min(first_only, second_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci_pp(
    first: list[int], second: list[int], seed: int = SEED, n_resamples: int = N_RESAMPLES
) -> tuple[float, float]:
    """Percentile CI, in percentage points, on mean(second - first) over paired probes."""
    if len(first) != len(second) or not first:
        raise ValueError("paired bootstrap needs two equal-length, non-empty lists")
    diffs = [b - a for a, b in zip(first, second, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n * 100 for _ in range(n_resamples))
    alpha = 1.0 - CI_LEVEL
    lo = means[math.floor(alpha / 2 * n_resamples)]
    hi = means[math.ceil((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


PROBES = "data/benchmarks/pmb_v0_full/probes.jsonl"


def persona_of_probe() -> dict[str, str]:
    """probe_id -> persona_id, from the PMB probe file."""
    mapping: dict[str, str] = {}
    for line in (ROOT / PROBES).read_text(encoding="utf-8").splitlines():
        if line.strip():
            probe = json.loads(line)
            mapping[probe["probe_id"]] = probe["persona_id"]
    return mapping


def persona_bootstrap_ci_pp(
    first: list[int],
    second: list[int],
    groups: list[str],
    seed: int = SEED,
    n_resamples: int = N_RESAMPLES,
) -> tuple[float, float]:
    """Percentile CI, in percentage points, on mean(second - first), resampling whole personas.

    Each resample draws as many personas as there are, with replacement, and pools every paired
    probe of each drawn persona; the statistic is the pooled mean difference. Same interval
    convention as paired_bootstrap_ci_pp. With few personas the interval is coarse: it can take
    only as many values as there are distinct persona multisets.
    """
    if not (len(first) == len(second) == len(groups)) or not first:
        raise ValueError("persona bootstrap needs equal-length, non-empty lists")
    by_group: dict[str, tuple[int, int]] = {}
    for a, b, g in zip(first, second, groups, strict=True):
        total, count = by_group.get(g, (0, 0))
        by_group[g] = (total + (b - a), count + 1)
    names = sorted(by_group)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choices(names, k=len(names))
        total = sum(by_group[g][0] for g in drawn)
        count = sum(by_group[g][1] for g in drawn)
        means.append(total / count * 100)
    means.sort()
    alpha = 1.0 - CI_LEVEL
    lo = means[math.floor(alpha / 2 * n_resamples)]
    hi = means[math.ceil((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def compare(comparison_id: str, first_rel: str, second_rel: str) -> dict[str, Any]:
    first = load_answerable(first_rel)
    second = load_answerable(second_rel)
    if set(first) != set(second):
        raise ValueError(
            f"{comparison_id}: {first_rel} and {second_rel} cover different answerable probes"
        )
    ids = sorted(first)
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    first_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    second_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    lo, hi = paired_bootstrap_ci_pp(a, b)
    personas = persona_of_probe()
    groups = [personas[i] for i in ids]
    plo, phi = persona_bootstrap_ci_pp(a, b, groups)
    n = len(ids)
    return {
        "first": first_rel,
        "second": second_rel,
        "n_answerable_probes": n,
        "first_correct": sum(a),
        "second_correct": sum(b),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "diff_pp": (sum(b) - sum(a)) / n * 100,
        "mcnemar_exact_p": mcnemar_exact(first_only, second_only),
        "bootstrap_ci_pp": [lo, hi],
        "n_personas": len(set(groups)),
        "persona_bootstrap_ci_pp": [plo, phi],
    }


def build() -> dict[str, Any]:
    comparisons: dict[str, dict[str, Any]] = {}
    for spec in (D_VS_E, *KSWEEP):
        comparisons[spec[0]] = compare(*spec)
    ksweep_ids = [spec[0] for spec in KSWEEP]
    adjusted = holm_adjust({cid: comparisons[cid]["mcnemar_exact_p"] for cid in ksweep_ids})
    for cid in ksweep_ids:
        comparisons[cid]["holm_adjusted_p"] = adjusted[cid]
    inputs = sorted({rel for spec in (D_VS_E, *KSWEEP) for rel in spec[1:]})
    return {
        "artifact_kind": "paired-significance-tests",
        "generated_by": "scripts/paired_tests_v0_1.py",
        "metric": f"pra_lenient (per-probe `{METRIC}`, answerable probes only)",
        "direction": "every difference is second minus first",
        "methods": {
            "test": "exact McNemar, two-sided: min(1, 2 * P(Binomial(b + c, 0.5) <= min(b, c)))",
            "interval": (
                f"paired bootstrap over probes, {N_RESAMPLES} resamples, seed {SEED}, "
                f"{int(CI_LEVEL * 100)}% percentile interval (nearest rank), Python random.Random"
            ),
            "persona_interval": (
                f"bootstrap over whole personas (persona_id from {PROBES}), {N_RESAMPLES} "
                f"resamples, seed {SEED}, same percentile convention; pooled mean difference"
            ),
            "multiplicity": "Holm across the four k-sweep steps; d_vs_e is a single comparison",
        },
        "caveats": [
            "Every input is a single seed and a single run.",
            "The k-sweep covers a 120-probe stratified subsample (15 per category), not the "
            "full 688.",
            "The k-sweep raw.jsonl files are checked against their metrics.json by "
            "scripts/validate.py but are not pinned by the Study 001 freeze manifest.",
        ],
        "inputs": {rel: sha256_lf_normalised(ROOT / rel) for rel in inputs},
        "comparisons": comparisons,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed output is stale"
    )
    args = parser.parse_args(argv)
    fresh = serialise(build())
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
        if current != fresh:
            print(f"{OUTPUT.relative_to(ROOT)} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} is fresh")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fresh, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
