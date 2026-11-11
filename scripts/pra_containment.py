"""Why ``pra_strict`` is ~0, and how a containment rule compares with the judge (ERRATA E41).

EXPLORATORY. The containment rule below was written after reading Study 001's responses, so its
agreement with the judge on those responses is descriptive, not validation. Any Study 002 metric
must be fixed in a pre-registration before the runs it scores.

For every full-harness run in ``results/registry.jsonl`` (``run_kind == "pmb_full_harness"``):

* **exact** -- responses whose whole normalised text equals the gold answer. This is
  ``exact_match`` in ``rule.py``, the rule behind ``pra_strict``. It is ~0 because responses are
  sentences, and it would stay ~0 with any ``acceptable_alternatives``, since those are compared the
  same way.
* **containment vs judge** -- on answerable probes, ``contains_answer`` in ``rule.py`` (the
  normalised gold answer occurs in the normalised response as a whole word or phrase,
  ``(?<!\\w)gold(?!\\w)``), cross-tabulated with the stored judge verdict ``lenient_correct``, with
  Cohen's kappa per run and counts per category. The same rule is pre-registered for Study 002 as
  ``pra_contain`` (``docs/preregistrations/study002-answer-containment.md``).

``normalize_text``, ``exact_match`` and ``contains_answer`` are ``rule.py``'s own functions, loaded
by path (it imports only ``re`` and ``string``), so this script and ``scripts/validate.py`` need no
third-party package.
Read-only: no ``raw.jsonl`` or ``metrics.json`` is modified.

Usage::

    uv run python scripts/pra_containment.py           # (re)write the output JSON
    uv run python scripts/pra_containment.py --check   # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "audit" / "pra_containment" / "pra_containment.json"
RULE_PATH = "src/onebee/evaluation/graders/rule.py"
REGISTRY = ROOT / "results" / "registry.jsonl"
CELLS = ("both", "contain_only", "judge_only", "neither")


def load_rule() -> types.ModuleType:
    """Import the working-tree rule.py by path."""
    spec = importlib.util.spec_from_file_location("onebee_rule_by_path", ROOT / RULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {RULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_harness_runs() -> list[str]:
    rows = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines()]
    return [r["run_path"] for r in rows if r.get("run_kind") == "pmb_full_harness"]


def kappa(cells: Counter[str]) -> float | None:
    n = sum(cells[c] for c in CELLS)
    observed = (cells["both"] + cells["neither"]) / n
    p_contain = (cells["both"] + cells["contain_only"]) / n
    p_judge = (cells["both"] + cells["judge_only"]) / n
    expected = p_contain * p_judge + (1 - p_contain) * (1 - p_judge)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def score_run(rule: types.ModuleType, run: str) -> dict[str, Any]:
    path = ROOT / run / "raw.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    with_gold = [r for r in rows if r["probe"]["gold_answer"]]
    cells: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    for row in rows:
        probe = row["probe"]
        if not probe["answerable"]:
            continue
        contained = rule.contains_answer(row["response"], probe["gold_answer"])
        judged = bool(row["lenient_correct"])
        cell = CELLS[0 if contained and judged else 1 if contained else 2 if judged else 3]
        cells[cell] += 1
        by_category.setdefault(probe["category"], Counter())[cell] += 1
    return {
        "responses": len(rows),
        "responses_with_gold": len(with_gold),
        "exact_whole_response": sum(
            rule.exact_match(r["response"], r["probe"]["gold_answer"]) for r in with_gold
        ),
        "answerable": sum(cells.values()),
        "containment_vs_judge": {c: cells[c] for c in CELLS},
        "cohen_kappa": kappa(cells),
        "by_category": {
            category: {c: counts[c] for c in CELLS}
            for category, counts in sorted(by_category.items())
        },
    }


def build() -> dict[str, Any]:
    rule = load_rule()
    runs = {run: score_run(rule, run) for run in full_harness_runs()}
    return {
        "artifact_kind": "pra-containment-exploratory",
        "generated_by": "scripts/pra_containment.py",
        "status": "exploratory: rule defined after reading these responses; not a reported metric",
        "rules": {
            "exact_whole_response": f"{RULE_PATH} exact_match(response, gold_answer)",
            "containment": "normalize_text(gold) occurs in normalize_text(response) as a whole "
            "word or phrase: (?<!\\w)gold(?!\\w)",
            "judge": "stored lenient_correct (judge score >= 3)",
        },
        "totals": {
            "runs": len(runs),
            "responses": sum(r["responses"] for r in runs.values()),
            "exact_whole_response": sum(r["exact_whole_response"] for r in runs.values()),
        },
        "runs": runs,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = parser.parse_args(argv)
    text = serialise(build())
    rel = OUTPUT.relative_to(ROOT).as_posix()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{rel} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{rel} is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
