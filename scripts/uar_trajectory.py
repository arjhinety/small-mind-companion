"""System E's abstention across the four SFT training corpora, re-scored under both phrase lists.

Four saved response sets of System E (SFT + memory, k=8) on the 688 PMB probes:

=============  ==========================================  ========================================
step           response set                                SFT corpus it was trained on
=============  ==========================================  ========================================
``v0``         results/v0.1/E_sft_memory (working tree)    data/sft/v0 (f0d7ece): 225 examples
``e9e5ca8``    results/v1_scale/E_sft_memory @ e9e5ca8     data/sft/v1 @ 2b4c308: dedup bug present
``5a9fe1b``    results/v1_scale/E_sft_memory @ 5a9fe1b     data/sft/v1 @ bc2f409: dedup fix only
``4fcf740``    results/v1_scale/E_sft_memory @ 4fcf740     data/sft/v1 @ 799390e: ratios + templates
=============  ==========================================  ========================================

The corpus for each run is inferred from commit order, not recorded by the run itself.

Every response is re-scored with ``detect_abstention``'s rule under ``_ABSTENTION_PHRASES[:17]``
(the list before the 2026-08-14 extensions) and under the full list. Stored ``abstained`` flags are
reported but not used, because the 5a9fe1b run was committed with flags from the original list.

For each list: UAR on the 80 unanswerable probes, answerable probes flagged as abstaining out of
608, and paired tests (exact McNemar, paired bootstrap 95% CI, imported from
``scripts/paired_tests_v0_1.py``) on both outcomes for adjacent steps and for 5a9fe1b against
4fcf740. These are probe-level tests on one training run per step: they cannot measure seed-to-seed
variance between training runs.

Needs git history, so it is not wired into ``scripts/validate.py``.

Usage::

    uv run python scripts/uar_trajectory.py          # (re)write the output JSON
    uv run python scripts/uar_trajectory.py --check  # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "audit" / "uar_trajectory" / "uar_trajectory.json"
RULE_PATH = "src/onebee/evaluation/graders/rule.py"
ORIGINAL_N = 17
V1_PATH = "results/v1_scale/E_sft_memory/raw.jsonl"
STEPS: tuple[tuple[str, str | None, str, str], ...] = (
    ("v0", None, "results/v0.1/E_sft_memory/raw.jsonl", "data/sft/v0 @ f0d7ece"),
    ("e9e5ca8", "e9e5ca8", V1_PATH, "data/sft/v1 @ 2b4c308 (dedup bug present)"),
    ("5a9fe1b", "5a9fe1b", V1_PATH, "data/sft/v1 @ bc2f409 (dedup fix only)"),
    ("4fcf740", "4fcf740", V1_PATH, "data/sft/v1 @ 799390e (ratio rebalance + templates)"),
)
COMPARISONS = (
    ("v0", "e9e5ca8"),
    ("e9e5ca8", "5a9fe1b"),
    ("5a9fe1b", "4fcf740"),
    ("v0", "4fcf740"),
)


def _import(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read(commit: str | None, rel: str) -> bytes:
    if commit is None:
        return (ROOT / rel).read_bytes()
    return subprocess.run(
        ["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True, check=True
    ).stdout


def _by_probe(data: bytes, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row["probe"]["probe_id"]
        if pid in out:
            raise ValueError(f"{label}: duplicate probe_id {pid}")
        out[pid] = row
    return out


def _abstains(phrases: list[str], response: str) -> int:
    lower = response.lower()
    return 1 if any(p in lower for p in phrases) else 0


def build() -> dict[str, Any]:
    tests = _import("paired_tests_v0_1", "scripts/paired_tests_v0_1.py")
    rule = _import("onebee_rule_by_path", RULE_PATH)
    phrases = list(rule._ABSTENTION_PHRASES)
    lists = {"original": phrases[:ORIGINAL_N], "current": phrases}

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, str] = {}
    corpora: dict[str, str] = {}
    for step, commit, rel, corpus in STEPS:
        data = _read(commit, rel)
        label = rel if commit is None else f"{commit}:{rel}"
        inputs[label] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        runs[step] = _by_probe(data, label)
        corpora[step] = corpus
    probe_sets = {frozenset(r) for r in runs.values()}
    if len(probe_sets) != 1:
        raise ValueError("the four runs cover different probes")
    any_run = next(iter(runs.values()))
    unans = sorted(pid for pid, row in any_run.items() if not row["probe"]["answerable"])
    ans = sorted(pid for pid, row in any_run.items() if row["probe"]["answerable"])

    def outcomes(step: str, phrase_list: list[str], ids: list[str]) -> list[int]:
        return [_abstains(phrase_list, str(runs[step][i]["response"])) for i in ids]

    steps: dict[str, Any] = {}
    for step, commit, rel, _ in STEPS:
        entry: dict[str, Any] = {
            "source": rel if commit is None else f"{commit}:{rel}",
            "trained_on": corpora[step],
            "stored": {
                "abstained_unanswerable": sum(1 for i in unans if runs[step][i]["abstained"]),
                "abstained_answerable": sum(1 for i in ans if runs[step][i]["abstained"]),
            },
        }
        for name, phrase_list in lists.items():
            u = sum(outcomes(step, phrase_list, unans))
            a = sum(outcomes(step, phrase_list, ans))
            entry[name] = {
                "abstained_unanswerable": u,
                "uar": u / len(unans),
                "abstained_answerable": a,
            }
        steps[step] = entry

    personas = tests.persona_of_probe()

    def paired(first: list[int], second: list[int], ids: list[str]) -> dict[str, Any]:
        first_only = sum(1 for x, y in zip(first, second, strict=True) if x and not y)
        second_only = sum(1 for x, y in zip(first, second, strict=True) if y and not x)
        lo, hi = tests.paired_bootstrap_ci_pp(first, second)
        groups = [personas[i] for i in ids]
        plo, phi = tests.persona_bootstrap_ci_pp(first, second, groups)
        return {
            "first": sum(first),
            "second": sum(second),
            "first_only": first_only,
            "second_only": second_only,
            "diff_pp": (sum(second) - sum(first)) / len(first) * 100,
            "mcnemar_exact_p": tests.mcnemar_exact(first_only, second_only),
            "bootstrap_ci_pp": [lo, hi],
            "n_personas": len(set(groups)),
            "persona_bootstrap_ci_pp": [plo, phi],
        }

    comparisons: dict[str, Any] = {}
    for first, second in COMPARISONS:
        comparisons[f"{first}_vs_{second}"] = {
            name: {
                "unanswerable_abstention": paired(
                    outcomes(first, pl, unans), outcomes(second, pl, unans), unans
                ),
                "answerable_flagged": paired(
                    outcomes(first, pl, ans), outcomes(second, pl, ans), ans
                ),
            }
            for name, pl in lists.items()
        }

    return {
        "artifact_kind": "uar-trajectory",
        "generated_by": "scripts/uar_trajectory.py",
        "lists": {
            "original": f"{RULE_PATH} _ABSTENTION_PHRASES[:{ORIGINAL_N}]",
            "current": f"{RULE_PATH} _ABSTENTION_PHRASES ({len(phrases)} phrases)",
        },
        "n_unanswerable": len(unans),
        "n_answerable": len(ans),
        "methods": {
            "test": "exact McNemar, two-sided (scripts/paired_tests_v0_1.mcnemar_exact)",
            "interval": (
                f"paired bootstrap, {tests.N_RESAMPLES} resamples, seed {tests.SEED}, 95% "
                "percentile interval (scripts/paired_tests_v0_1.paired_bootstrap_ci_pp)"
            ),
            "persona_interval": (
                "bootstrap over the 8 personas, resampling whole personas "
                "(scripts/paired_tests_v0_1.persona_bootstrap_ci_pp)"
            ),
            "direction": "differences are second minus first, in percentage points",
        },
        "scope": (
            "One training run (one seed) per step. Probe-level tests cannot measure "
            "seed-to-seed variance between training runs. The corpus behind each run is "
            "inferred from commit order."
        ),
        "inputs": inputs,
        "steps": steps,
        "comparisons": comparisons,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = parser.parse_args(argv)
    rel = OUTPUT.relative_to(ROOT).as_posix()
    fresh = serialise(build())
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
        if current != fresh:
            print(f"{rel} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{rel} is fresh")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fresh, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
