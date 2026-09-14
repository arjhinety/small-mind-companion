#!/usr/bin/env python
"""Audit validator: check every machine-checkable claim in this repository against its artifact.

Run it with::

    uv run python scripts/validate.py

It exits non-zero if anything fails. CPU-only, offline, no model download: every check reads
committed files. This is the executable form of ``docs/AUDIT.md``.

What it checks
--------------
1. ``freeze``       Study 001 freeze verification (the logic of ``freeze_study_001.py --check``).
2. ``hashes``       every dataset ``hash.txt`` (the logic of ``recompute_hashes.py``).
3. ``metrics``      ``uar`` / ``pra_lenient`` / ``pra_strict`` recomputed from every
                    ``results/**/raw.jsonl`` and compared to its ``metrics.json``.
4. ``claims``       every entry in ``registry/claims.jsonl``.
5. ``registries``   every count in ``registry/*.yaml`` against the data files themselves.
6. ``crossdoc``     the same metric must not be quoted with two values across README and docs.
7. ``hygiene``      determinism hygiene, reported but not fatal by default (see ``--strict``).
8. ``pollution``    no training corpus shares a 13-gram with the PMB evaluation benchmark.

``registry/claims.jsonl`` schema
--------------------------------
One JSON object per line::

    {
      "id":     "<stable slug>",
      "claim":  "<the sentence being asserted, quoted or paraphrased>",
      "where":  "<file>:<line> of the claim",          # physical line number in the file on disk
      "artifact": "<the file that backs it, or null>",
      "check":  { ...see below... },
      "status": "verified" | "manual" | "acknowledged_drift",
      "resolution": "<required when status is acknowledged_drift>"
    }

``check`` types
~~~~~~~~~~~~~~~
``{"type": "file_exists", "path": "<rel>", "equals": true|false}``
    ``equals`` defaults to true. A path that must *not* exist is expressible.

``{"type": "json_field", "path": "<rel>", "field": "a.b.c", "equals": v}``
    Loads JSON, or YAML when the path ends in ``.yaml``/``.yml``; walks the dotted field. The
    pseudo-field ``"len"`` yields the length of a top-level array or object. ``tolerance`` (default
    1e-12) gives absolute float comparison. With ``decimals`` and ``quoted``, the value is also
    checked as a percentage: ``round(value * 100, decimals) == quoted`` -- this is how a document
    that says "15.10%" is compared to an artifact that holds 0.1513157894...

``{"type": "count_lines", "glob": "<rel glob>", "equals": n}``
    Counts non-empty lines across the glob's file matches.

``{"type": "computed", "expr": "<expression>"}``
    Evaluates an expression in a restricted language: numeric/string literals, tuples, ``+ - * /
    // %``, ``== != < <= > >=``, ``and or not``, and calls to the named helpers in this module.
    No attribute access, no subscripts, no comprehensions, no builtins. Helpers take literal
    arguments only. ``equals``/``tolerance`` compare the result when the expression is not itself a
    comparison; an expression that already returns a bool is used directly.

``{"type": "cross_doc", "equals": v, "sources": [{"path": .., "pattern": ..}, ...]}``
    Extracts one capture group per source with a regex and requires every captured number to equal
    ``equals``. This is the cross-document consistency check in its most explicit form: every
    source must quote the same value, and that value must be the expected one.

``{"type": "manual", "reason": "<why>"}``
    Not machine-checkable. Always reported, never silently passed -- the count of manual claims is
    printed in the summary.

Statuses
~~~~~~~~
``verified``            the check must pass.
``manual``              not machine-checkable; ``check.type`` must be ``manual`` with a reason.
``acknowledged_drift``  the check is *expected to fail*. Used only where the quoting document is a
                        Study 001 frozen artifact that the audit rules forbid editing, so the drift
                        is recorded and reported instead of silently corrected. A
                        ``acknowledged_drift`` entry that starts passing is itself a failure: the
                        matrix is then stale.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

CLAIMS = ROOT / "registry" / "claims.jsonl"
STUDIES_REGISTRY = ROOT / "registry" / "studies.yaml"
DATASETS_REGISTRY = ROOT / "registry" / "datasets.yaml"
BENCHMARKS_REGISTRY = ROOT / "registry" / "benchmarks.yaml"
HASHES_REGISTRY = ROOT / "registry" / "hashes.json"
RESULT_REGISTRY = ROOT / "results" / "registry.jsonl"
FREEZE_MANIFEST = ROOT / "reports" / "data" / "study-001-freeze.json"
PMB_PROBES = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
GUARDRAILS = ROOT / "docs" / "GUARDRAILS.md"
ERRATA = ROOT / "reports" / "ERRATA.md"

TOLERANCE = 1e-12

# Top-level directories a repo-relative path reference may start with. `referenced_paths_exist`
# uses this so that a backticked model id (`google/gemma-4-E4B-it`) or a gitignored output directory
# (`outputs/`) is not mistaken for a tracked path that must resolve.
PATH_PREFIXES = (
    "src/",
    "docs/",
    "scripts/",
    "configs/",
    "data/",
    "results/",
    "tests/",
    "reports/",
    "registry/",
    "hf_readmes/",
    "paper/",
    "mobile/",
    "notebooks/",
    "experiments/",
    "assets/",
    "benchmarks/",
)

# Headline metrics whose value must be identical wherever it is quoted. Each entry pairs the
# document quote (a regex with exactly one capture group) with the value the documents state for it,
# and the cross-document check requires every source to agree with that value AND with the artifact
# that backs it. Each pattern must match exactly once per document, so a second contradictory quote
# inside one document fails too.
# Shared regex fragments for the docs/STUDIES.md stage-table rows, kept as named pieces so the
# patterns stay readable and inside the line limit. Each captures one percentage.
_STUDIES_A = r"\| A \| raw model, no memory \|[^|]*\| pra_lenient "
_STUDIES_A_UAR = _STUDIES_A + r"[\d.]+% / UAR ([\d.]+)%"
_STUDIES_A_PRA = _STUDIES_A + r"([\d.]+)%"
_STUDIES_D = r"\| D \| raw \+ hybrid retrieval memory \(k=8\) \|[^|]*\| "
_STUDIES_D_PRA = _STUDIES_D + r"\*?\*?([\d.]+)%"
_STUDIES_D_UAR = _STUDIES_D + r"\*?\*?[\d.]+% / \*?\*?([\d.]+)%"

HEADLINE_QUOTES: list[dict[str, Any]] = [
    {
        "metric": "pra_lenient_A_raw",
        "value": 0.16,
        "sources": [
            {"path": "README.md", "pattern": r"raw model, no memory \| ([\d.]+)%"},
            {"path": "docs/STUDIES.md", "pattern": _STUDIES_A_PRA},
        ],
    },
    {
        "metric": "uar_A_raw",
        "value": 13.75,
        "sources": [
            {"path": "README.md", "pattern": r"raw model, no memory \| [\d.]+% \| ([\d.]+)%"},
            {"path": "docs/STUDIES.md", "pattern": _STUDIES_A_UAR},
        ],
    },
    {
        "metric": "pra_lenient_B_sft_v0",
        "value": 0.16,
        "sources": [
            {"path": "README.md", "pattern": r"\| LoRA SFT, no memory \| ([\d.]+)%"},
            {"path": "docs/STUDIES.md", "pattern": r"\| B \(v0\) \|[^|]*\|[^|]*\| ([\d.]+)%"},
        ],
    },
    {
        "metric": "uar_B_sft_v0",
        "value": 16.25,
        "sources": [
            {"path": "README.md", "pattern": r"\| LoRA SFT, no memory \| [\d.]+% \| ([\d.]+)%"},
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| B \(v0\) \|[^|]*\|[^|]*\| [\d.]+% / ([\d.]+)%",
            },
        ],
    },
    {
        "metric": "pra_lenient_D_memory",
        "value": 15.13,
        "sources": [
            {
                "path": "README.md",
                "pattern": r"hybrid retrieval memory \(k=8\), no SFT \| ([\d.]+)%",
            },
            {"path": "docs/STUDIES.md", "pattern": _STUDIES_D_PRA},
        ],
    },
    {
        "metric": "uar_D_memory",
        "value": 8.75,
        "sources": [
            {
                "path": "README.md",
                "pattern": r"hybrid retrieval memory \(k=8\), no SFT \| [\d.]+% \| ([\d.]+)%",
            },
            {"path": "docs/STUDIES.md", "pattern": _STUDIES_D_UAR},
        ],
    },
    {
        "metric": "pra_lenient_E_sft_memory_v0",
        "value": 17.76,
        "sources": [
            {"path": "README.md", "pattern": r"\| LoRA SFT \+ memory \| ([\d.]+)%"},
            {"path": "docs/STUDIES.md", "pattern": r"\| E \(v0\) \|[^|]*\|[^|]*\| ([\d.]+)%"},
        ],
    },
    {
        "metric": "uar_E_sft_memory_v0",
        "value": 33.75,
        "sources": [
            {"path": "README.md", "pattern": r"\| LoRA SFT \+ memory \| [\d.]+% \| ([\d.]+)%"},
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| E \(v0\) \|[^|]*\|[^|]*\| [\d.]+% / ([\d.]+)%",
            },
        ],
    },
    {
        "metric": "pra_lenient_E_sft_memory_v1",
        "value": 15.30,
        "sources": [
            {"path": "README.md", "pattern": r"proper-scale LoRA SFT \+ memory \| ([\d.]+)%"},
            {"path": "docs/STUDIES.md", "pattern": r"\| E \(v1\) \|[^|]*\|[^|]*\| ([\d.]+)%"},
        ],
    },
    {
        "metric": "uar_E_sft_memory_v1",
        "value": 70.0,
        "sources": [
            {
                "path": "README.md",
                "pattern": r"proper-scale LoRA SFT \+ memory \| [\d.]+% \| ([\d.]+)%",
            },
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| E \(v1\) \|[^|]*\|[^|]*\| [\d.]+% / ([\d.]+)%",
            },
        ],
    },
    {
        "metric": "pra_lenient_E_distill_v1",
        "value": 18.59,
        "sources": [
            {"path": "README.md", "pattern": r"on-policy distillation from[^|]*\| ([\d.]+)%"},
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| E-distill \|[^|]*\|[^|]*\| \*?\*?([\d.]+)%",
            },
        ],
    },
    {
        "metric": "uar_E_distill_v1",
        "value": 71.25,
        "sources": [
            {
                "path": "README.md",
                "pattern": r"on-policy distillation from[^|]*\| [\d.]+% \| ([\d.]+)%",
            },
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| E-distill \|[^|]*\|[^|]*\| \*?\*?[\d.]+% / ([\d.]+)%",
            },
        ],
    },
    {
        "metric": "c_vs_e_gap_pp",
        "value": 24.7,
        "sources": [
            {
                "path": "README.md",
                "pattern": r"H6, H7 \|[^|]*\| Confirmed[^|]*?([\d.]+)pp pairwise",
            },
            {
                "path": "docs/STUDIES.md",
                "pattern": r"\| C-vs-E \|[^|]*\|[^|]*\| [\d.]+% vs [\d.]+% \(\+([\d.]+)pp\)",
            },
        ],
    },
]

# Config hyperparameters the README table asserts, and where each value actually lives. A value
# is only "pinned in this repository" if a file in this repository states it; values that come
# from a dataclass default are read from that dataclass's source, and the README's claim that a
# value is a run default is checked against the default rather than against the config.
CONFIG_VALUES: dict[str, dict[str, Any]] = {
    "sft_v1": {
        "config": "configs/training/sft_v1.yaml",
        "values": {
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 1e-4,
            "num_train_epochs": 2.0,
            "max_seq_length": 2048,
            "per_device_train_batch_size": 8,
            "seed": 1337,
        },
    },
    "sft_v0": {
        "config": "configs/training/sft.yaml",
        "values": {"per_device_train_batch_size": 4},
    },
    "dpo_v1_scale": {
        "config": "configs/training/dpo_v1_scale.yaml",
        "values": {},
    },
    "distill_v1": {
        "config": "configs/training/distill_v1.yaml",
        "values": {
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 2,
            "max_completion_length": 128,
            "teacher_model": "google/gemma-4-E4B-it",
        },
    },
}

# Dataclass defaults, read from source with `ast` so this stays offline and import-free (the
# training modules import torch/trl, which CI does not install).
TRAINING_DEFAULTS: dict[str, dict[str, Any]] = {
    "src/onebee/training/sft.py": {
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "num_train_epochs": 2.0,
        "max_seq_length": 2048,
        "per_device_train_batch_size": 8,
    },
    "src/onebee/training/dpo.py": {
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 5e-6,
        "num_train_epochs": 1.0,
        "max_seq_length": 2048,
        "per_device_train_batch_size": 4,
    },
    "src/onebee/training/distill.py": {
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-6,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 2,
    },
}

# Corpora whose hash.txt is recomputed. Kept in sync with scripts/recompute_hashes.py by asserting
# at check time that the two lists agree.
BENCHMARK_HASH_DIRS = [
    "data/benchmarks/pmb_v0_full",
    "data/benchmarks/sft_personas_v1",
    "data/benchmarks/pmb_v0",
    "data/benchmarks/sft_personas_v0",
    # H22 (abliteration judgment quality) and H24 (emotional range) probe sets. Committed corpora
    # that were missing from both this list and recompute_hashes.py, so they had no hash.txt; the
    # hygiene check found them. The sync assertion below is why this edit was required in both
    # files at once rather than one of them silently drifting.
    "data/benchmarks/h22_judgment",
    "data/benchmarks/emotional_range",
]
DATASET_HASH_DIRS = [
    "data/sft/v0",
    "data/sft/v1",
    "data/dpo/v0",
    "data/dpo/v1_scale",
    "data/distill/v1",
]

# Corpora that must clear the 13-gram check against the PMB benchmark.
CONTAMINATION_CORPORA = [
    "data/sft/v0/train.jsonl",
    "data/sft/v1/train.jsonl",
    "data/dpo/v0/train.jsonl",
    "data/dpo/v1_scale/train.jsonl",
    "data/distill/v1/train.jsonl",
]

REQUIRED_CLAIM_FIELDS = ("id", "claim", "where", "artifact", "check", "status")
VALID_STATUSES = ("verified", "manual", "acknowledged_drift")
# Only a claim is expected to carry one of these; used to make the acknowledged-drift ledger
# machine-readable rather than a comment.
REQUIRED_DRIFT_FIELDS = ("resolution",)


# --------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    import yaml  # pyyaml is a base runtime dependency (pyproject.toml)

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_document(path: Path) -> Any:
    if path.suffix in (".yaml", ".yml"):
        return load_yaml(path)
    return load_json(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf_normalised(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def nonempty_line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    )


def walk_dotted(document: Any, field: str) -> Any:
    if field == "len":
        return len(document)
    node = document
    for part in field.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def close(actual: Any, expected: Any, tolerance: float = TOLERANCE) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return actual == expected


def import_script(name: str, rel: str) -> Any:
    """Import a sibling script by path. Used so this validator reuses the repo's own logic rather
    than reimplementing it, and so a change to that logic changes both together."""
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"expected helper script {rel}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probe_categories() -> dict[str, str]:
    return {row["probe_id"]: row["category"] for row in load_jsonl(PMB_PROBES)}


def pmb_rows() -> list[dict[str, Any]]:
    return load_jsonl(PMB_PROBES)


def result_registry_rows() -> list[dict[str, Any]]:
    return [row for row in load_jsonl(RESULT_REGISTRY) if row.get("kind") != "header"]


def pmb_metric_entries() -> list[dict[str, Any]]:
    return [
        row for row in result_registry_rows() if (row.get("n_probes") == 688) and row.get("metrics")
    ]


def config_value(rel: str, key: str) -> Any:
    return load_document(ROOT / rel).get(key)


def dataclass_defaults(rel: str) -> dict[str, Any]:
    """Read `name: type = default` dataclass fields from a source file without importing it."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None:
                        try:
                            out[stmt.target.id] = ast.literal_eval(stmt.value)
                        except ValueError:
                            continue
    return out


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------
# named helpers available to `computed` expressions in registry/claims.jsonl
# --------------------------------------------------------------------------------------------
def split_counts(rel: str) -> tuple[int, int]:
    train = ROOT / rel / "train.jsonl"
    val = ROOT / rel / "val.jsonl"
    return (nonempty_line_count(train), nonempty_line_count(val))


def count_test_functions() -> int:
    total = 0
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    return total


def numbered_item_count(rel: str) -> int:
    return len(re.findall(r"^\d+\.\s", read_text(rel), re.M))


def companion_persona_test_count() -> int:
    return count_tests_in("tests/unit/test_companion_persona.py")


def count_tests_in(rel: str) -> int:
    tree = ast.parse(read_text(rel))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def pmb_persona_count() -> int:
    return len({row["persona_id"] for row in pmb_rows()})


def pmb_category_count() -> int:
    return len({row["category"] for row in pmb_rows()})


def pmb_probes_per_persona() -> int:
    rows = pmb_rows()
    return len(rows) // pmb_persona_count()


def pmb_design() -> bool:
    """688 probes, 8 personas x 8 categories, every one of the 64 cells non-empty."""
    rows = pmb_rows()
    personas = {row["persona_id"] for row in rows}
    categories = {row["category"] for row in rows}
    cells = {(row["persona_id"], row["category"]) for row in rows}
    return (
        len(rows) == 688
        and len(personas) == 8
        and len(categories) == 8
        and len(cells) == 64 == len(personas) * len(categories)
        and len(rows) == 86 * len(personas)
    )


def pmb_answerable_split() -> bool:
    rows = pmb_rows()
    answerable = sum(1 for row in rows if row.get("answerable"))
    return answerable == 608 and len(rows) - answerable == 80


def pmb_per_category() -> bool:
    counts: dict[str, int] = {}
    for row in pmb_rows():
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts == {
        "factual": 98,
        "continuity": 96,
        "distractor": 96,
        "outdated_fact": 96,
        "preference": 91,
        "unanswerable": 80,
        "episodic": 69,
        "temporal": 62,
    }


def pmb_unanswerable_has_no_gold() -> bool:
    return all(not row.get("gold_answer") for row in pmb_rows() if not row.get("answerable"))


def acceptable_alternatives_populated() -> int:
    return sum(1 for row in pmb_rows() if row.get("acceptable_alternatives"))


def all_pra_strict_near_zero() -> bool:
    values = [row["metrics"]["pra_strict"] for row in result_registry_rows() if row.get("metrics")]
    return bool(values) and max(values) < 0.01


def no_metrics_for_dpo_row() -> bool:
    """No result carries a full-PMB pra_lenient/uar for a DPO checkpoint.

    The only DPO-lineage entries that exist are pairwise runs, and they have no metrics object.
    """
    dpo_only = [
        row
        for row in result_registry_rows()
        if (row.get("training_stages_present") or {}).get("dpo")
        and not (row.get("training_stages_present") or {}).get("distill")
    ]
    if not dpo_only:
        return False
    return all(row.get("metrics") is None for row in dpo_only if row.get("run_kind") == "pairwise")


def _pmb_system(value: dict[str, Any]) -> float | None:
    return None if value is None else float(value)


def system_metrics(system_name: str) -> dict[str, Any]:
    for row in result_registry_rows():
        if row.get("system_name") == system_name or row.get("run_path", "").endswith(
            f"/{system_name}"
        ):
            if row.get("metrics"):
                return row["metrics"]
    raise KeyError(f"no indexed run named {system_name}")


def c_vs_e_gap_pp() -> bool:
    """45.7% vs 21.0% -- a 24.7pp gap, at the precision the documents quote."""
    summary = load_json(ROOT / "results/v1_scale/C_vs_E_pairwise/summary.json")
    c = round(summary["c_win_rate"] * 100, 1)
    e = round(summary["e_win_rate"] * 100, 1)
    return c == 45.7 and e == 21.0 and round(c - e, 1) == 24.7


def c_vs_f_gap_pp() -> bool:
    """F wins 38.1% vs C's 30.5% -- +7.6pp, the isolating H23 measurement (ERRATA E2)."""
    summary = load_json(ROOT / "results/v1_scale/C_vs_F_distill_pairwise/summary.json")
    c = round(summary["c_win_rate"] * 100, 1)
    f = round(summary["f_win_rate"] * 100, 1)
    return c == 30.5 and f == 38.1 and round(f - c, 1) == 7.6


def h23_uplift_pp() -> bool:
    """pra_lenient 15.30% -> 18.59%, i.e. +3.3pp, with UAR effectively flat."""
    before = system_metrics("E_sft_memory_v1")
    after = system_metrics("E_distill_v1")
    delta = round((after["pra_lenient"] - before["pra_lenient"]) * 100, 1)
    uar_delta = abs(round((after["uar"] - before["uar"]) * 100, 2))
    return delta == 3.3 and uar_delta <= 2.0


def h23_confound_stated() -> bool:
    """The +3.3pp spans DPO as well as distillation, and the write-ups say so (ERRATA E2)."""
    return all(
        "DPO stage" in read_text(rel)
        for rel in (
            "docs/STUDIES.md",
            "docs/distillation_results.md",
            "reports/ERRATA.md",
        )
    )


def c_vs_f_labeled_as_isolating() -> bool:
    """The C-vs-F comparison is labelled as the one that isolates distillation from DPO."""
    return (
        "DPO stage" in read_text("reports/ERRATA.md")
        and "DPO stage" in read_text("docs/STUDIES.md")
        and "isolating" in read_text("docs/GUARDRAILS.md")
    )


def distillation_effect_on_uar_pp() -> bool:
    """The UAR half of the retrieval finding: 13.75% -> 8.75% is a 5pp drop."""
    return (
        round((system_metrics("A_raw")["uar"] - system_metrics("D_memory")["uar"]) * 100, 1) == 5.0
    )


def retrieval_pra_delta_matches_quote() -> bool:
    """The pra_lenient half: the docs' +14.9pp follows from their own (drifted) 15.10% quote."""
    return round(15.10 - 0.16, 1) == 14.9


def retrieval_effect_pp() -> bool:
    """Adding memory to the raw model: pra_lenient 0.16% -> 15.13% (+15.0pp), UAR -5pp.

    The +15.0pp was +14.9pp until ERRATA E31 corrected D_memory's quoted pra_lenient from 15.10%
    to the artifact's 15.13%.
    """
    raw = system_metrics("A_raw")
    mem = system_metrics("D_memory")
    pra_delta = round((mem["pra_lenient"] - raw["pra_lenient"]) * 100, 1)
    uar_delta = round((raw["uar"] - mem["uar"]) * 100, 1)
    return pra_delta == 15.0 and uar_delta == 5.0


def training_config_revision_pinning() -> bool:
    """Only the SFT configs pin a 40-char SHA; the DPO and distillation configs use "main"."""
    pinned = {}
    for path in sorted((ROOT / "configs/training").glob("*.yaml")):
        value = load_document(path).get("base_model_revision")
        pinned[path.name] = value
    expected = {
        "sft.yaml": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "sft_v1.yaml": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "dpo.yaml": "main",
        "dpo_v1_scale.yaml": "main",
        "dpo_v1_more_epochs.yaml": "main",
        "distill_v1.yaml": "main",
    }
    return pinned == expected


def base_model_revision_claims() -> bool:
    sha = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    return (
        config_value("configs/training/sft.yaml", "base_model") == "google/gemma-4-E2B-it"
        and config_value("configs/training/sft_v1.yaml", "base_model") == "google/gemma-4-E2B-it"
        and config_value("configs/training/sft.yaml", "base_model_revision") == sha
        and sha in read_text("docs/STUDIES.md")
        and len(sha) == 40
    )


def config_values_match_claims() -> bool:
    for spec in CONFIG_VALUES.values():
        for key, expected in spec["values"].items():
            actual = config_value(spec["config"], key)
            if not close(actual, expected):
                return False
    for rel, expected_defaults in TRAINING_DEFAULTS.items():
        defaults = dataclass_defaults(rel)
        for key, expected in expected_defaults.items():
            if not close(defaults.get(key), expected):
                return False
    # The README says the DPO learning rate is the run default, so it must NOT be pinned in the
    # DPO config; a value appearing there would make the table's "(DPO default)" text wrong.
    return "learning_rate" not in load_document(ROOT / "configs/training/dpo_v1_scale.yaml")


def referenced_paths_exist(rel: str) -> bool:
    """Every repo-relative path in backticks or a markdown link inside `rel` resolves.

    A reference is only treated as a path when it starts with a known top-level directory, so model
    ids and prose fragments in backticks are not misread. `outputs/` is deliberately excluded: it is
    a gitignored training-output directory that does not exist in a fresh clone, which is a
    documented property of this repository (README's Checkpoints section), not a broken link.
    """
    text = read_text(rel)
    candidates = set(re.findall(r"\]\(([^)#]+)\)", text)) | set(re.findall(r"`([^`\s]+)`", text))
    missing = []
    for candidate in sorted(candidates):
        candidate = candidate.strip()
        if candidate.startswith(("http://", "https://", "#")):
            continue
        target = candidate.split("#", 1)[0].strip().rstrip("/")
        if not target or "*" in target or not target.startswith(PATH_PREFIXES):
            continue
        if not (ROOT / target).exists():
            missing.append(target)
    return not missing


def hash_clean_corpus_count() -> int:
    module = import_script("recompute_hashes", "scripts/recompute_hashes.py")
    clean = 0
    for rel, fn in [(d, module.benchmark_hash) for d in BENCHMARK_HASH_DIRS] + [
        (d, module.dataset_hash) for d in DATASET_HASH_DIRS
    ]:
        directory = ROOT / rel
        if not directory.is_dir():
            continue
        stored = (directory / "hash.txt").read_text(encoding="utf-8").strip()
        if stored == fn(directory):
            clean += 1
    return clean


def hash_corpus_count_with_hash_txt() -> int:
    return len(
        [d for d in BENCHMARK_HASH_DIRS + DATASET_HASH_DIRS if (ROOT / d / "hash.txt").is_file()]
    )


def freeze_manifest() -> dict[str, Any]:
    return load_json(FREEZE_MANIFEST)


def freeze_file_count_matches_evidence() -> bool:
    manifest = freeze_manifest()
    return manifest["file_count"] == sum(len(group) for group in manifest["evidence"].values())


def freeze_shape_claims() -> bool:
    manifest = freeze_manifest()
    return (
        manifest["file_count"] == 76
        and sorted(manifest["evidence"])
        == sorted(
            [
                "benchmark",
                "training_data",
                "eval_results",
                "eval_raw",
                "report",
                "config",
                "public_surface",
            ]
        )
        and len(manifest["evidence"]) == 7
    )


def freeze_check_passes() -> bool:
    module = import_script("freeze_study_001", "scripts/freeze_study_001.py")
    fresh = module.build()
    recorded = freeze_manifest()
    return (
        recorded.get("evidence") == fresh["evidence"]
        and recorded.get("file_count") == fresh["file_count"]
    )


def freeze_metadata_matches_studies_doc() -> bool:
    manifest = freeze_manifest()
    studies = load_yaml(STUDIES_REGISTRY)
    study = next(s for s in studies["studies"] if s["id"] == "001")
    doc = read_text("docs/STUDIES.md")
    return (
        manifest["study_id"] == "001"
        and manifest["evidence_tag"] == study["evidence_tag"] == "study-001"
        and manifest["frozen_on"] == study["frozen_on"] == "2026-09-13"
        and "FROZEN" in doc
        and study["evidence_tag"] in doc
    )


def benchmark_corpora_freeze_pinned() -> bool:
    """The evaluation benchmark's bytes are pinned by the freeze manifest.

    Only `pmb_v0_full` is in the manifest's `benchmark` group: the sibling corpora under
    `data/benchmarks/` are training-persona sets, and pinning them as benchmarks would re-make the
    E3 mistake ("40 personas" read as the benchmark's persona count).
    """
    entries = freeze_manifest()["evidence"]["benchmark"]
    required = [
        "data/benchmarks/pmb_v0_full/probes.jsonl",
        "data/benchmarks/pmb_v0_full/DATASHEET.md",
        "data/benchmarks/pmb_v0_full/hash.txt",
    ]
    if not all(key in entries for key in required):
        return False
    personas = [key for key in entries if key.startswith("data/benchmarks/pmb_v0_full/personas/")]
    training = [key for key in entries if key.startswith("data/benchmarks/sft_personas")]
    return len(personas) == 8 and not training


def freeze_hash_gap_note_resolved() -> bool:
    """The manifest's not_claimed note says distill/v1 has no hash.txt. It has one now.

    Returns False while the stale sentence is still in the manifest, which is what makes the
    acknowledged-drift claim for it fail as recorded.
    """
    note = " ".join(freeze_manifest().get("not_claimed", []))
    return "data/distill/v1/ has no hash.txt" not in note


def distill_hash_claim_resolved() -> bool:
    """data/distill/v1/hash.txt exists and is a valid pin, so the E29/not_claimed gap is closed."""
    hash_file = ROOT / "data/distill/v1/hash.txt"
    if not hash_file.is_file():
        return False
    module = import_script("recompute_hashes", "scripts/recompute_hashes.py")
    return hash_file.read_text(encoding="utf-8").strip() == module.dataset_hash(
        ROOT / "data/distill/v1"
    )


def distill_subset_relation() -> bool:
    """The DATASHEET says the val prompts match data/sft/v1/val.jsonl 224/248.

    The actual relation: all 2232 distill (train+val) prompts come from the SFT v1 *train* split,
    and 0 of the 224 val prompts appear in the SFT v1 val split. The claim as written does not
    hold, which is why its matrix entry is an acknowledged drift.
    """
    sft_train = {
        _prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/train.jsonl")
    }
    sft_val = {_prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/val.jsonl")}
    dst_val = {_prompt_key(row["prompt"]) for row in load_jsonl(ROOT / "data/distill/v1/val.jsonl")}
    return len(dst_val & sft_val) == 224 and dst_val <= sft_train


def distill_subset_actual() -> dict[str, int]:
    sft_train = {
        _prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/train.jsonl")
    }
    sft_val = {_prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/val.jsonl")}
    dst_train = [
        _prompt_key(row["prompt"]) for row in load_jsonl(ROOT / "data/distill/v1/train.jsonl")
    ]
    dst_val = [_prompt_key(row["prompt"]) for row in load_jsonl(ROOT / "data/distill/v1/val.jsonl")]
    return {
        "distill_train_in_sft_train": sum(1 for x in dst_train if x in sft_train),
        "distill_train_total": len(dst_train),
        "distill_val_in_sft_val": sum(1 for x in dst_val if x in sft_val),
        "distill_val_in_sft_train": sum(1 for x in dst_val if x in sft_train),
        "distill_val_total": len(dst_val),
    }


def _prompt_key(messages: list[dict[str, Any]]) -> str:
    """The concatenated system+user text of an example, which is what identifies a prompt."""
    return "\x00".join(m["content"] for m in messages if m["role"] in ("system", "user"))


def dpo_sft_prompt_overlap() -> int:
    """How many DPO v1_scale prompts appear verbatim among the SFT v1 system texts (0)."""
    sft_systems = {
        m["content"]
        for row in load_jsonl(ROOT / "data/sft/v1/train.jsonl")
        for m in row["messages"]
        if m["role"] == "system"
    }
    dpo_prompts = [
        row["prompt"]
        for split in ("train", "val")
        for row in load_jsonl(ROOT / f"data/dpo/v1_scale/{split}.jsonl")
    ]
    return sum(1 for prompt in dpo_prompts if prompt in sft_systems)


def training_persona_corpus_counts() -> bool:
    v1 = load_jsonl(ROOT / "data/benchmarks/sft_personas_v1/probes.jsonl")
    return (
        len(v1) == 3437
        and len({row["persona_id"] for row in v1}) == 40
        and len(load_jsonl(ROOT / "data/benchmarks/sft_personas_v0/probes.jsonl")) == 344
        and len(load_jsonl(ROOT / "data/benchmarks/pmb_v0/probes.jsonl")) == 36
    )


def persona_fact_overlap() -> dict[str, int]:
    """How far the training-persona fact pool overlaps the PMB evaluation fact pool.

    The README says the training personas are "generated disjoint from the PMB eval personas".
    Persona *identity* differs (different generation runs, different files); the fact *values* come
    from a shared template pool, so this overlap is large. The number is recorded rather than
    smoothed over -- see docs/AUDIT.md.
    """

    def facts(rel: str) -> set[tuple[str, str, str]]:
        out: set[tuple[str, str, str]] = set()
        for path in sorted((ROOT / rel).glob("*.json")):
            payload = load_json(path)
            for fact in payload["persona"]["fact_sheet"]:
                out.add((fact["predicate"], fact["object"], fact["category"]))
        return out

    pmb = facts("data/benchmarks/pmb_v0_full/personas")
    sft = facts("data/benchmarks/sft_personas_v1/personas")
    return {"pmb_facts": len(pmb), "sft_facts": len(sft), "shared": len(pmb & sft)}


def stddev(values: list[float]) -> float:  # noqa: D103 - tiny statistics helper
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def mirrored_metric_sets() -> list[tuple[str, str]]:  # noqa: D103
    import yaml

    registry = yaml.safe_load(read_text("registry/datasets.yaml"))
    return [(entry["id"], entry["hash_algorithm"]) for entry in registry["datasets"]]


def training_personas_disjoint_from_pmb() -> bool:
    """Are the training-persona corpora disjoint from the PMB evaluation personas?

    Two separate questions, deliberately not conflated:

    * probe questions -- disjoint, and this is the check that matters for leakage.
    * persona files -- distinct generation runs with distinct identities, but the same fact-template
      pool, so the *values* overlap heavily (see persona_fact_overlap / the hygiene report).

    This returns the question's strict reading, which is `False`. The registry records that, and the
    narrower probe-level claim is checked separately by `training_persona_probe_overlap()`.
    """
    overlap = persona_fact_overlap()
    return overlap["shared"] == 0 and training_persona_probe_overlap() == 0


def training_persona_probe_overlap() -> int:
    """How many probe questions the training-persona corpora share with PMB (0)."""
    pmb_questions = {row["question"] for row in pmb_rows()}
    shared = 0
    for rel in ("data/benchmarks/sft_personas_v1", "data/benchmarks/sft_personas_v0"):
        shared += sum(
            1 for row in load_jsonl(ROOT / rel / "probes.jsonl") if row["question"] in pmb_questions
        )
    return shared


def contamination_clean_13gram() -> bool:
    """The 13-gram check from scripts/check_contamination.py, run on every training corpus."""
    module = import_script("check_contamination", "scripts/check_contamination.py")
    eval_file = str(PMB_PROBES)
    for rel in CONTAMINATION_CORPORA:
        train = str(ROOT / rel)
        if module.check_contamination([train], [eval_file], 13, 1):
            return False
    return True


def emotional_range_registers() -> int:
    rows = load_jsonl(ROOT / "data/benchmarks/emotional_range/probes.jsonl")
    return len({row.get("emotional_register") for row in rows})


def build_pmb_default_seed() -> int:
    text = read_text("scripts/build_pmb.py")
    match = re.search(r'"--seed",\s*type=int,\s*default=(\d+)', text)
    if not match:
        raise ValueError("could not find --seed default in scripts/build_pmb.py")
    return int(match.group(1))


def imatrix_calibration_shape() -> bool:
    path = ROOT / "data/imatrix_calibration.txt"
    # Compared over LF-normalised bytes, not raw st_size. This repo has no .gitattributes and sets
    # core.autocrlf=true, so the raw size differs by one byte per CRLF line between a Windows and a
    # Linux checkout (81,998 bytes here) -- the raw check passed locally and failed in CI. The line
    # count is platform-independent either way.
    normalised_size = len(path.read_bytes().replace(b"\r\n", b"\n"))
    return normalised_size == 11004368 and nonempty_line_count(path) == 66383


def license_files_exist() -> bool:
    return (ROOT / "LICENSE").is_file() and (ROOT / "LICENSE-DATA").is_file()


def no_quantization_artifact_committed() -> bool:
    """No llama-bench/llama-imatrix output and no .gguf file is committed (ERRATA E6)."""
    tracked = _tracked_files()
    if tracked is None:
        candidates = [
            p.relative_to(ROOT).as_posix()
            for pattern in ("**/*.gguf", "**/*llama-bench*", "**/*llama-imatrix*")
            for p in ROOT.glob(pattern)
            if ".venv" not in p.parts and ".git" not in p.parts
        ]
    else:
        candidates = [
            p for p in tracked if p.endswith(".gguf") or "llama-bench" in p or "llama-imatrix" in p
        ]
    return not candidates and not (ROOT / "results" / "quantization").exists()


def _tracked_files() -> list[str] | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.splitlines()


def recomputed_run_count() -> int:
    """How many runs' metrics recompute exactly from their own saved responses (7)."""
    return sum(
        1
        for row in pmb_metric_entries()
        if row.get("recomputed_matches_recorded") and row.get("recomputed_in_claim_audit")
    )


def reported_v0_rows_reconcile() -> bool:
    """Every v0 eval-ladder row the write-ups quote reconciles with its artifact at its own
    precision: A 0.16%/13.75%, B 0.16%/16.25%, D 15.13%/8.75%, E 17.76%/33.75%.

    D's pra_lenient was quoted as 15.10% until ERRATA E31; this check is what caught it.
    """
    expected = {
        "A_raw": (0.16, 13.75),
        "B_sft": (0.16, 16.25),
        "D_memory": (15.13, 8.75),
        "E_sft_memory": (17.76, 33.75),
    }
    for system, (pra_pct, uar_pct) in expected.items():
        metrics = system_metrics(system)
        if round(metrics["pra_lenient"] * 100, 2) != pra_pct:
            return False
        if round(metrics["uar"] * 100, 2) != uar_pct:
            return False
    return True


def study_002_has_no_results() -> bool:
    studies = load_yaml(STUDIES_REGISTRY)
    study = next(s for s in studies["studies"] if s["id"] == "002")
    if study["status"] != "IN_PROGRESS" or study["results"] or study["result_count"] != 0:
        return False
    return all(row.get("study") == "001" for row in result_registry_rows())


def human_review_notes_precise() -> bool:
    """Every corpus a reported claim rests on is documented as not human-reviewed.

    data/distill/v1 inherits its parent's status (strict subset) and data/benchmarks/pmb_v0 is a
    deterministic fixture; both are documented as such instead.
    """
    must_state = [
        "data/sft/v0",
        "data/sft/v1",
        "data/dpo/v0",
        "data/dpo/v1_scale",
        "data/benchmarks/pmb_v0_full",
        "data/benchmarks/sft_personas_v0",
        "data/benchmarks/sft_personas_v1",
    ]
    for rel in must_state:
        if "human-reviewed" not in read_text(f"{rel}/DATASHEET.md"):
            return False
    return "subset" in read_text("data/distill/v1/DATASHEET.md") and "fixture" in read_text(
        "data/benchmarks/pmb_v0/DATASHEET.md"
    )


def datasheets_present() -> bool:
    registry = load_yaml(DATASETS_REGISTRY)
    missing = [
        row["datasetsheet"]
        for row in registry["datasets"]
        if row.get("datasetsheet") and not (ROOT / row["datasetsheet"]).is_file()
    ]
    return not missing


def dataset_registry_evidence_resolves() -> bool:
    """Every `evidence` anchor resolves and the line it names contains its `evidence_text`."""
    registry = load_yaml(DATASETS_REGISTRY)
    for entry in registry["datasets"]:
        anchors = entry.get("evidence") or {}
        needles = entry.get("evidence_text") or {}
        if set(anchors) != set(needles):
            return False
        for key, anchor in anchors.items():
            path_part, _, line_part = anchor.rpartition(":")
            if not path_part or not line_part.isdigit():
                return False
            path = ROOT / path_part
            if not path.is_file():
                return False
            lines = path.read_text(encoding="utf-8").splitlines()
            index = int(line_part) - 1
            if index >= len(lines) or needles[key] not in lines[index]:
                return False
        if not isinstance(entry.get("records"), int):
            return False
    return True


def training_contamination_status() -> bool:
    """Every training corpus carries a resolved contamination status, and it matches the check."""
    registry = load_yaml(DATASETS_REGISTRY)
    allowed = {
        "CLEAN_VERIFIED",
        "INHERITED_CLEAN",
        "EVAL_REFERENCE",
        "TRAINING_CORPUS",
        "NOT_ASSESSED",
        "NOT_APPLICABLE",
    }
    for entry in registry["datasets"]:
        status = entry.get("contamination_status")
        if status not in allowed:
            return False
        if entry["kind"] in ("training", "training_persona_corpus"):
            if status not in {"CLEAN_VERIFIED", "INHERITED_CLEAN", "TRAINING_CORPUS"}:
                return False
            if status == "CLEAN_VERIFIED" and not (entry.get("evidence") or {}).get(
                "contamination"
            ):
                return False
    if not contamination_clean_13gram():
        return False
    return all(
        entry["contamination_status"] in {"CLEAN_VERIFIED", "INHERITED_CLEAN"}
        for entry in registry["datasets"]
        if entry["id"] in {"sft-v0", "sft-v1", "dpo-v0", "dpo-v1-scale", "distill-v1"}
    )


def freeze_pinned_flags_match() -> bool:
    """registry/datasets.yaml's freeze_pinned agrees with what the freeze manifest pins."""
    manifest = freeze_manifest()
    pinned_paths = {path for group in manifest["evidence"].values() for path in group}
    registry = load_yaml(DATASETS_REGISTRY)
    for entry in registry["datasets"]:
        declared = bool(entry.get("freeze_pinned"))
        actual = any(path.startswith(entry["path"] + "/") for path in pinned_paths)
        if declared != actual:
            return False
    return True


def result_registry_covers_all_raw_runs() -> bool:
    """The index lists exactly the runs that have a raw.jsonl -- no more, no fewer."""
    expected = {
        path.parent.relative_to(ROOT).as_posix() for path in ROOT.glob("results/**/raw.jsonl")
    }
    indexed = {
        row["run_path"]
        for row in result_registry_rows()
        if (row.get("artifacts") or {}).get("raw_jsonl")
    }
    return expected == indexed


def result_registry_never_invents_metrics() -> bool:
    """A run without metrics.json has metrics: null -- never a value carried from elsewhere."""
    for row in result_registry_rows():
        has_metrics_file = bool((row.get("artifacts") or {}).get("metrics_json"))
        if has_metrics_file != (row.get("metrics") is not None):
            return False
        if not has_metrics_file and row.get("metrics_source") is not None:
            return False
        if row.get("metrics") and row.get("recomputed_matches_recorded") is False:
            return False
    return True


def sft_v1_by_kind_has_no_schema_field() -> bool:
    """The 'by kind' balance rests on a generation-time annotation absent from the corpus.

    The committed examples carry only `messages`, so the balance cannot be re-derived by reading
    them. This asserts the *reason* the DATASHEET claim is manual, and fails if a `kind` field is
    ever added -- at which point the claim becomes machine-checkable and this row should change.
    """
    rows = load_jsonl(ROOT / "data/sft/v1/train.jsonl")
    if not rows or "kind" in rows[0]:
        return False
    return set(rows[0]) == {"messages"}


def dpo_records_have_no_user_message_field() -> bool:
    """The DPO records carry one flattened `prompt`, so per-turn comparison is not reproducible."""
    rows = load_jsonl(ROOT / "data/dpo/v1_scale/train.jsonl")
    if not rows:
        return False
    return set(rows[0]) == {"prompt", "chosen", "rejected"} and isinstance(rows[0]["prompt"], str)


# NOTE: the single definition of `distill_val_sourced_from_train_split` lives with the other
# distill helpers below. It is intentionally not duplicated here: two definitions of a check is
# exactly the drift this repository has been bitten by (guardrail G4/G5).


# ------------------------------------------------------------------ registry-vs-artifact checks
def dataset_registry_counts_match() -> bool:
    registry = load_yaml(DATASETS_REGISTRY)
    for entry in registry["datasets"]:
        path = ROOT / entry["path"]
        declared = entry["records"]
        if path.is_file():
            actual = nonempty_line_count(path)
        elif entry["kind"] == "derived_store":
            actual = len([p for p in path.glob("*") if p.is_file()])
        else:
            actual = sum(nonempty_line_count(ROOT / split["path"]) for split in entry["splits"])
        if actual != declared:
            return False
        for split in entry.get("splits", []):
            split_path = ROOT / split["path"]
            if split_path.is_file() and split["records"] != nonempty_line_count(split_path):
                return False
            if split_path.is_dir() and split["records"] != len(
                [p for p in split_path.glob("*") if p.is_file()]
            ):
                return False
    return True


def dataset_registry_hashes_match() -> bool:
    module = import_script("recompute_hashes", "scripts/recompute_hashes.py")
    registry = load_yaml(DATASETS_REGISTRY)
    for entry in registry["datasets"]:
        hash_file = entry.get("hash_file")
        if not hash_file:
            continue
        directory = ROOT / entry["path"]
        if entry["hash_algorithm"] == "sha256_tree_lf":
            actual = module.benchmark_hash(directory)
        elif entry["hash_algorithm"] == "sha256_concat_jsonl":
            actual = module.dataset_hash(directory)
        else:
            return False
        if (ROOT / hash_file).read_text(encoding="utf-8").strip() != actual:
            return False
    return True


def benchmark_registry_counts_match() -> bool:
    if not pmb_design() or not pmb_per_category() or not pmb_answerable_split():
        return False
    registry = load_yaml(BENCHMARKS_REGISTRY)
    for entry in registry["benchmarks"]:
        if entry["id"] == "pmb_v0_full":
            expected = {
                "probes": len(pmb_rows()),
                "personas": pmb_persona_count(),
                "categories": pmb_category_count(),
                "probes_per_persona": pmb_probes_per_persona(),
                "answerable": 608,
                "unanswerable": 80,
                "populated_cells": 64,
                "total_cells": 64,
                "probes_per_category": _pmb_category_counts(),
            }
            for key, value in expected.items():
                if entry.get(key) != value:
                    return False
            # The unpopulated field is the reason pra_strict is ~0, so its zero must be real.
            if acceptable_alternatives_populated() != entry.get(
                "acceptable_alternatives_populated"
            ):
                return False
        probes = entry.get("probes")
        if probes is None:
            continue
        path = ROOT / entry["path"]
        if path.is_dir():
            path = path / "probes.jsonl"
        if path.is_file() and path.suffix == ".jsonl" and probes != len(load_jsonl(path)):
            return False
    return True


def _pmb_category_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in pmb_rows():
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def no_result_from_non_benchmark() -> bool:
    """No metric is attributed to a corpus that is not a benchmark.

    The training-persona corpora, the fixture, and the unrun H22/H24 probe sets must all be
    unscored, and no indexed run may point at one of them.
    """
    registry = load_yaml(BENCHMARKS_REGISTRY)
    unscored = {"pmb_v0", "sft_personas_v1", "sft_personas_v0", "h22_judgment", "emotional_range"}
    by_id = {entry["id"]: entry for entry in registry["benchmarks"]}
    for corpus in unscored:
        if corpus not in by_id or by_id[corpus].get("results_available"):
            return False
    for row in result_registry_rows():
        run_path = row.get("run_path") or ""
        if any(
            corpus in run_path for corpus in ("sft_personas", "h22_judgment", "emotional_range")
        ):
            return False
    return True


def studies_registry_matches_config() -> bool:
    studies = load_yaml(STUDIES_REGISTRY)
    study = next(s for s in studies["studies"] if s["id"] == "001")
    sha = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    return (
        study["base_model"]["revision"] == sha
        and study["base_model"]["revision_pinned_in"]
        == [
            "configs/training/sft.yaml",
            "configs/training/sft_v1.yaml",
        ]
        and study["evidence_tag"] in read_text("docs/STUDIES.md")
    )


def result_registry_is_fresh() -> bool:
    module = import_script("build_result_registry", "scripts/build_result_registry.py")
    if not RESULT_REGISTRY.is_file():
        return False
    return RESULT_REGISTRY.read_text(encoding="utf-8") == module.serialise(module.build())


def result_registry_hashes_match() -> bool:
    for row in result_registry_rows():
        artifacts = row.get("artifacts") or {}
        raw = artifacts.get("raw_jsonl")
        metrics = artifacts.get("metrics_json")
        summary = artifacts.get("summary_json")
        for artifact in (raw, metrics, summary):
            if not artifact:
                continue
            path = ROOT / artifact["path"]
            # LF-normalised, for the same cross-platform reason as imatrix_calibration_shape: the
            # registry digests are generated on one platform and verified on another.
            if not path.is_file() or sha256_lf_normalised(path) != artifact["sha256"]:
                return False
        # The index must not claim a metric it cannot recompute.
        if row.get("metrics") and not row.get("recomputed_matches_recorded"):
            return False
    return True


def guardrail_references_resolve() -> bool:
    """Every G<n> referenced anywhere in the repo is defined in docs/GUARDRAILS.md."""
    defined = {"G" + m for m in re.findall(r"\*\*G(\d+b?)\.", read_text("docs/GUARDRAILS.md"))}
    referenced: set[str] = set()
    for pattern in (
        "tests/**/*.py",
        "reports/**/*.md",
        "docs/**/*.md",
        "scripts/**/*.py",
        "src/**/*.py",
        "README.md",
    ):
        for path in ROOT.glob(pattern):
            if path.is_file():
                referenced |= {
                    "G" + m
                    for m in re.findall(
                        r"\bG(\d+b?)\b", read_text(path.relative_to(ROOT).as_posix())
                    )
                }
    return not (referenced - defined)


def guardrail_references_report() -> dict[str, list[str]]:
    defined = {"G" + m for m in re.findall(r"\*\*G(\d+b?)\.", read_text("docs/GUARDRAILS.md"))}
    referenced: set[str] = set()
    for pattern in (
        "tests/**/*.py",
        "reports/**/*.md",
        "docs/**/*.md",
        "scripts/**/*.py",
        "src/**/*.py",
        "README.md",
    ):
        for path in ROOT.glob(pattern):
            if path.is_file():
                referenced |= {
                    "G" + m
                    for m in re.findall(
                        r"\bG(\d+b?)\b", read_text(path.relative_to(ROOT).as_posix())
                    )
                }
    return {"dangling": sorted(referenced - defined), "undefined_ids": sorted(defined - referenced)}


def errata_ids_resolve() -> bool:
    """Every id in the audit findings ledger has an `E<n>` correction in reports/ERRATA.md."""
    findings = load_json(ROOT / "reports/audits/001-claim-audit/findings.json")
    text = read_text("reports/ERRATA.md")
    return all(f"E{finding['id']}" in text for finding in findings)


def errata_presents_every_finding() -> bool:
    """Every finding is *presented* in ERRATA.md, not merely mentioned in passing.

    A finding counts as presented when ERRATA gives it a section heading (`### E7 — ...`) or a row
    in the lower-severity table (`| E17 | ... |`). This is stricter than `errata_ids_resolve`, which
    only asks whether the string `E<n>` occurs somewhere: E31 was cited in a parenthetical for a
    while with no entry of its own, so the README's "31 claims ... and their corrections" pointed at
    a correction the document never made.
    """
    findings = load_json(ROOT / "reports/audits/001-claim-audit/findings.json")
    headings = set(re.findall(r"^###\s+(E\d+)\b", read_text("reports/ERRATA.md"), re.M))
    rows = set(re.findall(r"^\|\s*(E\d+)\s*\|", read_text("reports/ERRATA.md"), re.M))
    presented = headings | rows
    return {f"E{finding['id']}" for finding in findings} <= presented


def errata_count_matches_readme() -> bool:
    """README's "N claims the committed artifacts did not support" equals the ledger's size."""
    match = re.search(
        r"(\d+) claims the committed artifacts did not support", read_text("README.md")
    )
    if match is None:
        return False
    findings = load_json(ROOT / "reports/audits/001-claim-audit/findings.json")
    return int(match.group(1)) == len(findings)


def distill_val_sourced_from_train_split() -> bool:
    """Every distill v1 val prompt is present in the SFT v1 *train* split (224 of 224).

    Two of them also occur in the SFT v1 val split, because prompt text repeats across the corpus;
    that coincidence is why this check is stated as "all present in train" rather than "none present
    in val".
    """
    sft_train = {
        _prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/train.jsonl")
    }
    dst_val = [_prompt_key(row["prompt"]) for row in load_jsonl(ROOT / "data/distill/v1/val.jsonl")]
    return bool(dst_val) and all(key in sft_train for key in dst_val)


def distill_train_set_equals_sft_train_set() -> bool:
    """distill v1 (train + val) is exactly the SFT v1 train prompt set, as sets."""
    sft_train = {
        _prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/train.jsonl")
    }
    distill = {
        _prompt_key(row["prompt"])
        for split in ("train", "val")
        for row in load_jsonl(ROOT / f"data/distill/v1/{split}.jsonl")
    }
    return bool(distill) and distill == sft_train


def distill_subset_verified() -> bool:
    """The substance of the claim: every distill prompt comes from the SFT v1 train split.

    All 2008 train and all 224 val prompts are present in `data/sft/v1/train.jsonl`, and
    distill(train + val) equals the SFT v1 train prompt set. The count of distill val prompt texts
    that *also* occur in `data/sft/v1/val.jsonl` is reported separately by
    `distill_val_also_in_sft_val()`; see the DATASHEET claim in registry/claims.jsonl.
    """
    actual = distill_subset_actual()
    return (
        actual["distill_train_in_sft_train"] == actual["distill_train_total"] == 2008
        and actual["distill_val_in_sft_train"] == actual["distill_val_total"] == 224
        and distill_train_set_equals_sft_train_set()
    )


def distill_val_also_in_sft_val() -> int:
    """Distill v1 val prompt texts that also occur in the SFT v1 val split.

    Not zero, contrary to data/distill/v1/DATASHEET.md's "0 of them appear there". It is 2, caused
    by prompt-text duplication inside the SFT corpus itself (4 SFT val rows duplicate SFT train
    rows), not by how the distill prompt set was carved. The whole set still comes from train.jsonl.
    """
    sft_val = {_prompt_key(row["messages"]) for row in load_jsonl(ROOT / "data/sft/v1/val.jsonl")}
    return len(
        {_prompt_key(row["prompt"]) for row in load_jsonl(ROOT / "data/distill/v1/val.jsonl")}
        & sft_val
    )


def persona_fact_overlap_matches_registry() -> bool:
    """The measured training/eval fact-pool overlap equals what registry/benchmarks.yaml records."""
    registry = load_yaml(BENCHMARKS_REGISTRY)
    entry = next(e for e in registry["benchmarks"] if e["id"] == "sft_personas_v1")
    declared = entry.get("overlap_with_pmb") or {}
    measured = persona_fact_overlap()
    return (
        declared.get("fact_triples_shared") == measured["shared"]
        and declared.get("fact_triples_pmb_total") == measured["pmb_facts"]
        and declared.get("fact_triples_sft_total") == measured["sft_facts"]
        and declared.get("probe_question_rows_shared") == training_persona_probe_overlap()
    )


def studies_ladder_row(name: str) -> bool:
    """Look up a docs/STUDIES.md stage-table row and compare it to the indexed metrics."""
    row = next(
        (
            line
            for line in read_text("docs/STUDIES.md").splitlines()
            if line.startswith(f"| {name} ")
        ),
        None,
    )
    if row is None:
        return False
    metrics = None
    if name == "E (v1)":
        metrics = system_metrics("E_sft_memory_v1")
    elif name == "E-distill":
        metrics = system_metrics("E_distill_v1")
    if metrics is None:
        return False
    values = re.findall(r"([\d.]+)%", row)
    if len(values) < 2:
        return False
    return round(metrics["pra_lenient"] * 100, 2) == float(values[0].rstrip(".")) and round(
        metrics["uar"] * 100, 2
    ) == float(values[1].rstrip("."))


def studies_ladder_rows_verified() -> bool:
    """The v0.1 stage-table rows (A, B (v0), D, E (v0)) against the v0.1 artifacts."""
    expected = {
        "A": ("A_raw", 0.16, 13.75),
        "B (v0)": ("B_sft", 0.16, 16.25),
        "E (v0)": ("E_sft_memory", 17.76, 33.75),
    }
    lines = read_text("docs/STUDIES.md").splitlines()
    for label, (system, pra, uar) in expected.items():
        row = next((line for line in lines if line.startswith(f"| {label} ")), None)
        if row is None:
            return False
        values = re.findall(r"([\d.]+)%", row)
        if len(values) < 2 or float(values[0]) != pra or float(values[1]) != uar:
            return False
        metrics = system_metrics(system)
        if round(metrics["pra_lenient"] * 100, 2) != pra or round(metrics["uar"] * 100, 2) != uar:
            return False
    # D is intentionally not compared here: its quoted pra_lenient is an acknowledged drift.
    return True


def ladder_rows_agree(left: str, right: str) -> bool:
    """README and docs/STUDIES.md quote the same value for every one of the six ladder rows."""
    for spec in HEADLINE_QUOTES:
        values = []
        for source in spec["sources"]:
            if source["path"] not in (left, right):
                continue
            value = extract_one(source["path"], source["pattern"])
            if value is None:
                return False
            values.append(value)
        if len(values) != 2 or len({round(value, 4) for value in values}) != 1:
            return False
    return True


def extract_one(rel: str, pattern: str) -> float | None:
    match = re.search(pattern, read_text(rel))
    return None if match is None else float(match.group(1))


def metric_values_consistent() -> bool:
    """No headline metric is quoted with two values across README and docs/STUDIES.md.

    Requires, for every curated metric: exactly one quote in each document, the same value in both,
    and that value equal to the documented value in this module's table. Rounding is applied at the
    document's own precision, so 70.0 and 70.00 agree.
    """
    for spec in HEADLINE_QUOTES:
        values = []
        for source in spec["sources"]:
            matches = re.findall(source["pattern"], read_text(source["path"]))
            if len(matches) != 1:
                return False
            values.append(float(matches[0]))
        if len({round(value, 4) for value in values}) != 1:
            return False
        if not close(values[0], spec["value"], 1e-9):
            return False
    return True


# Report-only sweep of the *frozen* per-experiment write-ups: each pattern captures one
# pra_lenient percentage from a ladder row, at the precision that document uses, and the sweep
# compares it to the artifact. This is where the D_memory 15.10%-vs-15.13% drift lived (ERRATA
# E31), in documents that are frozen and therefore cannot be edited without a re-freeze. The
# curated cross-document check in `HEADLINE_QUOTES` covers README and docs/STUDIES.md; this covers
# the result write-ups, and reports rather than fails so a historical run record is not treated as
# a broken claim.
DRIFT_SWEEP: list[tuple[str, str, str, int]] = [
    ("docs/day3_memory_results.md", r"\| A \(raw, no memory\) \| [\d.]+% \| ([\d.]+)%", "A_raw", 2),
    (
        "docs/day3_memory_results.md",
        r"\| D \(raw \+ memory, k=8\) \| [\d.]+% \| ([\d.]+)%",
        "D_memory",
        1,
    ),
    ("docs/day4_sft_results.md", r"\| A \| raw, no memory \| [\d.]+% \| ([\d.]+)%", "A_raw", 2),
    ("docs/day4_sft_results.md", r"\| B \| SFT, no memory \| [\d.]+% \| ([\d.]+)%", "B_sft", 2),
    (
        "docs/day4_sft_results.md",
        r"\| D \| raw \+ memory \(k=8\) \| [\d.]+% \| ([\d.]+)%",
        "D_memory",
        2,
    ),
    (
        "docs/proper_scale_results.md",
        r"\| D \| raw \+ memory \(k=8\) \| ([\d.]+)%",
        "D_memory",
        2,
    ),
    (
        "docs/proper_scale_results.md",
        r"\| E \(v0\) \| SFT v0 \+ memory \| ([\d.]+)%",
        "E_sft_memory",
        2,
    ),
    (
        "docs/distillation_results.md",
        r"\| E \(v1, rebalanced\) \|[^|]*\| ([\d.]+)%",
        "E_sft_memory_v1",
        2,
    ),
    (
        "docs/distillation_results.md",
        r"E-distill \(H23\)\*\* \| \*\*[^|]*\*\* \| \*\*([\d.]+)%\*\*",
        "E_distill_v1",
        2,
    ),
]


def metric_drift_report() -> list[str]:
    """Report-only: any frozen write-up quote that disagrees with its artifact."""
    findings: list[str] = []
    for rel, pattern, system, decimals in DRIFT_SWEEP:
        match = re.search(pattern, read_text(rel))
        if match is None:
            findings.append(f"{rel}: sweep pattern for {system} matched nothing")
            continue
        quoted = float(match.group(1))
        actual = round(system_metrics(system)["pra_lenient"] * 100, decimals)
        if abs(quoted - actual) > 1e-9:
            findings.append(f"{rel}: quotes {system} pra_lenient as {quoted}% (artifact {actual}%)")
    return findings


# --------------------------------------------------------------------------------------------
# claim evaluation
# --------------------------------------------------------------------------------------------
class Finding:
    __slots__ = ("group", "claim_id", "ok", "message")

    def __init__(self, group: str, claim_id: str, ok: bool, message: str) -> None:
        self.group = group
        self.claim_id = claim_id
        self.ok = ok
        self.message = message


class Evaluator:
    """Restricted expression evaluator for `computed` checks."""

    _OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.USub: lambda a: -a,
        ast.Not: lambda a: not a,
    }

    def __init__(self, functions: dict[str, Callable[..., Any]]) -> None:
        self.functions = functions

    def evaluate(self, expr: str) -> Any:
        return self._eval(ast.parse(expr, mode="eval").body)

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [self._eval(item) for item in node.elts]
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.BoolOp):
            values = [self._eval(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = self._OPS[type(node.op)](result, value)
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator)
                if type(op) not in self._OPS or not self._OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in self.functions:
                raise NameError(f"unknown helper in computed expression: {name}()")
            if node.keywords:
                raise ValueError("computed expressions do not accept keyword arguments")
            args = [self._eval(arg) for arg in node.args]
            return self.functions[name](*args)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")


def _computed_functions() -> dict[str, Callable[..., Any]]:
    return {
        "split_counts": split_counts,
        "count_test_functions": count_test_functions,
        "test_function_count": count_test_functions,
        "numbered_item_count": numbered_item_count,
        "companion_persona_test_count": companion_persona_test_count,
        "pmb_persona_count": pmb_persona_count,
        "pmb_category_count": pmb_category_count,
        "pmb_probes_per_persona": pmb_probes_per_persona,
        "pmb_design": pmb_design,
        "pmb_answerable_split": pmb_answerable_split,
        "pmb_per_category": pmb_per_category,
        "pmb_unanswerable_has_no_gold": pmb_unanswerable_has_no_gold,
        "acceptable_alternatives_populated": acceptable_alternatives_populated,
        "all_pra_strict_near_zero": all_pra_strict_near_zero,
        "no_metrics_for_dpo_row": no_metrics_for_dpo_row,
        "c_vs_e_gap_pp": c_vs_e_gap_pp,
        "c_vs_f_gap_pp": c_vs_f_gap_pp,
        "h23_uplift_pp": h23_uplift_pp,
        "h23_confound_stated": h23_confound_stated,
        "c_vs_f_labeled_as_isolating": c_vs_f_labeled_as_isolating,
        "retrieval_effect_pp": retrieval_effect_pp,
        "training_config_revision_pinning": training_config_revision_pinning,
        "base_model_revision_claims": base_model_revision_claims,
        "config_values_match_claims": config_values_match_claims,
        "referenced_paths_exist": referenced_paths_exist,
        "hash_clean_corpus_count": hash_clean_corpus_count,
        "hash_corpus_count_with_hash_txt": hash_corpus_count_with_hash_txt,
        "freeze_file_count_matches_evidence": freeze_file_count_matches_evidence,
        "freeze_shape_claims": freeze_shape_claims,
        "freeze_check_passes": freeze_check_passes,
        "freeze_metadata_matches_studies_doc": freeze_metadata_matches_studies_doc,
        "benchmark_corpora_freeze_pinned": benchmark_corpora_freeze_pinned,
        "freeze_hash_gap_note_resolved": freeze_hash_gap_note_resolved,
        "distill_hash_claim_resolved": distill_hash_claim_resolved,
        "distill_subset_relation": distill_subset_relation,
        "distill_subset_actual": distill_subset_actual,
        "dpo_sft_prompt_overlap": dpo_sft_prompt_overlap,
        "training_persona_corpus_counts": training_persona_corpus_counts,
        "persona_fact_overlap": persona_fact_overlap,
        "training_personas_disjoint_from_pmb": training_personas_disjoint_from_pmb,
        "contamination_clean_13gram": contamination_clean_13gram,
        "emotional_range_registers": emotional_range_registers,
        "build_pmb_default_seed": build_pmb_default_seed,
        "imatrix_calibration_shape": imatrix_calibration_shape,
        "license_files_exist": license_files_exist,
        "no_quantization_artifact_committed": no_quantization_artifact_committed,
        "recomputed_run_count": recomputed_run_count,
        "reported_v0_rows_reconcile": reported_v0_rows_reconcile,
        "study_002_has_no_results": study_002_has_no_results,
        "human_review_notes_precise": human_review_notes_precise,
        "datasheets_present": datasheets_present,
        "dataset_registry_counts_match": dataset_registry_counts_match,
        "dataset_registry_hashes_match": dataset_registry_hashes_match,
        "dataset_registry_evidence_resolves": dataset_registry_evidence_resolves,
        "training_contamination_status": training_contamination_status,
        "freeze_pinned_flags_match": freeze_pinned_flags_match,
        "result_registry_covers_all_raw_runs": result_registry_covers_all_raw_runs,
        "result_registry_never_invents_metrics": result_registry_never_invents_metrics,
        "sft_v1_by_kind_has_no_schema_field": sft_v1_by_kind_has_no_schema_field,
        "dpo_records_have_no_user_message_field": dpo_records_have_no_user_message_field,
        "distill_val_sourced_from_train_split": distill_val_sourced_from_train_split,
        "training_persona_probe_overlap": training_persona_probe_overlap,
        "benchmark_registry_counts_match": benchmark_registry_counts_match,
        "no_result_from_non_benchmark": no_result_from_non_benchmark,
        "studies_registry_matches_config": studies_registry_matches_config,
        "result_registry_is_fresh": result_registry_is_fresh,
        "result_registry_hashes_match": result_registry_hashes_match,
        "guardrail_references_resolve": guardrail_references_resolve,
        "errata_ids_resolve": errata_ids_resolve,
        "errata_presents_every_finding": errata_presents_every_finding,
        "errata_count_matches_readme": errata_count_matches_readme,
        "distill_train_set_equals_sft_train_set": distill_train_set_equals_sft_train_set,
        "distill_subset_verified": distill_subset_verified,
        "distill_val_also_in_sft_val": distill_val_also_in_sft_val,
        "persona_fact_overlap_matches_registry": persona_fact_overlap_matches_registry,
        "studies_ladder_row": studies_ladder_row,
        "studies_ladder_rows_verified": studies_ladder_rows_verified,
        "ladder_rows_agree": ladder_rows_agree,
        "metric_values_consistent": metric_values_consistent,
        "extract_one": extract_one,
    }


def evaluate_check(check: dict[str, Any], evaluator: Evaluator) -> tuple[bool, str]:
    kind = check.get("type")
    if kind == "manual":
        return True, f"manual: {check.get('reason', '(no reason given)')}"
    if kind == "file_exists":
        path = ROOT / check["path"]
        expected = check.get("equals", True)
        actual = path.exists()
        return actual is expected, f"{check['path']} exists={actual}, expected {expected}"
    if kind == "json_field":
        path = ROOT / check["path"]
        if not path.is_file():
            return False, f"{check['path']} does not exist"
        document = load_document(path)
        try:
            actual = walk_dotted(document, check["field"])
        except (KeyError, IndexError, TypeError) as exc:
            return False, f"{check['path']}: cannot read field {check['field']!r} ({exc})"
        tolerance = float(check.get("tolerance", TOLERANCE))
        if "equals" in check and not close(actual, check["equals"], tolerance):
            return (
                False,
                f"{check['path']}:{check['field']} = {actual!r}, expected {check['equals']!r}",
            )
        if "decimals" in check and "quoted" in check:
            rounded = round(float(actual) * 100, int(check["decimals"]))
            if not close(rounded, check["quoted"], 1e-9):
                return (
                    False,
                    f"{check['path']}:{check['field']} = {actual!r} -> "
                    f"{rounded}%, but the document quotes {check['quoted']}%",
                )
            return True, f"{check['path']}:{check['field']} = {actual!r} ({rounded}%)"
        return True, f"{check['path']}:{check['field']} = {actual!r}"
    if kind == "count_lines":
        total = 0
        matches = sorted(ROOT.glob(check["glob"]))
        if not matches:
            return False, f"{check['glob']} matched no files"
        for path in matches:
            total += nonempty_line_count(path)
        return (
            total == check["equals"],
            f"{check['glob']} -> {total} non-empty lines, expected {check['equals']}",
        )
    if kind == "computed":
        try:
            result = evaluator.evaluate(check["expr"])
        except Exception as exc:  # noqa: BLE001 - reported as a check failure, not a crash
            return False, f"expr {check['expr']!r} raised {type(exc).__name__}: {exc}"
        if "equals" in check:
            tolerance = float(check.get("tolerance", TOLERANCE))
            return (
                close(result, check["equals"], tolerance),
                f"{check['expr']} -> {result!r}, expected {check['equals']!r}",
            )
        return bool(result), f"{check['expr']} -> {result!r}"
    if kind == "cross_doc":
        observed = []
        for source in check["sources"]:
            match = re.search(source["pattern"], read_text(source["path"]))
            if match is None:
                return False, f"{source['path']}: pattern {source['pattern']!r} matched nothing"
            observed.append((source["path"], float(match.group(1))))
        expected = float(check["equals"])
        bad = [f"{path}={value}" for path, value in observed if not close(value, expected, 1e-9)]
        return (
            not bad,
            f"quoted {observed} vs expected {expected}" if bad else f"all sources quote {expected}",
        )
    return False, f"unknown check type {kind!r}"


def check_claims(evaluator: Evaluator) -> tuple[list[Finding], list[Finding], list[Finding]]:
    """Returns (findings, manual, acknowledged_drift)."""
    findings: list[Finding] = []
    manual: list[Finding] = []
    drift: list[Finding] = []

    raw_lines = [line for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids: set[str] = set()
    for lineno, line in enumerate(raw_lines, 1):
        try:
            claim = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(Finding("claims", f"line-{lineno}", False, f"invalid JSON: {exc}"))
            continue
        missing = [field for field in REQUIRED_CLAIM_FIELDS if field not in claim]
        if missing:
            findings.append(
                Finding(
                    "claims", claim.get("id", f"line-{lineno}"), False, f"missing fields: {missing}"
                )
            )
            continue
        claim_id = claim["id"]
        if claim_id in seen_ids:
            findings.append(Finding("claims", claim_id, False, "duplicate claim id"))
        seen_ids.add(claim_id)
        status = claim["status"]
        if status not in VALID_STATUSES:
            findings.append(Finding("claims", claim_id, False, f"unknown status {status!r}"))
            continue
        if not re.match(r"^[^:]+:\d+", claim["where"]):
            findings.append(
                Finding(
                    "claims", claim_id, False, f"`where` must be file:line, got {claim['where']!r}"
                )
            )
        if status == "manual":
            if claim["check"].get("type") != "manual":
                findings.append(
                    Finding("claims", claim_id, False, "status manual but check.type is not manual")
                )
                continue
            manual.append(
                Finding(
                    "manual",
                    claim_id,
                    True,
                    f"{claim['where']} -- {claim['check'].get('reason', '')}",
                )
            )
            continue
        if status == "acknowledged_drift":
            for field in REQUIRED_DRIFT_FIELDS:
                if not claim.get(field):
                    findings.append(
                        Finding(
                            "claims",
                            claim_id,
                            False,
                            f"acknowledged_drift requires a non-empty {field!r}",
                        )
                    )
            passed, detail = evaluate_check(claim["check"], evaluator)
            if passed:
                findings.append(
                    Finding(
                        "claims",
                        claim_id,
                        False,
                        "recorded as acknowledged_drift but the check now passes -- "
                        "the claims matrix is stale: " + detail,
                    )
                )
            else:
                drift.append(
                    Finding(
                        "drift",
                        claim_id,
                        True,
                        f"{claim['where']}: {detail} :: {claim.get('resolution', '')}",
                    )
                )
            continue
        # status == verified
        passed, detail = evaluate_check(claim["check"], evaluator)
        findings.append(
            Finding(
                "claims", claim_id, passed, f"{claim['where']}: {detail}" if not passed else detail
            )
        )
    return findings, manual, drift


# --------------------------------------------------------------------------------------------
# section checks
# --------------------------------------------------------------------------------------------
def check_freeze() -> list[Finding]:
    findings: list[Finding] = []
    if not FREEZE_MANIFEST.is_file():
        return [Finding("freeze", "manifest", False, f"{FREEZE_MANIFEST} missing")]
    manifest = freeze_manifest()
    problems: list[str] = []
    if not freeze_check_passes():
        module = import_script("freeze_study_001", "scripts/freeze_study_001.py")
        fresh = module.build()
        for group in sorted(set(manifest["evidence"]) | set(fresh["evidence"])):
            recorded = manifest["evidence"].get(group, {})
            actual = fresh["evidence"].get(group, {})
            for path in sorted(set(recorded) | set(actual)):
                if recorded.get(path) != actual.get(path):
                    problems.append(
                        f"{group}/{path}: recorded {recorded.get(path)} != now {actual.get(path)}"
                    )
    if manifest.get("file_count") != sum(len(g) for g in manifest["evidence"].values()):
        problems.append("file_count disagrees with the recorded evidence")
    findings.append(
        Finding(
            "freeze",
            "study-001",
            not problems,
            (
                "; ".join(problems)
                if problems
                else f"{manifest['file_count']} artifacts match (frozen {manifest['frozen_on']})"
            ),
        )
    )
    return findings


def check_hashes() -> list[Finding]:
    module = import_script("recompute_hashes", "scripts/recompute_hashes.py")
    problems: list[str] = []
    clean = 0
    targets = [(d, module.benchmark_hash) for d in BENCHMARK_HASH_DIRS] + [
        (d, module.dataset_hash) for d in DATASET_HASH_DIRS
    ]
    # The declared lists here and in the script must not drift apart.
    if sorted(module.BENCHMARK_DIRS) != sorted(BENCHMARK_HASH_DIRS):
        problems.append(
            "BENCHMARK_HASH_DIRS disagrees with scripts/recompute_hashes.BENCHMARK_DIRS"
        )
    if sorted(module.DATASET_DIRS) != sorted(DATASET_HASH_DIRS):
        problems.append("DATASET_HASH_DIRS disagrees with scripts/recompute_hashes.DATASET_DIRS")
    for rel, fn in targets:
        directory = ROOT / rel
        if not directory.is_dir():
            problems.append(f"{rel}: missing directory")
            continue
        hash_file = directory / "hash.txt"
        if not hash_file.is_file():
            problems.append(f"{rel}: no hash.txt")
            continue
        stored = hash_file.read_text(encoding="utf-8").strip()
        actual = fn(directory)
        if stored == actual:
            clean += 1
        else:
            problems.append(f"{rel}: recorded {stored[:12]}... != actual {actual[:12]}...")
    findings = [
        Finding(
            "hashes",
            "corpora",
            not problems,
            "; ".join(problems) if problems else f"{clean}/{len(targets)} corpora hash-clean",
        )
    ]
    unpinned = [
        rel
        for rel in ["data/benchmarks/emotional_range", "data/benchmarks/h22_judgment"]
        if not (ROOT / rel / "hash.txt").is_file()
    ]
    findings.append(
        Finding(
            "hygiene",
            "unpinned-probe-sets",
            not unpinned,
            (
                f"no hash.txt (not integrity-pinned): {unpinned}"
                if unpinned
                else "all probe sets pinned"
            ),
        )
    )
    return findings


def check_metrics() -> list[Finding]:
    categories = probe_categories()
    findings: list[Finding] = []
    for raw_path in sorted(ROOT.glob("results/**/raw.jsonl")):
        run_dir = raw_path.parent
        metrics_path = run_dir / "metrics.json"
        rel = run_dir.relative_to(ROOT).as_posix()
        if not metrics_path.is_file():
            continue
        rows = load_jsonl(raw_path)
        if not rows or "probe" not in rows[0]:
            continue
        recorded = load_json(metrics_path)["metrics"]
        unanswerable = [r for r in rows if categories.get(r["probe"]["probe_id"]) == "unanswerable"]
        answerable = [r for r in rows if categories.get(r["probe"]["probe_id"]) != "unanswerable"]
        uar = sum(1 for r in unanswerable if r.get("abstained")) / len(unanswerable)
        pra_lenient = sum(1 for r in answerable if r.get("lenient_correct")) / len(answerable)
        pra_strict = sum(1 for r in answerable if r.get("strict_correct")) / len(answerable)
        mismatches = [
            f"{name}: recomputed {value} != metrics.json {recorded.get(name)}"
            for name, value in (
                ("uar", uar),
                ("pra_lenient", pra_lenient),
                ("pra_strict", pra_strict),
            )
            if not close(recorded.get(name), value)
        ]
        findings.append(
            Finding(
                "metrics",
                rel,
                not mismatches,
                (
                    "; ".join(mismatches)
                    if mismatches
                    else f"{len(rows)} responses recompute to metrics.json exactly"
                ),
            )
        )
    return findings


def check_registries() -> list[Finding]:
    specs = [
        ("datasets.yaml counts", dataset_registry_counts_match),
        ("datasets.yaml hashes", dataset_registry_hashes_match),
        ("benchmarks.yaml counts", benchmark_registry_counts_match),
        ("benchmarks.yaml not-a-benchmark rule", no_result_from_non_benchmark),
        ("studies.yaml model revision", studies_registry_matches_config),
        ("studies.yaml Study 002 has no results", study_002_has_no_results),
        ("results/registry.jsonl freshness", result_registry_is_fresh),
        ("results/registry.jsonl digests", result_registry_hashes_match),
    ]
    findings: list[Finding] = []
    for name, fn in specs:
        try:
            ok = bool(fn())
            findings.append(Finding("registries", name, ok, "ok" if ok else "failed"))
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("registries", name, False, f"{type(exc).__name__}: {exc}"))
    return findings


def check_crossdoc() -> list[Finding]:
    findings: list[Finding] = []
    ok = metric_values_consistent()
    findings.append(
        Finding(
            "crossdoc",
            "headline-metrics-agree",
            ok,
            (
                "every headline metric is quoted with one value in README.md and docs/STUDIES.md"
                if ok
                else "README.md and docs/STUDIES.md disagree about a headline metric"
            ),
        )
    )
    agree = ladder_rows_agree("README.md", "docs/STUDIES.md")
    findings.append(
        Finding(
            "crossdoc",
            "ladder-rows-agree",
            agree,
            (
                "README and docs/STUDIES.md quote identical ladder values"
                if agree
                else "the README ladder and the docs/STUDIES.md stage table disagree"
            ),
        )
    )
    dangling = guardrail_references_report()
    findings.append(
        Finding(
            "crossdoc",
            "guardrail-references",
            not dangling["dangling"],
            (
                f"undefined guardrail ids referenced: {dangling['dangling']}"
                if dangling["dangling"]
                else "every referenced guardrail id is defined in docs/GUARDRAILS.md"
            ),
        )
    )
    return findings


def check_hygiene() -> list[Finding]:
    findings: list[Finding] = []
    unpinned: list[str] = []
    for config in sorted((ROOT / "configs" / "training").glob("*.yaml")):
        for key in ("base_model_revision", "teacher_model_revision"):
            value = load_document(config).get(key)
            if value is not None and not re.fullmatch(r"[0-9a-f]{40}", str(value)):
                unpinned.append(f"{config.name}:{key}={value}")
    findings.append(
        Finding(
            "hygiene",
            "config-revisions",
            not unpinned,
            (
                f"{len(unpinned)} revision pin(s) are not a 40-char SHA: {unpinned}"
                if unpinned
                else "every config revision is a 40-char SHA"
            ),
        )
    )
    registry = load_yaml(DATASETS_REGISTRY)
    missing_hash = [
        entry["id"]
        for entry in registry["datasets"]
        if entry.get("kind") in ("training", "benchmark", "training_persona_corpus")
        and not entry.get("hash_file")
    ]
    findings.append(
        Finding(
            "hygiene",
            "dataset-hashes",
            not missing_hash,
            (
                f"corpora with no hash.txt: {missing_hash}"
                if missing_hash
                else "every corpus has a hash.txt"
            ),
        )
    )
    drift = metric_drift_report()
    findings.append(
        Finding(
            "hygiene",
            "documented-vs-artifact-metrics",
            not drift,
            (
                "; ".join(drift)
                if drift
                else "no document quote of a headline metric disagrees with its artifact"
            ),
        )
    )
    facts = persona_fact_overlap()
    findings.append(
        Finding(
            "hygiene",
            "persona-fact-pool-overlap",
            # A measured quantity, not a defect: it is the number behind the `manual` status on
            # README's "generated disjoint" claim, so it is reported and never fatal.
            True,
            f"pmb_v0_full and sft_personas_v1 share {facts['shared']}/{facts['pmb_facts']} "
            "(predicate, object, category) facts",
        )
    )
    return findings


def check_pollution() -> list[Finding]:
    try:
        ok = contamination_clean_13gram()
    except Exception as exc:  # noqa: BLE001
        return [Finding("pollution", "13-gram", False, f"{type(exc).__name__}: {exc}")]
    return [
        Finding(
            "pollution",
            "13-gram",
            ok,
            (
                "no training corpus shares a 13-gram with pmb_v0_full/probes.jsonl"
                if ok
                else "a training corpus shares a 13-gram with the PMB benchmark"
            ),
        )
    ]


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------
def run(
    include_pollution: bool, strict_hygiene: bool
) -> tuple[list[Finding], list[Finding], list[Finding]]:
    evaluator = Evaluator(_computed_functions())
    findings: list[Finding] = []
    findings += check_freeze()
    findings += check_hashes()
    findings += check_metrics()
    findings += check_registries()
    claim_findings, manual, drift = check_claims(evaluator)
    findings += claim_findings
    findings += check_crossdoc()
    hygiene = check_hygiene()
    if include_pollution:
        findings += check_pollution()
    if strict_hygiene:
        # The hygiene findings carry `ok=False` exactly when there is something to report, so in
        # strict mode the documented gaps (unpinned config revisions, corpora with no hash.txt)
        # fail the run. That is the point of the flag: `make validate` passes on the repository as
        # it stands, `--strict` tells you how far it is from fully deterministic.
        findings += hygiene
    else:
        manual = manual + [Finding("hygiene", f.claim_id, f.ok, f.message) for f in hygiene]
    return findings, manual, drift


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="make determinism-hygiene findings fatal as well",
    )
    parser.add_argument(
        "--no-contamination",
        action="store_true",
        help="skip the 13-gram contamination check (the slowest check)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    findings, manual, drift = run(
        include_pollution=not args.no_contamination, strict_hygiene=args.strict
    )
    failures = [f for f in findings if not f.ok]

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not failures,
                    "failures": [
                        {"group": f.group, "id": f.claim_id, "message": f.message} for f in failures
                    ],
                    "manual": [{"id": f.claim_id, "message": f.message} for f in manual],
                    "acknowledged_drift": [{"id": f.claim_id, "message": f.message} for f in drift],
                    "checks": len(findings) + len(manual) + len(drift),
                },
                indent=2,
            )
        )
        return 1 if failures else 0

    groups: dict[str, list[Finding]] = {}
    for finding in findings:
        groups.setdefault(finding.group, []).append(finding)

    print(f"validate: auditing {ROOT}")
    for group in sorted(groups):
        entries = groups[group]
        bad = [f for f in entries if not f.ok]
        status = "FAIL" if bad else "ok"
        print(f"\n[{status}] {group} ({len(entries) - len(bad)}/{len(entries)} passed)")
        for finding in entries:
            mark = "  x" if not finding.ok else "  ."
            if finding.ok and group == "hygiene":
                print(f"{mark} {finding.claim_id}: {finding.message}")
            elif not finding.ok:
                print(f"{mark} {finding.claim_id}: {finding.message}")

    if drift:
        print(
            f"\n[acknowledged drift] {len(drift)} claim(s) that do not hold, recorded not corrected"
        )
        for finding in drift:
            print(f"  ! {finding.claim_id}: {finding.message}")

    if manual:
        hygiene_notes = [f for f in manual if f.group == "hygiene"]
        manual_claims = [f for f in manual if f.group == "manual"]
        if manual_claims:
            print(f"\n[manual] {len(manual_claims)} claim(s) not machine-checkable")
            for finding in manual_claims:
                print(f"  ? {finding.claim_id}: {finding.message[:160]}")
        if hygiene_notes:
            print(
                f"\n[hygiene] {len(hygiene_notes)} report(s), non-fatal (use --strict to enforce)"
            )
            for finding in hygiene_notes:
                print(f"  i {finding.claim_id}: {finding.message[:200]}")

    total = len(findings) + len(manual) + len(drift)
    print(
        f"\nSUMMARY: {total} checks, {len(failures)} failed, {len(drift)} acknowledged drift, "
        f"{len([f for f in manual if f.group == 'manual'])} manual"
    )
    if failures:
        print("FAIL: the repository does not match its own claims.")
        for finding in failures:
            print(f"  {finding.group}/{finding.claim_id}")
        return 1
    print("OK: every machine-checkable claim matches its artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
