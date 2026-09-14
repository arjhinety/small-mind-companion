#!/usr/bin/env python
"""Build ``results/registry.jsonl`` -- a derived index of every evaluated run.

One JSON object per line, one line per evaluated run, derived entirely from committed artifacts.
This file is a *view*, not a source of truth: it holds no number that is not already in the
``metrics.json`` or ``raw.jsonl`` it points at, and ``scripts/validate.py`` reports it stale if a
regeneration would differ.

Adapted from OpenGrad's ``results/registry.jsonl`` convention (machine-readable derived index over
authoritative artifacts), scoped to this repository's run model: a run is a directory under
``results/`` that carries ``raw.jsonl``, and optionally ``metrics.json``.

Usage::

    uv run python scripts/build_result_registry.py            # write the index
    uv run python scripts/build_result_registry.py --check     # fail if it is stale

Runs on CPU, offline, and reads no model. ``pra_lenient``/``uar``/``pra_strict`` are recomputed
from ``raw.jsonl`` rather than copied from ``metrics.json``, so a mismatch between the two makes
the two digests in the entry disagree with the recomputed metrics -- which is exactly what
``scripts/validate.py`` reports.

The DPO-lineage pairwise runs carry no ``metrics.json`` and, deliberately, no fabricated metric
values: their ``metrics`` object is ``null`` and they expose ``pairwise`` instead. There is no
full-PMB ``pra_lenient``/``uar`` measurement for any DPO checkpoint, and this index must not imply
one exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "results" / "registry.jsonl"
SCHEMA_VERSION = 1
SCHEMA_NOTE = (
    "Derived index over results/**. Regenerate with scripts/build_result_registry.py; "
    "scripts/validate.py fails if it is stale or if any digest disagrees with the file. "
    "No field here is hand-written."
)

# Which Study each run belongs to. Study 001's evaluated ladder is frozen; anything added later is
# Study 002 and is marked as such rather than being folded into the frozen ladder.
STUDY_BY_RUN_PREFIX: list[tuple[str, str]] = [
    ("results/v0.0/", "001"),
    ("results/v0.1/", "001"),
    ("results/v1_scale/", "001"),
    ("results/v2", "002"),
]

# The run's own name for the system under test does not always match the label the write-ups use.
# This map is the machine-readable form of the attribution rule in docs/GUARDRAILS.md G1 ("a
# result's label comes from the script that produced it"). Every entry states what the label is
# and, where the label is a correction, which errata entry established it.
SYSTEM_LABELS: dict[str, dict[str, Any]] = {
    "results/v0.1/A_raw": {
        "ladder_row": "A",
        "label": "raw model, no memory",
        "checkpoint": "google/gemma-4-E2B-it",
        "memory": False,
        "sft": False,
        "dpo": False,
        "distill": False,
    },
    "results/v0.1/B_sft": {
        "ladder_row": "B (v0)",
        "label": "LoRA SFT v0, no memory",
        "checkpoint": "outputs/sft/v0/merged",
        "memory": False,
        "sft": True,
        "dpo": False,
        "distill": False,
    },
    "results/v0.1/D_memory": {
        "ladder_row": "D",
        "label": "raw + hybrid retrieval memory (k=8)",
        "checkpoint": "google/gemma-4-E2B-it",
        "memory": True,
        "sft": False,
        "dpo": False,
        "distill": False,
    },
    "results/v0.1/E_sft_memory": {
        "ladder_row": "E (v0)",
        "label": "LoRA SFT v0 + memory",
        "checkpoint": "outputs/sft/v0/merged",
        "memory": True,
        "sft": True,
        "dpo": False,
        "distill": False,
    },
    "results/v1_scale/B_sft": {
        "ladder_row": "B (v1)",
        "label": "proper-scale LoRA SFT v1, no memory",
        "checkpoint": "outputs/sft/v1/merged",
        "memory": False,
        "sft": True,
        "dpo": False,
        "distill": False,
    },
    "results/v1_scale/E_sft_memory": {
        "ladder_row": "E (v1)",
        "label": "proper-scale LoRA SFT v1 + memory",
        "checkpoint": "outputs/sft/v1/merged",
        "memory": True,
        "sft": True,
        "dpo": False,
        "distill": False,
        # G1/E1: this artifact is SFT-only. It is NOT a DPO result, however it has been quoted.
        "attribution_note": (
            "SFT v1 + memory. The 70.0% UAR / 15.30% pra_lenient figures belong here and not to "
            "any DPO checkpoint (reports/ERRATA.md E1)."
        ),
    },
    "results/v1_scale/E_distill": {
        "ladder_row": "E-distill",
        "label": "SFT v1 + DPO v1_scale + on-policy distillation + memory",
        "checkpoint": "outputs/distill/v1/merged",
        "memory": True,
        "sft": True,
        "dpo": True,
        "distill": True,
        "attribution_note": (
            "Chains off the DPO output, so a comparison against E_sft_memory spans the DPO stage "
            "as well as distillation (reports/ERRATA.md E2). The isolating measurement is "
            "results/v1_scale/C_vs_F_distill_pairwise."
        ),
    },
    "results/v1_scale/C_vs_E_pairwise": {
        "ladder_row": "C-vs-E",
        "label": "pairwise: C (DPO v1_scale + memory) vs E (SFT v1 + memory)",
        "memory": True,
        "sft": True,
        "dpo": True,
        "distill": False,
        "pairwise_sides": {"c": "dpo v1_scale + memory", "e": "sft v1 + memory"},
    },
    "results/v1_scale/C_vs_F_distill_pairwise": {
        "ladder_row": "C-vs-F",
        "label": "pairwise: C (DPO v1_scale + memory) vs F (distill v1 + memory)",
        "memory": True,
        "sft": True,
        "dpo": True,
        "distill": True,
        "pairwise_sides": {"c": "dpo v1_scale + memory", "f": "distill v1 + memory"},
        "attribution_note": (
            "Both sides carry the DPO stage, so this is the isolating measurement for H23."
        ),
    },
    "results/v0.1/C_vs_E_pairwise": {
        "ladder_row": None,
        "label": "pairwise (v0): C (DPO v0 + memory) vs E (SFT v0 + memory)",
        "memory": True,
        "sft": True,
        "dpo": True,
        "distill": False,
        "pairwise_sides": {"c": "dpo v0 + memory", "e": "sft v0 + memory"},
    },
    "results/v0.1/C1_vs_E_pairwise_small": {
        "ladder_row": None,
        "label": "pairwise (v0, small): C1 vs E, 35 probes",
        "memory": True,
        "sft": True,
        "dpo": True,
        "distill": False,
        "pairwise_sides": {"c1": "dpo v0 (small)", "e": "sft v0 + memory"},
    },
}

KSWEEP_LABELS = {
    0: "raw + memory (k=0)",
    2: "raw + memory (k=2)",
    4: "raw + memory (k=4)",
    8: "raw + memory (k=8)",
    16: "raw + memory (k=16)",
}

# Only these runs are quoted in the Study 001 write-ups. Everything else in results/ is either a
# v0 exploratory run or a superseded small pairwise pass; the index records all of them so the
# coverage is visible, and marks which ones a document actually cites.
CITED_IN_WRITEUPS = {
    "results/v0.1/A_raw",
    "results/v0.1/B_sft",
    "results/v0.1/D_memory",
    "results/v0.1/E_sft_memory",
    "results/v0.1/ksweep/k0",
    "results/v0.1/ksweep/k2",
    "results/v0.1/ksweep/k4",
    "results/v0.1/ksweep/k8",
    "results/v0.1/ksweep/k16",
    "results/v1_scale/B_sft",
    "results/v1_scale/E_sft_memory",
    "results/v1_scale/E_distill",
    "results/v1_scale/C_vs_E_pairwise",
    "results/v1_scale/C_vs_F_distill_pairwise",
    "results/v0.1/C_vs_E_pairwise",
    "results/v1_scale/pcs_judge_analysis",
    "results/v1_scale/pcs_stylometric_analysis",
}

# The seven runs whose metrics.json the claim audit recomputed and matched exactly.
AUDIT_RECOMPUTED_RUNS = [
    "results/v0.1/A_raw",
    "results/v0.1/B_sft",
    "results/v0.1/D_memory",
    "results/v0.1/E_sft_memory",
    "results/v1_scale/B_sft",
    "results/v1_scale/E_sft_memory",
    "results/v1_scale/E_distill",
]

RUN_KINDS = {
    "results/v0.0/": "bakeoff",
    "results/v0.1/ksweep/": "k_sweep",
    "results/v1_scale/pcs_": "persona_consistency",
    "results/v0.1/pcs_": "persona_consistency",
    ":_pairwise": "pairwise",
}


def sha256_file(path: Path) -> str:
    """SHA-256 over LF-normalised bytes.

    The digests this writes are verified on a different machine from the one that generated them,
    and this repo has no `.gitattributes` while setting `core.autocrlf=true`. Hashing raw bytes
    produces an index that is fresh on the platform that built it and stale on every other -- which
    is exactly how this check passed locally and failed in CI.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def probe_categories() -> dict[str, str]:
    pmb = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
    return {row["probe_id"]: row["category"] for row in load_jsonl(pmb)}


def recompute_metrics(raw_path: Path, categories: dict[str, str]) -> dict[str, Any]:
    """Recompute uar / pra_lenient / pra_strict from saved responses.

    This mirrors the audit's recomputation and ``tests/unit/test_study_001_freeze.py``. Answerable
    and unanswerable probes are split on the probe's own PMB category, not on anything the response
    records.
    """
    rows = load_jsonl(raw_path)
    unanswerable = [r for r in rows if categories.get(r["probe"]["probe_id"]) == "unanswerable"]
    answerable = [r for r in rows if categories.get(r["probe"]["probe_id"]) != "unanswerable"]
    return {
        "uar": (
            (sum(1 for r in unanswerable if r.get("abstained")) / len(unanswerable))
            if unanswerable
            else None
        ),
        "pra_lenient": (
            (sum(1 for r in answerable if r.get("lenient_correct")) / len(answerable))
            if answerable
            else None
        ),
        "pra_strict": (
            (sum(1 for r in answerable if r.get("strict_correct")) / len(answerable))
            if answerable
            else None
        ),
    }


def pairwise_summary(path: Path, run_dir: Path) -> dict[str, Any] | None:
    summary = run_dir / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    rows = load_jsonl(path)
    scores = [r.get("dual_order_score") for r in rows if r.get("dual_order_score") is not None]
    if not scores:
        return None
    return {
        "n": len(scores),
        "mean_dual_order_score": sum(scores) / len(scores),
        "derived": "mean of raw.jsonl dual_order_score (no summary.json committed)",
    }


def run_kind(rel: str) -> str:
    if "pairwise" in rel:
        return "pairwise"
    if "ksweep" in rel:
        return "k_sweep"
    if "pcs_" in rel:
        return "persona_consistency"
    return "pmb_full_harness"


def study_for(rel: str) -> str | None:
    for prefix, study in STUDY_BY_RUN_PREFIX:
        if rel.startswith(prefix):
            return study
    return None


def metrics_artifact_kind(rel: str) -> str:
    run_dir = ROOT / rel
    if (run_dir / "metrics.json").is_file():
        return "pmb_metrics"
    if (run_dir / "summary.json").is_file():
        return "summary"
    return "none"


def build_entry(rel: str, categories: dict[str, str]) -> dict[str, Any] | None:
    run_dir = ROOT / rel
    raw_path = run_dir / "raw.jsonl"
    if not raw_path.is_file():
        return None

    metrics_path = run_dir / "metrics.json"
    recorded: dict[str, Any] | None = None
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        recorded = payload.get("metrics")
        system_name = payload.get("system")
        n_probes = payload.get("n_probes")
    else:
        system_name = None
        n_probes = None

    # Pairwise runs record one row per probe pair (``probe_id`` at the top level) and have no
    # metrics.json. They are never given a pra_lenient/uar value here.
    recomputed = recompute_metrics(raw_path, categories) if recorded is not None else None
    kind = run_kind(rel)
    label_info = SYSTEM_LABELS.get(rel, {})

    if "ksweep" in rel:
        k = int(rel.rsplit("/k", 1)[1])
        label = KSWEEP_LABELS.get(k, f"raw + memory (k={k})")
        system_name = system_name or f"ksweep_k{k}"
        label_info = {
            "label": label,
            "ladder_row": None,
            "memory": k > 0,
            "sft": False,
            "dpo": False,
            "distill": False,
        }

    metrics: dict[str, Any] | None
    if recorded is not None:
        metrics = {
            "pra_lenient": recorded.get("pra_lenient"),
            "uar": recorded.get("uar"),
            "pra_strict": recorded.get("pra_strict"),
            "mur": recorded.get("mur"),
        }
        metrics_source = "metrics.json"
    else:
        # Drives the index's central negative fact: a DPO-lineage run with no metrics.json has no
        # full-PMB metric, and this file must not invent one.
        metrics = None
        metrics_source = None

    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_path": rel,
        "run_kind": kind,
        "study": study_for(rel),
        "system_name": system_name,
        "label": label_info.get("label"),
        "ladder_row": label_info.get("ladder_row"),
        "cited_in_writeups": rel in CITED_IN_WRITEUPS,
        "recomputed_in_claim_audit": rel in AUDIT_RECOMPUTED_RUNS,
        "n_probes": n_probes if n_probes is not None else len(load_jsonl(raw_path)),
        "metrics_artifact_kind": metrics_artifact_kind(rel),
        "metrics": metrics,
        "metrics_source": metrics_source,
        "recomputed_from_raw": recomputed,
        "recomputed_matches_recorded": (
            None
            if metrics is None or recomputed is None
            else all(
                metrics[key] is None
                or recomputed[key] is None
                or abs(float(metrics[key]) - float(recomputed[key])) < 1e-12
                for key in ("pra_lenient", "uar", "pra_strict")
            )
        ),
        "artifacts": {
            "metrics_json": (
                {"path": f"{rel}/metrics.json", "sha256": sha256_file(metrics_path)}
                if metrics_path.is_file()
                else None
            ),
            "raw_jsonl": {"path": f"{rel}/raw.jsonl", "sha256": sha256_file(raw_path)},
        },
        "training_stages_present": {
            "memory": label_info.get("memory"),
            "sft": label_info.get("sft"),
            "dpo": label_info.get("dpo"),
            "distill": label_info.get("distill"),
        },
    }

    if kind == "pairwise":
        entry["pairwise"] = pairwise_summary(raw_path, run_dir)
        entry["pairwise_sides"] = label_info.get("pairwise_sides")
    if "attribution_note" in label_info:
        entry["attribution_note"] = label_info["attribution_note"]

    for summary_name in ("summary.json",):
        candidate = run_dir / summary_name
        if candidate.is_file():
            entry["summary_json_sha256"] = sha256_file(candidate)

    return entry


def build() -> list[dict[str, Any]]:
    categories = probe_categories()
    entries: list[dict[str, Any]] = []
    for raw_path in sorted(ROOT.glob("results/**/raw.jsonl")):
        rel = raw_path.parent.relative_to(ROOT).as_posix()
        entry = build_entry(rel, categories)
        if entry is not None:
            entries.append(entry)

    analysis_entries = []
    for run_dir in sorted(
        d for d in ROOT.glob("results/**/") if d.is_dir() and (d / "summary.json").is_file()
    ):
        rel = run_dir.relative_to(ROOT).as_posix()
        if (run_dir / "raw.jsonl").is_file():
            continue
        payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        analysis_entries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_path": rel,
                "run_kind": run_kind(rel),
                "study": study_for(rel),
                "system_name": None,
                "label": "derived analysis over committed responses (no single system under test)",
                "ladder_row": None,
                "cited_in_writeups": rel in CITED_IN_WRITEUPS,
                "recomputed_in_claim_audit": False,
                "n_probes": None,
                "metrics_artifact_kind": "summary",
                "metrics": None,
                "metrics_source": None,
                "recomputed_from_raw": None,
                "recomputed_matches_recorded": None,
                "artifacts": {
                    "metrics_json": None,
                    "raw_jsonl": None,
                    "summary_json": {
                        "path": f"{rel}/summary.json",
                        "sha256": sha256_file(run_dir / "summary.json"),
                    },
                },
                "training_stages_present": {
                    "memory": None,
                    "sft": None,
                    "dpo": None,
                    "distill": None,
                },
                "analyses": sorted(payload.keys()),
            }
        )
    return entries + analysis_entries


def serialise(entries: list[dict[str, Any]]) -> str:
    header = {"kind": "header", "schema_version": SCHEMA_VERSION, "note": SCHEMA_NOTE}
    lines = [json.dumps(header, sort_keys=False)]
    lines += [json.dumps(e, sort_keys=False) for e in entries]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail (exit 1) if results/registry.jsonl differs from a fresh build",
    )
    args = parser.parse_args()

    fresh = serialise(build())
    if args.check:
        if not REGISTRY.is_file():
            print(f"FAIL: {REGISTRY.relative_to(ROOT)} does not exist")
            return 1
        current = REGISTRY.read_text(encoding="utf-8")
        if current != fresh:
            current_lines = current.splitlines()
            fresh_lines = fresh.splitlines()
            print(f"FAIL: {REGISTRY.relative_to(ROOT)} is stale")
            for i in range(max(len(current_lines), len(fresh_lines))):
                a = current_lines[i] if i < len(current_lines) else "<missing>"
                b = fresh_lines[i] if i < len(fresh_lines) else "<missing>"
                if a != b:
                    print(f"  line {i + 1}:\n    committed: {a[:200]}\n    rebuilt:   {b[:200]}")
            print("\nRe-run `uv run python scripts/build_result_registry.py`.")
            return 1
        print(
            f"OK: {REGISTRY.relative_to(ROOT)} is current ({len(current.splitlines()) - 1} runs)."
        )
        return 0

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(fresh, encoding="utf-8", newline="\n")
    entries = build()
    print(
        f"wrote {REGISTRY.relative_to(ROOT)}: {len(entries)} entries "
        f"({sum(1 for e in entries if e.get('metrics'))} with recorded metrics, "
        f"{sum(1 for e in entries if (e.get('n_probes') or 0) == 688)} over the full 688-probe PMB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
