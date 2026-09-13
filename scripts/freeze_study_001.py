#!/usr/bin/env python
"""Freeze Study 001.

Records a SHA-256 for every artifact Study 001's findings rest on, so the claims on the study page
and in `reports/ERRATA.md` can be checked against a fixed state rather than a moving tree. Writing
a freeze does not change any result; it only pins what already exists.

Line endings: this repo sets `core.autocrlf=true`, so a Windows working tree is CRLF while the git
blobs are LF. Hashes are computed over **LF-normalised** bytes so they verify on a fresh clone
(see `docs/GUARDRAILS.md` G9 and OpenGrad's #27, which is the mistake this avoids).

Usage:

    uv run python scripts/freeze_study_001.py            # write the manifest
    uv run python scripts/freeze_study_001.py --check     # verify without writing (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "reports" / "data" / "study-001-freeze.json"

STUDY_ID = "001"
EVIDENCE_TAG = "study-001"
FROZEN_ON = "2026-09-13"

# Every path Study 001's claims depend on, grouped by what it is evidence *of*. Globs are resolved
# at freeze time and must be non-empty, so a renamed or deleted artifact fails the freeze loudly
# rather than silently dropping out of it.
GROUPS: dict[str, list[str]] = {
    "benchmark": [
        "data/benchmarks/pmb_v0_full/probes.jsonl",
        "data/benchmarks/pmb_v0_full/DATASHEET.md",
        "data/benchmarks/pmb_v0_full/hash.txt",
        "data/benchmarks/pmb_v0_full/personas/*.json",
    ],
    "training_data": [
        "data/sft/v0/train.jsonl",
        "data/sft/v0/val.jsonl",
        "data/sft/v0/DATASHEET.md",
        "data/sft/v1/train.jsonl",
        "data/sft/v1/val.jsonl",
        "data/sft/v1/DATASHEET.md",
        "data/dpo/v0/train.jsonl",
        "data/dpo/v0/val.jsonl",
        "data/dpo/v1_scale/train.jsonl",
        "data/dpo/v1_scale/val.jsonl",
        "data/dpo/v1_scale/DATASHEET.md",
        "data/distill/v1/train.jsonl",
        "data/distill/v1/val.jsonl",
        "data/distill/v1/DATASHEET.md",
    ],
    "eval_results": [
        "results/v0.0/bakeoff_scores.json",
        "results/v0.0/latency_baseline.json",
        "results/v0.1/*/metrics.json",
        "results/v0.1/*/summary.json",
        "results/v0.1/ksweep/*/metrics.json",
        "results/v1_scale/*/metrics.json",
        "results/v1_scale/*/summary.json",
    ],
    # Raw per-probe responses: the ground truth the metrics were recomputed from during the audit.
    "eval_raw": [
        "results/v0.1/A_raw/raw.jsonl",
        "results/v0.1/B_sft/raw.jsonl",
        "results/v0.1/D_memory/raw.jsonl",
        "results/v0.1/E_sft_memory/raw.jsonl",
        "results/v1_scale/B_sft/raw.jsonl",
        "results/v1_scale/E_sft_memory/raw.jsonl",
        "results/v1_scale/E_distill/raw.jsonl",
    ],
    "report": [
        "docs/day3_memory_results.md",
        "docs/day4_sft_results.md",
        "docs/day4_sft_v1_results.md",
        "docs/dpo_results.md",
        "docs/distillation_results.md",
        "docs/proper_scale_results.md",
        "docs/quantization_results.md",
        "docs/model_quirks.md",
        "docs/research_questions.md",
        "docs/reproduction.md",
    ],
    "config": [
        "configs/training/sft.yaml",
        "configs/training/sft_v1.yaml",
        "configs/training/dpo.yaml",
        "configs/training/dpo_v1_scale.yaml",
        "configs/training/dpo_v1_more_epochs.yaml",
        "configs/training/distill_v1.yaml",
    ],
    "public_surface": [
        "README.md",
        "CITATION.cff",
        "hf_readmes/generate_cards.py",
        "hf_readmes/checkpoint_README_template.py",
        "hf_readmes/README_distill-v1.md",
        "hf_readmes/README_distill-v1-gguf.md",
        "hf_readmes/gguf_README.md",
    ],
}

# Re-freezes that legitimately changed a pinned artifact. Recording these is the difference between
# a freeze that means something and one that gets quietly refreshed whenever a check fails.
REFREEZE_LOG: list[dict[str, str]] = [
    {
        "on": "2026-09-13",
        "artifact": "hf_readmes/generate_cards.py",
        "reason": "Formatting only (one long line wrapped by black to satisfy E501; no values "
        "touched). Verified after the change that the generator still emits the corrected card "
        "content: no '70.0%' UAR on the DPO card, 'not measured' in the DPO eval table, and "
        "'15.30% / 70.0%' attributed to SFT+memory. No Study 001 finding depends on this file's "
        "byte content, only on the cards it produces.",
    },
    {
        "on": "2026-09-13",
        "artifact": "README.md",
        "reason": "Corrected to name the four hash.txt files that verify and to point at this "
        "manifest as the authoritative pin for the benchmark corpora (finding #29). A correction "
        "to a claim about the artifacts, not a change to any artifact.",
    },
    {
        "on": "2026-09-13",
        "artifact": "README.md",
        "reason": "Added the Study 001 write-up link (https://small-mind.arjhinety.com) and the "
        "'make validate' command to the Studies section, and pointed the freeze bullet at the "
        "manifest rather than restating its file count. Documentation only: no result, dataset, "
        "metric or limitation changed.",
    },
]

# Process documents that are *expected to keep changing* — errata grows as mistakes are found,
# guardrails gain rules, and the audit ledger gains resolutions. Hashing them into the enforced set
# would mean every future correction to the record "breaks the freeze", which is backwards. They
# are recorded here for provenance and deliberately excluded from `--check`.
LIVING_DOCUMENTS: list[str] = [
    "docs/GUARDRAILS.md",
    "docs/STUDIES.md",
    "reports/ERRATA.md",
    "reports/audits/001-claim-audit/README.md",
    "reports/audits/001-claim-audit/findings.json",
    "reports/audits/001-claim-audit/resolutions.json",
]


def sha256_lf_normalised(path: Path) -> tuple[str, int]:
    """Hash the file's **LF-normalised** bytes, and report that same normalised size.

    Both values come from the normalised stream on purpose. This repo sets
    `core.autocrlf=true`, so a Windows working tree is CRLF while a fresh clone is LF — reporting
    the raw working-tree size against a hash of the normalised bytes produces a manifest that
    disagrees with itself on checkout, which is exactly what a freeze check must never do.
    """
    normalised = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalised).hexdigest(), len(normalised)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def collect() -> dict[str, dict[str, dict[str, object]]]:
    evidence: dict[str, dict[str, dict[str, object]]] = {}
    missing: list[str] = []
    for group, patterns in GROUPS.items():
        entries: dict[str, dict[str, object]] = {}
        for pattern in patterns:
            matches = sorted(ROOT.glob(pattern)) if "*" in pattern else [ROOT / pattern]
            matches = [m for m in matches if m.is_file()]
            if not matches:
                missing.append(f"{group}: {pattern}")
                continue
            for path in matches:
                digest, size = sha256_lf_normalised(path)
                entries[path.relative_to(ROOT).as_posix()] = {"bytes": size, "sha256": digest}
        evidence[group] = dict(sorted(entries.items()))
    if missing:
        raise SystemExit(
            "freeze failed — these paths matched nothing, so the manifest would silently omit "
            "evidence:\n  " + "\n  ".join(missing)
        )
    return evidence


def collect_living() -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for pattern in LIVING_DOCUMENTS:
        path = ROOT / pattern
        if path.is_file():
            digest, size = sha256_lf_normalised(path)
            entries[pattern] = {"bytes": size, "sha256": digest}
    return dict(sorted(entries.items()))


def build() -> dict[str, object]:
    evidence = collect()
    n_files = sum(len(v) for v in evidence.values())
    return {
        "artifact_kind": "STUDY_001_RESULT_FREEZE",
        "study_id": STUDY_ID,
        "evidence_tag": EVIDENCE_TAG,
        "frozen_on": FROZEN_ON,
        "git_commit": git_commit(),
        "hash_method": "sha256 over LF-normalised bytes",
        "file_count": n_files,
        "evidence": evidence,
        "refreeze_log": REFREEZE_LOG,
        "refreeze_log_note": (
            "The freeze was re-taken. Each entry records what moved and why it did not change a "
            "Study 001 finding. An unrecorded re-freeze makes the manifest meaningless, so this "
            "list is part of the artifact."
        ),
        "living_documents": collect_living(),
        "living_documents_note": (
            "Recorded for provenance only; excluded from --check. These documents are expected to "
            "keep changing as mistakes are found and rules are added."
        ),
        "not_claimed": [
            "No full-PMB pra_lenient or UAR measurement exists for any DPO checkpoint; "
            "dpo-v1-scale appears only in pairwise comparisons.",
            "No quantization figure in this repository is backed by a committed artifact "
            "(*.gguf is gitignored and results/ holds no quantization output).",
            "data/distill/v1/ has no hash.txt, so its contents are not integrity-pinned here.",
            "The acceptable_alternatives field is unpopulated in 688/688 PMB probes, which is "
            "why pra_strict is ~0 throughout and pra_lenient is the reported metric.",
            "Every result is a single seed and a single run. Nothing here is replicated.",
            "No human evaluation exists; no reviewer log or teacher transcript was retained, so "
            "the teacher identity is self-reported and unverifiable from this repository.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the manifest without writing it"
    )
    args = parser.parse_args()

    fresh = build()

    if args.check:
        if not MANIFEST.exists():
            print(f"FAIL: {MANIFEST.relative_to(ROOT)} does not exist")
            return 1
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
        problems: list[str] = []
        if recorded.get("evidence") != fresh["evidence"]:
            rec, now = recorded.get("evidence", {}), fresh["evidence"]
            for group in sorted(set(rec) | set(now)):
                r, n = rec.get(group, {}), now.get(group, {})
                for path in sorted(set(r) | set(n)):
                    if r.get(path) != n.get(path):
                        problems.append(
                            f"  {group}/{path}: recorded {r.get(path)} != actual {n.get(path)}"
                        )
        if recorded.get("file_count") != fresh["file_count"]:
            problems.append(
                f"  file_count: recorded {recorded.get('file_count')} "
                f"!= actual {fresh['file_count']}"
            )
        if problems:
            print("FAIL: the frozen Study 001 artifacts have changed since the freeze:")
            print("\n".join(problems))
            print(
                "\nStudy 001 is frozen. If a result genuinely changed, that belongs to Study 002 — "
                "see docs/STUDIES.md."
            )
            return 1
        print(f"OK: {fresh['file_count']} Study 001 artifacts match the freeze ({FROZEN_ON}).")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(fresh, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"wrote {MANIFEST.relative_to(ROOT)}: {fresh['file_count']} artifacts "
        f"across {len(fresh['evidence'])} groups"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
