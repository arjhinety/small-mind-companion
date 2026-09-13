"""Study 001 is frozen: its evidence must not change.

These tests are the enforcement for `docs/STUDIES.md`. If one fails, the correct response is almost
never to update the manifest — it is to decide whether the change is a *correction* to a Study 001
claim (which goes in `reports/ERRATA.md` against the frozen artifact) or a new *result* (which
belongs to Study 002). Re-freezing Study 001 to make a test pass defeats the point of the freeze.

Runs on CPU with no model download or network access.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports" / "data" / "study-001-freeze.json"
FREEZE_SCRIPT = ROOT / "scripts" / "freeze_study_001.py"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), f"missing freeze manifest: {MANIFEST}"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sha256_lf_normalised(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_manifest_has_the_expected_shape(manifest: dict) -> None:
    assert manifest["artifact_kind"] == "STUDY_001_RESULT_FREEZE"
    assert manifest["study_id"] == "001"
    assert manifest["evidence_tag"] == "study-001"
    assert manifest["hash_method"] == "sha256 over LF-normalised bytes"
    assert manifest["evidence"], "freeze manifest records no evidence"
    assert manifest["file_count"] == sum(len(g) for g in manifest["evidence"].values())


def test_every_frozen_artifact_still_exists(manifest: dict) -> None:
    missing = [
        path
        for group in manifest["evidence"].values()
        for path in group
        if not (ROOT / path).is_file()
    ]
    assert not missing, (
        "Study 001 artifacts recorded in the freeze no longer exist: "
        f"{missing}. Deleting evidence does not un-freeze Study 001."
    )


def test_every_frozen_artifact_is_unchanged(manifest: dict) -> None:
    """The core check: a changed byte in any frozen artifact fails the freeze."""
    changed = []
    for group, entries in manifest["evidence"].items():
        for path, recorded in entries.items():
            actual = sha256_lf_normalised(ROOT / path)
            if actual != recorded["sha256"]:
                changed.append(f"{group}/{path}: recorded {recorded['sha256'][:12]}… != actual {actual[:12]}…")
    assert not changed, (
        "Frozen Study 001 artifacts have changed:\n  "
        + "\n  ".join(changed)
        + "\n\nA correction belongs in reports/ERRATA.md against the frozen artifact; a new result "
        "belongs to Study 002. See docs/STUDIES.md."
    )


def test_eval_metrics_match_the_raw_responses(manifest: dict) -> None:
    """Recompute `uar` and `pra_lenient` from saved responses and compare to `metrics.json`.

    This is the check that a reader most wants to trust: that the headline numbers are derivable
    from the per-probe records rather than only asserted. It is also what the claim audit ran.
    """
    probes = {}
    pmb = ROOT / "data/benchmarks/pmb_v0_full/probes.jsonl"
    for line in pmb.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            probes[row["probe_id"]] = row

    runs = sorted({p.rsplit("/raw.jsonl", 1)[0] for p in manifest["evidence"].get("eval_raw", {})})
    assert runs, "freeze records no raw eval responses to recompute from"

    mismatches = []
    for run in runs:
        raw_path = ROOT / run / "raw.jsonl"
        metrics_path = ROOT / run / "metrics.json"
        if not raw_path.is_file() or not metrics_path.is_file():
            continue
        responses = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        claimed = json.loads(metrics_path.read_text(encoding="utf-8")).get("metrics", {})

        unanswerable = [r for r in responses if probes.get(r["probe"]["probe_id"], {}).get("category") == "unanswerable"]
        answerable = [r for r in responses if probes.get(r["probe"]["probe_id"], {}).get("category") != "unanswerable"]

        uar = sum(1 for r in unanswerable if r.get("abstained")) / len(unanswerable)
        pra = sum(1 for r in answerable if r.get("lenient_correct")) / len(answerable)

        if abs(uar - claimed.get("uar", -1)) > 1e-9:
            mismatches.append(f"{run}: recomputed uar {uar} != metrics.json {claimed.get('uar')}")
        if abs(pra - claimed.get("pra_lenient", -1)) > 1e-9:
            mismatches.append(f"{run}: recomputed pra_lenient {pra} != metrics.json {claimed.get('pra_lenient')}")

    assert not mismatches, "Headline metrics are not derivable from the saved responses:\n  " + "\n  ".join(mismatches)


def test_freeze_check_script_agrees(manifest: dict) -> None:
    """The CLI check CI runs must agree with what the tests just verified."""
    result = subprocess.run(
        [sys.executable, str(FREEZE_SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"freeze --check failed:\n{result.stdout}\n{result.stderr}"
    assert "OK:" in result.stdout


def test_living_documents_are_recorded_but_not_enforced(manifest: dict) -> None:
    """Errata, guardrails and the audit ledger are expected to keep changing."""
    assert manifest.get("living_documents"), "expected living documents to be recorded for provenance"
    overlap = set(manifest["living_documents"]) & {
        path for group in manifest["evidence"].values() for path in group
    }
    assert not overlap, (
        "these documents are both enforced and declared living, which would make every correction "
        f"to them break the freeze: {sorted(overlap)}"
    )
