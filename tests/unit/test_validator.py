"""``scripts/validate.py`` must fail when a claim is wrong, not just pass when they are right.

A validator that has only ever been observed to pass is indistinguishable from a validator that
always passes. These tests exercise the comparison machinery directly and drive the claims evaluator
with a deliberately wrong claim, so the failure path is covered.

Runs on CPU with no model download and no network access.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate.py"


def _load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("smc_validate", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["smc_validate"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v() -> Any:
    return _load_validator()


@pytest.fixture(scope="module")
def evaluator(v: Any) -> Any:
    return v.Evaluator(v._computed_functions())


# ---------------------------------------------------------------------------------------------
# the comparison primitives
# ---------------------------------------------------------------------------------------------
def test_close_is_absolute_not_relative(v: Any) -> None:
    assert v.close(0.7, 0.7)
    assert v.close(0.1513157894736842, 0.1513157894736842)
    assert not v.close(0.1513, 0.1510)
    # A 0.03pp difference on a 15% metric must not be swallowed by float tolerance.
    assert not v.close(15.10, 15.13, 0.001)


def test_walk_dotted_reads_nested_fields(v: Any) -> None:
    document = {"a": {"b": [{"c": 1}]}, "top": [1, 2, 3]}
    assert v.walk_dotted(document, "a.b.0.c") == 1
    assert v.walk_dotted(document, "top.1") == 2
    assert v.walk_dotted(document, "len") == 2


def test_evaluator_rejects_anything_outside_its_language(v: Any, evaluator: Any) -> None:
    assert evaluator.evaluate("1 + 2 * 3") == 7
    assert evaluator.evaluate("(1, 2) == (1, 2)")
    assert evaluator.evaluate("split_counts('data/sft/v0') == (202, 23)")
    assert evaluator.evaluate("not False and 2 > 1")
    for hostile in ("__import__('os')", "open('x')", "(1).__class__", "[x for x in (1,)]"):
        with pytest.raises((ValueError, NameError, SyntaxError)):
            evaluator.evaluate(hostile)


def test_evaluator_unknown_helper_is_an_error_not_a_false(v: Any, evaluator: Any) -> None:
    # A typo'd helper name must crash the check loudly rather than read as "claim failed".
    with pytest.raises(NameError):
        evaluator.evaluate("no_such_helper()")


# ---------------------------------------------------------------------------------------------
# the negative cases: a wrong claim must be reported as wrong
# ---------------------------------------------------------------------------------------------
def test_wrong_json_field_claim_fails(v: Any, evaluator: Any, tmp_path: Path) -> None:
    wrong = {
        "id": "deliberately-wrong-pra-lenient",
        "claim": "D_memory pra_lenient is 99%",
        "where": "README.md:1",
        "artifact": "results/v0.1/D_memory/metrics.json",
        "check": {
            "type": "json_field",
            "path": "results/v0.1/D_memory/metrics.json",
            "field": "metrics.pra_lenient",
            "equals": 0.99,
            "decimals": 2,
            "quoted": 99.0,
        },
        "status": "verified",
    }
    passed, detail = v.evaluate_check(wrong["check"], evaluator)
    assert not passed
    assert "0.99" in detail


def test_wrong_computed_claim_fails(v: Any, evaluator: Any) -> None:
    check = {
        "type": "computed",
        "expr": "pmb_persona_count()",
        "equals": 40,
    }
    passed, detail = v.evaluate_check(check, evaluator)
    assert not passed
    assert "8" in detail


def test_wrong_count_lines_claim_fails(v: Any, evaluator: Any) -> None:
    check = {
        "type": "count_lines",
        "glob": "data/benchmarks/pmb_v0_full/probes.jsonl",
        "equals": 100,
    }
    passed, _detail = v.evaluate_check(check, evaluator)
    assert not passed


def test_claim_pointed_at_a_missing_artifact_fails(v: Any, evaluator: Any) -> None:
    check = {"type": "file_exists", "path": "registry/does-not-exist.jsonl"}
    passed, _detail = v.evaluate_check(check, evaluator)
    assert not passed


def test_evaluating_a_wrong_claims_file_reports_a_failure(
    v: Any, evaluator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a claims file with one wrong entry produces a failure."""
    claims = tmp_path / "claims.jsonl"
    good = {
        "id": "pmb-probe-count",
        "claim": "688 probes",
        "where": "README.md:43",
        "artifact": "data/benchmarks/pmb_v0_full/probes.jsonl",
        "check": {
            "type": "count_lines",
            "glob": "data/benchmarks/pmb_v0_full/probes.jsonl",
            "equals": 688,
        },
        "status": "verified",
    }
    bad = dict(good)
    bad["id"] = "pmb-probe-count-wrong"
    bad["check"] = {
        "type": "count_lines",
        "glob": "data/benchmarks/pmb_v0_full/probes.jsonl",
        "equals": 689,
    }
    claims.write_text("\n".join(json.dumps(row) for row in (good, bad)) + "\n", encoding="utf-8")
    monkeypatch.setattr(v, "CLAIMS", claims)

    findings, _manual, _drift = v.check_claims(evaluator)
    failed = [finding for finding in findings if not finding.ok]
    assert [finding.claim_id for finding in failed] == ["pmb-probe-count-wrong"]
    assert any(finding.ok for finding in findings)


def test_acknowledged_drift_requires_a_resolution(
    v: Any, evaluator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`acknowledged_drift` may not be used as a silent escape hatch."""
    claims = tmp_path / "claims.jsonl"
    row = {
        "id": "unresolved-drift",
        "claim": "a claim that does not hold",
        "where": "README.md:1",
        "artifact": None,
        "check": {"type": "computed", "expr": "pmb_persona_count()", "equals": 40},
        "status": "acknowledged_drift",
    }
    claims.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(v, "CLAIMS", claims)
    findings, _manual, _drift = v.check_claims(evaluator)
    assert any("resolution" in finding.message for finding in findings if not finding.ok)


def test_manual_status_must_have_a_manual_check(
    v: Any, evaluator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claims = tmp_path / "claims.jsonl"
    row = {
        "id": "manual-with-a-real-check",
        "claim": "claim",
        "where": "README.md:1",
        "artifact": None,
        "check": {"type": "computed", "expr": "pmb_persona_count()", "equals": 8},
        "status": "manual",
    }
    claims.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(v, "CLAIMS", claims)
    findings, _manual, _drift = v.check_claims(evaluator)
    assert any("not manual" in finding.message for finding in findings if not finding.ok)


def test_cross_document_check_detects_a_disagreement(
    v: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing one document's quote must break the cross-document check."""
    original = v.read_text

    def tampered(rel: str) -> str:
        text = original(rel)
        if rel == "docs/STUDIES.md":
            return text.replace(
                "| D | raw + hybrid retrieval memory (k=8) | — | 15.13%",
                "| D | raw + hybrid retrieval memory (k=8) | — | 12.34%",
            )
        return text

    monkeypatch.setattr(v, "read_text", tampered)
    assert not v.metric_values_consistent()
    assert not v.ladder_rows_agree("README.md", "docs/STUDIES.md")


# ---------------------------------------------------------------------------------------------
# the real repository, through the real checks
# ---------------------------------------------------------------------------------------------
def test_freeze_check_passes(v: Any) -> None:
    assert v.freeze_check_passes()
    assert v.freeze_file_count_matches_evidence()
    assert v.freeze_shape_claims()


def test_all_corpora_hash_clean(v: Any) -> None:
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    expected = len(v.BENCHMARK_HASH_DIRS) + len(v.DATASET_HASH_DIRS)
    assert len(registry["datasets"]) >= expected
    assert v.hash_clean_corpus_count() == expected
    assert v.hash_corpus_count_with_hash_txt() == expected


def test_every_saved_run_recomputes_to_its_metrics_json(v: Any) -> None:
    findings = v.check_metrics()
    assert findings, "no runs were recomputed, which means the check found nothing to check"
    assert all(finding.ok for finding in findings)
    assert v.recomputed_run_count() == 7


def test_claims_matrix_has_no_failing_entry(v: Any, evaluator: Any) -> None:
    findings, manual, drift = v.check_claims(evaluator)
    failures = [finding for finding in findings if not finding.ok]
    assert not failures, "\n".join(f"{f.claim_id}: {f.message}" for f in failures)
    assert manual, "expected at least one honestly-manual claim"
    for finding in manual:
        assert finding.message.strip(), "a manual claim must state why it is manual"
    for finding in drift:
        assert finding.message.strip()


def test_claims_matrix_entries_are_well_formed(v: Any) -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "registry/claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 100, "the claims matrix should cover the whole public surface"
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert set(row) >= {"id", "claim", "where", "artifact", "check", "status"}
        assert row["status"] in v.VALID_STATUSES
        # `where` is a locator, so its file must exist and its line must be inside the file. The
        # line's *content* is deliberately not checked: it moves whenever an unrelated paragraph is
        # added above it, and asserting on it would make every doc edit look like a broken claim.
        where_file, _, line_text = row["where"].rpartition(":")
        path = ROOT / where_file
        assert path.is_file(), f"{row['id']} points at missing file {where_file}"
        first_line = line_text.split("-")[0]
        assert first_line.isdigit(), f"{row['id']} has a malformed locator {row['where']!r}"
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert int(first_line) <= line_count, (
            f"{row['id']} points past the end of {where_file} "
            f"(line {first_line}, file has {line_count} lines)"
        )
        if row["status"] == "acknowledged_drift":
            assert row.get("resolution"), f"{row['id']} is drift with no resolution"


def test_registry_counts_match_the_data_files(v: Any) -> None:
    assert v.dataset_registry_counts_match()
    assert v.dataset_registry_hashes_match()
    assert v.dataset_registry_evidence_resolves()
    assert v.benchmark_registry_counts_match()


def test_derived_result_registry_is_current(v: Any) -> None:
    assert v.result_registry_is_fresh()
    assert v.result_registry_hashes_match()
    assert v.result_registry_covers_all_raw_runs()
    assert v.result_registry_never_invents_metrics()


def test_no_dpo_checkpoint_has_a_full_pmb_metric(v: Any) -> None:
    assert v.no_metrics_for_dpo_row()


def test_guardrail_references_resolve(v: Any) -> None:
    assert v.guardrail_references_resolve()
    assert v.errata_ids_resolve()
    assert v.errata_count_matches_readme()


def test_full_run_exits_zero(v: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The validator's own summary line must report success, which is what CI consumes."""
    findings, manual, drift = v.run(include_pollution=False, strict_hygiene=False)
    assert not [finding for finding in findings if not finding.ok]
    assert manual or drift


def test_strict_hygiene_is_fatal_and_only_for_hygiene(v: Any) -> None:
    """`--strict` must actually change the outcome, and only through hygiene findings.

    Non-strict, the repository passes. Strict, the documented determinism gaps (five training
    configs that do not pin a revision, two probe sets with no hash.txt) make it fail -- which is
    what the flag advertises. If this test ever sees strict == non-strict, the flag is a no-op.
    """
    relaxed, _manual, _drift = v.run(include_pollution=False, strict_hygiene=False)
    strict, _manual2, _drift2 = v.run(include_pollution=False, strict_hygiene=True)
    relaxed_failures = {finding.claim_id for finding in relaxed if not finding.ok}
    strict_failures = {finding.claim_id for finding in strict if not finding.ok}

    assert not relaxed_failures, "the repository should pass without --strict"
    assert strict_failures, "--strict must fail on the documented hygiene gaps"
    assert strict_failures <= {
        "config-revisions",
        "dataset-hashes",
        "unpinned-probe-sets",
        "documented-vs-artifact-metrics",
    }
    assert "config-revisions" in strict_failures, "the five 'main' revision pins are a real gap"
