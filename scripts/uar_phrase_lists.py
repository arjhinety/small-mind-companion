"""UAR under the original and the extended abstention phrase list, from saved responses.

UAR (unanswerable-abstention rate) is scored by ``detect_abstention`` in
``src/onebee/evaluation/graders/rule.py``: a response counts as an abstention if it contains any
phrase in ``_ABSTENTION_PHRASES``. That list was extended twice on 2026-08-14 (commits 2615598 and
2156e20), after some v1 outputs had been seen, to include the phrasing the SFT data teaches. This
script re-scores every saved response under both versions, so the effect of the extension is
measured instead of argued:

* **original** -- ``_ABSTENTION_PHRASES[:17]``, the 17 phrases the list held before either
  extension. ``--verify-history`` confirms they equal the whole list at commit 552cf47.
* **current** -- the full ``_ABSTENTION_PHRASES``.

Both lists are the module's own list object, loaded by importing ``rule.py`` from its path (it
imports only ``re`` and ``string``), so this script and ``scripts/validate.py`` stay free of the
``onebee`` package's heavier dependencies. Matching is ``detect_abstention``'s rule: lowercase the
response, then test substring membership.

Reads ``results/v0.1/*/raw.jsonl`` and ``results/v1_scale/*/raw.jsonl``. Files whose rows are not
per-probe ``probe``/``response``/``abstained`` records (the pairwise comparisons) are listed as
skipped. Nothing is written back: stored ``abstained`` flags and ``metrics.json`` are only read.

Usage::

    uv run python scripts/uar_phrase_lists.py                  # (re)write the output JSON
    uv run python scripts/uar_phrase_lists.py --check          # exit 1 if the output is stale
    uv run python scripts/uar_phrase_lists.py --verify-history # [:17] == list at 552cf47 (git)
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
OUTPUT = ROOT / "results" / "audit" / "uar_phrase_lists" / "uar_phrase_lists.json"
RULE_PATH = "src/onebee/evaluation/graders/rule.py"
ORIGINAL_N = 17
ORIGINAL_COMMIT = "552cf47"
RAW_GLOBS = ("results/v0.1/*/raw.jsonl", "results/v1_scale/*/raw.jsonl")


def _load_module(name: str, source: str, filename: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102 - repository code
    return module


def load_rule() -> types.ModuleType:
    """Import the working-tree rule.py by path."""
    spec = importlib.util.spec_from_file_location("onebee_rule_by_path", ROOT / RULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {RULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phrase_lists() -> dict[str, list[str]]:
    phrases = list(load_rule()._ABSTENTION_PHRASES)
    return {"original": phrases[:ORIGINAL_N], "current": phrases}


def abstains(phrases: list[str], response: str) -> bool:
    """detect_abstention's rule, applied with an explicit list."""
    lower = response.lower()
    return any(phrase in lower for phrase in phrases)


def sha256_lf_normalised(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]] | None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all({"probe", "response", "abstained"} <= row.keys() for row in rows):
        return None
    return rows


def score_run(rows: list[dict[str, Any]], lists: dict[str, list[str]]) -> dict[str, Any]:
    unanswerable = [row for row in rows if not row["probe"]["answerable"]]
    answerable = [row for row in rows if row["probe"]["answerable"]]
    result: dict[str, Any] = {
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "stored": {
            "abstained_unanswerable": sum(1 for row in unanswerable if row["abstained"]),
            "abstained_answerable": sum(1 for row in answerable if row["abstained"]),
        },
    }
    for name, phrases in lists.items():
        unans = sum(1 for row in unanswerable if abstains(phrases, row["response"]))
        ans = sum(1 for row in answerable if abstains(phrases, row["response"]))
        result[name] = {
            "abstained_unanswerable": unans,
            "uar": unans / len(unanswerable) if unanswerable else 0.0,
            "abstained_answerable": ans,
            "stored_flags_match": all(
                bool(row["abstained"]) == abstains(phrases, row["response"]) for row in rows
            ),
        }
    return result


def build() -> dict[str, Any]:
    lists = phrase_lists()
    runs: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    inputs: dict[str, str] = {}
    for pattern in RAW_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            rows = load_rows(path)
            if rows is None:
                skipped[rel] = "rows are not per-probe probe/response/abstained records"
                continue
            inputs[rel] = sha256_lf_normalised(path)
            runs[path.parent.relative_to(ROOT).as_posix()] = score_run(rows, lists)
    return {
        "artifact_kind": "uar-phrase-list-comparison",
        "generated_by": "scripts/uar_phrase_lists.py",
        "metric": (
            "uar = abstentions on unanswerable probes / unanswerable probes; "
            "abstained_answerable counts answerable probes flagged as abstaining"
        ),
        "lists": {
            "original": {
                "source": f"{RULE_PATH} _ABSTENTION_PHRASES[:{ORIGINAL_N}]",
                "equals": f"the whole list at commit {ORIGINAL_COMMIT} (--verify-history)",
                "phrases": lists["original"],
            },
            "current": {
                "source": f"{RULE_PATH} _ABSTENTION_PHRASES",
                "n_phrases": len(lists["current"]),
                "added_after_original": lists["current"][ORIGINAL_N:],
            },
        },
        "note": "Read-only re-scoring. Stored abstained flags and metrics.json are not modified.",
        "inputs": inputs,
        "skipped": skipped,
        "runs": runs,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def verify_history() -> bool:
    source = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:{RULE_PATH}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    historical = _load_module("rule_at_original", source, f"{ORIGINAL_COMMIT}:{RULE_PATH}")
    return list(historical._ABSTENTION_PHRASES) == phrase_lists()["original"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="fail if the output is stale")
    group.add_argument(
        "--verify-history",
        action="store_true",
        help=f"confirm the original list equals rule.py at {ORIGINAL_COMMIT} (needs git history)",
    )
    args = parser.parse_args(argv)
    rel = OUTPUT.relative_to(ROOT).as_posix()
    if args.verify_history:
        ok = verify_history()
        print(
            f"_ABSTENTION_PHRASES[:{ORIGINAL_N}] "
            f"{'equals' if ok else 'DIFFERS FROM'} the list at {ORIGINAL_COMMIT}"
        )
        return 0 if ok else 1
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
