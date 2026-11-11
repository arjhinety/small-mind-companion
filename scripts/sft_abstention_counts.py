"""Count abstention and irrelevant-retrieval training examples in each committed SFT corpus.

The SFT corpora carry no ``kind`` field: ``generate_sft_data.py`` assigned a kind at generation
time and recorded only the totals in each ``DATASHEET.md``. The two non-recall kinds, however, use
fixed target responses, so they can be recovered from the committed records by exact match on the
assistant turn:

* **abstention** -- the target is one of ``ABSTAIN_RESPONSES``;
* **irrelevant_retrieval** -- the target is one of ``IRRELEVANT_RESPONSES``;
* **memory_relevant** -- anything else (real teacher text).

The template lists are read from ``generate_sft_data.py`` with ``ast.literal_eval`` on their
assignments. That script cannot be imported for them: it runs the whole generation pipeline at
module level and imports ``sentence_transformers``. The v0 corpus was generated with one fixed
string per kind (commit f0d7ece); each of those strings is the first entry of the current list,
and ``--verify-history`` confirms that from git.

Standard library only, so ``scripts/validate.py`` can recompute every count. Nothing is written
back to ``data/``.

Usage::

    uv run python scripts/sft_abstention_counts.py                  # (re)write the output JSON
    uv run python scripts/sft_abstention_counts.py --check          # exit 1 if stale
    uv run python scripts/sft_abstention_counts.py --verify-history # v0 templates + v1 history
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "audit" / "sft_abstention_counts" / "sft_abstention_counts.json"
GENERATOR = "generate_sft_data.py"
CORPORA = ("data/sft/v0", "data/sft/v1")
SPLITS = ("train", "val")
V0_COMMIT = "f0d7ece"
# Commits that rewrote data/sft/v1, oldest first, for --verify-history.
V1_HISTORY = (
    ("2b4c308", "first proper-scale generation (dedup bug present)"),
    ("bc2f409", "regenerated after the dedup fix (bd7e556)"),
    ("799390e", "regenerated after the ratio rebalance and template diversification (2156e20)"),
)


def _list_assignments(source: str, names: tuple[str, ...]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = list(ast.literal_eval(node.value))
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"{GENERATOR}: no list assignment for {missing}")
    return found


def templates() -> dict[str, list[str]]:
    source = (ROOT / GENERATOR).read_text(encoding="utf-8")
    lists = _list_assignments(source, ("ABSTAIN_RESPONSES", "IRRELEVANT_RESPONSES"))
    return {
        "abstention": lists["ABSTAIN_RESPONSES"],
        "irrelevant_retrieval": lists["IRRELEVANT_RESPONSES"],
    }


def assistant_text(record: dict[str, Any]) -> str:
    turns = [m for m in record["messages"] if m["role"] == "assistant"]
    if len(turns) != 1:
        raise ValueError(f"expected exactly one assistant turn, found {len(turns)}")
    return str(turns[0]["content"])


def classify(text: str, lists: dict[str, list[str]]) -> str:
    for kind, responses in lists.items():
        if text in responses:
            return kind
    return "memory_relevant"


def count_lines(lines: list[str], lists: dict[str, list[str]]) -> dict[str, int]:
    counts = {"abstention": 0, "irrelevant_retrieval": 0, "memory_relevant": 0}
    for line in lines:
        if line.strip():
            counts[classify(assistant_text(json.loads(line)), lists)] += 1
    return counts


def sha256_lf_normalised(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> dict[str, Any]:
    lists = templates()
    corpora: dict[str, Any] = {}
    inputs: dict[str, str] = {}
    for corpus in CORPORA:
        by_split: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            path = ROOT / corpus / f"{split}.jsonl"
            inputs[path.relative_to(ROOT).as_posix()] = sha256_lf_normalised(path)
            by_split[split] = count_lines(path.read_text(encoding="utf-8").splitlines(), lists)
        total = {kind: sum(s[kind] for s in by_split.values()) for kind in by_split["train"]}
        n = sum(total.values())
        corpora[corpus] = {
            "n_examples": n,
            "by_split": by_split,
            "total": total,
            "share": {kind: count / n for kind, count in total.items()},
            "distinct_targets": {
                kind: sorted(
                    {
                        text
                        for split in SPLITS
                        for line in (ROOT / corpus / f"{split}.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                        for text in [assistant_text(json.loads(line))]
                        if classify(text, lists) == kind
                    }
                )
                for kind in ("abstention", "irrelevant_retrieval")
            },
        }
    return {
        "artifact_kind": "sft-abstention-example-counts",
        "generated_by": "scripts/sft_abstention_counts.py",
        "method": (
            "exact match of each record's assistant turn against generate_sft_data.py's "
            "ABSTAIN_RESPONSES and IRRELEVANT_RESPONSES; everything else is memory_relevant"
        ),
        "templates": lists,
        "inputs": inputs,
        "corpora": corpora,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _git_show(spec: str) -> str:
    return subprocess.run(
        ["git", "show", spec],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def verify_history() -> bool:
    """v0's fixed targets are in the current lists; print each historical v1 corpus's counts."""
    lists = templates()
    tree = ast.parse(_git_show(f"{V0_COMMIT}:{GENERATOR}"))
    v0_targets = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "target" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    covered = all(any(t in responses for responses in lists.values()) for t in v0_targets)
    print(f"{V0_COMMIT} fixed targets ({len(v0_targets)}) in current lists: {covered}")
    for commit, label in V1_HISTORY:
        lines = [
            line
            for split in SPLITS
            for line in _git_show(f"{commit}:data/sft/v1/{split}.jsonl").splitlines()
        ]
        print(f"data/sft/v1 at {commit} ({label}): {count_lines(lines, lists)}")
    return covered and len(v0_targets) == 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="fail if the output is stale")
    group.add_argument(
        "--verify-history", action="store_true", help="check v0 templates; count v1 history"
    )
    args = parser.parse_args(argv)
    if args.verify_history:
        return 0 if verify_history() else 1
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
