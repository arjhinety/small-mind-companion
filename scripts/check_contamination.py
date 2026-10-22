#!/usr/bin/env python3
"""13-gram overlap contamination check (per MMLU/Llama convention).

Checks whether any eval file shares n-grams with any train file.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from typing import Any


def extract_text_fields(obj: Any) -> list[str]:
    """Return every non-empty string value in a record, recursing into dicts and lists.

    Reading a fixed list of keys made the check see nothing in records that use other
    field names (emotional_range's "context", h22's "prompt", DPO's "prompt"/"chosen"/
    "rejected", distillation's "prompt"), so those files were reported clean without
    being compared. Every string value can carry leaked text, so every string is read.
    """
    results: list[str] = []
    if isinstance(obj, str):
        if obj:
            results.append(obj)
    elif isinstance(obj, dict):
        for val in obj.values():
            results.extend(extract_text_fields(val))
    elif isinstance(obj, list):
        for val in obj:
            results.extend(extract_text_fields(val))
    return results


_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def load_jsonl_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                print(
                    f"WARNING: skipping malformed JSON at {path}:{lineno}",
                    file=sys.stderr,
                )
                continue
            texts.extend(extract_text_fields(obj))
    return texts


def check_contamination(
    train_files: list[str],
    eval_files: list[str],
    n: int,
    min_overlap: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for eval_file in eval_files:
        eval_texts = load_jsonl_texts(eval_file)
        eval_ngrams: set[tuple[str, ...]] = set()
        for text in eval_texts:
            eval_ngrams |= ngrams(tokenize(text), n)

        for train_file in train_files:
            train_texts = load_jsonl_texts(train_file)
            train_ngrams: set[tuple[str, ...]] = set()
            for text in train_texts:
                train_ngrams |= ngrams(tokenize(text), n)

            shared = train_ngrams & eval_ngrams
            if len(shared) >= min_overlap:
                example_ngrams = [" ".join(g) for g in sorted(shared)[:5]]
                findings.append(
                    {
                        "train_file": train_file,
                        "eval_file": eval_file,
                        "n": n,
                        "overlap_count": len(shared),
                        "example_ngrams": example_ngrams,
                    }
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="n-gram contamination check between train and eval JSONL files",
    )
    parser.add_argument(
        "--train-glob",
        nargs="*",
        default=[],
        help="glob pattern(s) for train JSONL files (supports **)",
    )
    parser.add_argument(
        "--eval-glob",
        nargs="*",
        default=[],
        help="glob pattern(s) for eval/probe JSONL files (supports **)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=13,
        help="n-gram size (default: 13, per MMLU/Llama convention)",
    )
    parser.add_argument(
        "--min-overlap-ngrams",
        type=int,
        default=1,
        help=(
            "minimum number of shared n-grams to flag as contamination "
            "(default: 1 — a real production version would tune this)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print additional details during processing",
    )
    args = parser.parse_args()

    train_files: list[str] = []
    for pattern in args.train_glob:
        matched = sorted(glob.glob(pattern, recursive=True))
        if args.verbose:
            print(f"train-glob '{pattern}' matched {len(matched)} file(s)")
        train_files.extend(matched)

    eval_files: list[str] = []
    for pattern in args.eval_glob:
        matched = sorted(glob.glob(pattern, recursive=True))
        if args.verbose:
            print(f"eval-glob '{pattern}' matched {len(matched)} file(s)")
        eval_files.extend(matched)

    if not train_files or not eval_files:
        side = "train" if not train_files else "eval"
        print(
            f"WARNING: zero {side} files matched — nothing to check yet, "
            f"exiting cleanly (an empty corpus is not contamination).",
            file=sys.stderr,
        )
        return 0

    findings = check_contamination(
        train_files,
        eval_files,
        args.n,
        args.min_overlap_ngrams,
    )

    if findings:
        print(f"CONTAMINATION DETECTED ({len(findings)} finding(s)):\n")
        for f in findings:
            print(f"  train: {f['train_file']}")
            print(f"  eval:  {f['eval_file']}")
            print(f"  {f['n']}-gram overlap count: {f['overlap_count']}")
            print(f"  example n-grams: {f['example_ngrams']}")
            print()
        return 1
    else:
        print("No contamination found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
