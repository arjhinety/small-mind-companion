#!/usr/bin/env python
"""Recompute every dataset ``hash.txt`` from a clean, platform-independent encode.

Why this exists (see ``reports/ERRATA.md`` E29 and ``docs/GUARDRAILS.md`` G9c): the four benchmark
``hash.txt`` files committed with Study 001 do not match their corpora under any algorithm tested,
including thirteen variants of path encoding, byte/text reading, ordering, and DATASHEET inclusion.
They may never have matched. ``data/distill/v1/`` never had a hash at all. The four SFT/DPO hashes
do verify, so they are only rewritten if this script is asked to.

Every digest is computed over **LF-normalised** bytes so it is a property of the content rather than
of the checkout's line-ending convention (this repo sets ``core.autocrlf=true`` and has no
``.gitattributes``).

Usage::

    uv run python scripts/recompute_hashes.py            # report drift, change nothing
    uv run python scripts/recompute_hashes.py --write    # rewrite hash.txt files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Corpora hashed by scripts/build_pmb.py: sorted directory walk, forward-slash relative path then
# file bytes, skipping hash.txt itself.
BENCHMARK_DIRS = [
    "data/benchmarks/pmb_v0_full",
    "data/benchmarks/sft_personas_v1",
    "data/benchmarks/pmb_v0",
    "data/benchmarks/sft_personas_v0",
]

# Corpora hashed by generate_sft_data.py / generate_dpo_data.py: sha256 over the concatenated
# contents of every *.jsonl in sorted order.
DATASET_DIRS = [
    "data/sft/v0",
    "data/sft/v1",
    "data/dpo/v0",
    "data/dpo/v1_scale",
    "data/distill/v1",
]


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def benchmark_hash(out_dir: Path) -> str:
    hasher = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(out_dir)):
        for fname in sorted(files):
            if fname == "hash.txt":
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(out_dir)
            hasher.update(_lf(rel.as_posix().encode("utf-8")))
            hasher.update(_lf(fpath.read_bytes()))
    return hasher.hexdigest()


def dataset_hash(out_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(out_dir.glob("*.jsonl")):
        hasher.update(_lf(path.read_bytes()))
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the hash.txt files")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also rewrite datasets whose hash currently matches (default: only fix mismatches)",
    )
    args = parser.parse_args()

    targets = [(d, benchmark_hash) for d in BENCHMARK_DIRS]
    targets += [(d, dataset_hash) for d in DATASET_DIRS]

    drift: list[str] = []
    written: list[str] = []
    ok: list[str] = []

    for rel, fn in targets:
        p = ROOT / rel
        if not p.is_dir():
            drift.append(f"{rel}: directory missing")
            continue
        calc = fn(p)
        hf = p / "hash.txt"
        stored = hf.read_text(encoding="utf-8").strip() if hf.exists() else None
        if stored == calc:
            ok.append(rel)
            if args.write and args.all:
                hf.write_text(calc + "\n", encoding="utf-8", newline="\n")
                written.append(rel)
            continue
        drift.append(f"{rel}: recorded {stored or '(none)'} != actual {calc}")
        if args.write:
            hf.write_text(calc + "\n", encoding="utf-8", newline="\n")
            written.append(rel)

    for r in sorted(ok):
        print(f"  OK      {r}")
    for d in drift:
        print(f"  DRIFT   {d}")
    if written:
        print(f"\nrewrote {len(written)} hash.txt file(s)")
    if drift and not args.write:
        print(
            f"\n{len(drift)} corpus(es) do not match their committed hash.txt. "
            "Re-run with --write."
        )
        return 1
    print(f"\n{len(ok) + len(written)}/{len(targets)} corpora hash-clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
