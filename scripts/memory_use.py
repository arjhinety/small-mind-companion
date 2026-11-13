"""What ``mur`` would have measured, and whether models use a retrieved memory (Study 002 item 12).

POST-HOC. These figures were computed after reading Study 001's results; they describe those runs
and validate nothing. The Study 002 metrics are fixed in
``docs/preregistrations/study002-memory-use.md``.

``mur`` was 0.0 in every run because probes cite benchmark turn ids while the store assigns random
UUIDs (ERRATA E36). Each stored memory records its source turns (``provenance.turn_ids``), so this
script maps every retrieved memory in each full-harness run's ``raw.jsonl`` back to its turns
through the committed stores in ``data/stores/pmb_v0_full/``, then reports per run:

* **gold retrieval** -- answerable probes where a retrieved memory came from a gold supporting turn.
  The retriever decides this, not the model; ``hit_set_sha256`` shows whether runs share the set.
* **accuracy when retrieved** -- of those probes, how many the judge (``lenient_correct``) and
  ``contains_answer`` mark correct; and how many are correct *without* the gold memory retrieved.

``memory_use.py`` and ``rule.py`` are loaded by path (standard library only), so this script and
``scripts/validate.py`` need no third-party package. Read-only.

Usage::

    uv run python scripts/memory_use.py           # (re)write the output JSON
    uv run python scripts/memory_use.py --check   # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "audit" / "memory_use" / "memory_use.json"
STORES = ROOT / "data" / "stores" / "pmb_v0_full"
PROBES = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
REGISTRY = ROOT / "results" / "registry.jsonl"


def _load(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _builder_drops(run: str, rows: list[dict[str, Any]]) -> int:
    """Memories the Study 001 context builder would drop from this run's saved lists.

    Rebuilds each prompt with ``ContextBuilder`` and the run script's persona, reading memory rows
    from the committed stores read-only. 0 means what reached the prompt is the retriever's list.
    """
    import sqlite3

    from onebee.context.builder import ContextBuilder

    script = (ROOT / "run_system_d.py").read_text(encoding="utf-8")
    start = script.index("COMPANION_PERSONA = {")
    namespace: dict[str, Any] = {}
    end = script.index("\n}\n", start) + 3
    exec(script[start:end], namespace)  # noqa: S102 - a literal from our own run script
    persona = namespace["COMPANION_PERSONA"]
    records: dict[str, dict[str, Any]] = {}
    for db in sorted(STORES.glob("p*.db")):
        uri = f"file:{db.resolve().as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            records.update({row["id"]: dict(row) for row in conn.execute("SELECT * FROM memory")})
        finally:
            conn.close()
    builder = ContextBuilder()
    dropped = 0
    for r in rows:
        ids = r["retrieved_memory_ids"]
        if not ids:
            continue
        _, trace = builder.build(
            turn_id=r["probe"]["probe_id"],
            persona=persona,
            profile={},
            boundaries=[],
            retrieved_memories=[records[i] for i in ids],
            recent_turns=[],
            user_turn="",
        )
        dropped += len(trace.dropped_memory_ids) + (
            trace.kept_memory_ids != ids[: len(trace.kept_memory_ids)]
        )
    return dropped


def build() -> dict[str, Any]:
    mu = _load("onebee_memory_use_by_path", "src/onebee/evaluation/metrics/memory_use.py")
    rule = _load("onebee_rule_by_path", "src/onebee/evaluation/graders/rule.py")
    dbs = sorted(STORES.glob("p*.db"))
    turn_map = mu.source_turn_map(dbs)
    extracted = set().union(*turn_map.values())
    answerable = [p for p in _rows(PROBES) if p["answerable"]]
    runs_meta = [r for r in _rows(REGISTRY) if r.get("run_kind") == "pmb_full_harness"]

    runs: dict[str, Any] = {}
    for meta in runs_meta:
        run = meta["run_path"]
        all_rows = _rows(ROOT / run / "raw.jsonl")
        rows = [r for r in all_rows if r["probe"]["answerable"]]

        def gold(r: dict[str, Any]) -> list[str]:
            return list(r["probe"]["gold_supporting_memory_ids"])

        def needed(r: dict[str, Any]) -> frozenset[str]:
            return frozenset(mu.needed_turns(gold(r), r["probe"]["category"]))

        hits = [mu.gold_retrieved(r["retrieved_memory_ids"], needed(r), turn_map) for r in rows]
        any_gold = [mu.gold_retrieved(r["retrieved_memory_ids"], gold(r), turn_map) for r in rows]
        all_gold = [
            mu.all_gold_retrieved(r["retrieved_memory_ids"], gold(r), turn_map) for r in rows
        ]
        stale = [
            mu.stale_fact_only(r["retrieved_memory_ids"], gold(r), r["probe"]["category"], turn_map)
            for r in rows
        ]
        ranks = [mu.gold_rank(r["retrieved_memory_ids"], needed(r), turn_map) for r in rows]
        found = [k for k in ranks if k is not None]
        judge = [bool(r["lenient_correct"]) for r in rows]
        contain = [rule.contains_answer(r["response"], r["probe"]["gold_answer"]) for r in rows]
        j_ok, retrieved, j_share = mu.accuracy_when_retrieved(hits, judge)
        c_ok, _, c_share = mu.accuracy_when_retrieved(hits, contain)
        hit_ids = sorted(r["probe"]["probe_id"] for r, h in zip(rows, hits, strict=True) if h)
        runs[run] = {
            "answerable": len(rows),
            "uses_memory": any(r["retrieved_memory_ids"] for r in rows),
            "gold_retrieved": retrieved,
            "gold_retrieval_rate": None if not rows else round(retrieved / len(rows), 4),
            "hit_set_sha256": hashlib.sha256("\n".join(hit_ids).encode()).hexdigest(),
            "prompt_builder_dropped_or_reordered": _builder_drops(run, all_rows),
            "any_gold_turn_retrieved": sum(any_gold),
            "all_gold_retrieved": sum(all_gold),
            "stale_fact_only": sum(stale),
            "gold_rank_counts": {str(k): found.count(k) for k in sorted(set(found))},
            "judge_correct_when_retrieved": j_ok,
            "judge_accuracy_when_retrieved": None if j_share is None else round(j_share, 4),
            "judge_correct_without_retrieval": sum(
                1 for h, ok in zip(hits, judge, strict=True) if ok and not h
            ),
            "contain_correct_when_retrieved": c_ok,
            "contain_accuracy_when_retrieved": None if c_share is None else round(c_share, 4),
            "contain_correct_without_retrieval": sum(
                1 for h, ok in zip(hits, contain, strict=True) if ok and not h
            ),
        }

    return {
        "artifact_kind": "memory-use-posthoc",
        "generated_by": "scripts/memory_use.py",
        "status": "post-hoc: computed after reading Study 001 results; not a reported metric",
        "stores": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in dbs
        },
        "memories": len(turn_map),
        "answerable_probes": len(answerable),
        "answerable_gold_turn_extracted": sum(
            1
            for p in answerable
            if mu.needed_turns(p["gold_supporting_memory_ids"], p["category"]) & extracted
        ),
        "answerable_any_gold_turn_extracted": sum(
            1 for p in answerable if set(p["gold_supporting_memory_ids"]) & extracted
        ),
        "runs": runs,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = parser.parse_args(argv)
    text = serialise(build())
    rel = OUTPUT.relative_to(ROOT).as_posix()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{rel} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{rel} is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
