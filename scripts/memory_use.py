"""What ``mur`` would have measured, and whether models use a retrieved memory (Study 002 item 12).

POST-HOC. These figures were computed after reading Study 001's results; they describe those runs
and validate nothing. The Study 002 metrics are fixed in
``docs/preregistrations/study002-memory-use.md``.

``mur`` was 0.0 in every run because probes cite benchmark turn ids while the store assigns random
UUIDs (ERRATA E36). Each stored memory records its source turns (``provenance.turn_ids``), so this
script maps every retrieved memory in each full-harness run's ``raw.jsonl`` back to its turns
through the committed stores in ``data/stores/pmb_v0_full/``, then reports per run:

* **gold retrieval** -- counted probes where a retrieved memory came from a gold supporting turn.
  The retriever decides this, not the model; ``hit_set_sha256`` shows whether runs share the set.
  Counted probes are those whose answer key ``scripts/pmb_answer_key_audit.py`` finds sound
  (ERRATA E43; amendment 1 of the pre-registration). ``original_rule`` keeps the figures the
  pre-registration was first written with, and ``any_gold_turn_all_answerable`` the E42 count.
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
from math import comb
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


def _original_rule(mu: Any, rows: list[dict[str, Any]], turn_map: dict[str, Any]) -> dict[str, Any]:
    """The rule as first registered (PR #18): for outdated_fact the needed turn is the latest gold
    turn, which on PMB is the shared update id (E43). Kept so the registration's original figures
    stay checkable; superseded by its 2026-11-13 amendment."""

    def order(turn_id: str) -> tuple[int, int]:
        session, turn = turn_id.rsplit("_s", 1)[1].split("_t")
        return int(session), int(turn)

    hits, stale, judge = [], [], []
    for r in rows:
        gold = list(r["probe"]["gold_supporting_memory_ids"])
        latest = [max(gold, key=order)] if r["probe"]["category"] == "outdated_fact" else gold
        need = mu.memories_from_turns(latest, turn_map)
        ids = r["retrieved_memory_ids"]
        hit = mu.gold_retrieved(ids, need, turn_map)
        old = mu.memories_from_turns([g for g in gold if g not in latest], turn_map)
        hits.append(hit)
        stale.append(bool(set(ids) & old) and not hit)
        judge.append(bool(r["lenient_correct"]))
    ok, retrieved, share = mu.accuracy_when_retrieved(hits, judge)
    return {
        "gold_retrieved": retrieved,
        "stale_fact_only": sum(stale),
        "judge_accuracy_when_retrieved": None if share is None else round(share, 4),
        "judge_correct_without_retrieval": sum(1 for h, j in zip(hits, judge) if j and not h),
    }


def build() -> dict[str, Any]:
    mu = _load("onebee_memory_use_by_path", "src/onebee/evaluation/metrics/memory_use.py")
    rule = _load("onebee_rule_by_path", "src/onebee/evaluation/graders/rule.py")
    audit = _load("pmb_answer_key_audit_by_path", "scripts/pmb_answer_key_audit.py").build()
    sound = {pid for pid, s in audit["probes"].items() if s["sound"]}
    dbs = sorted(STORES.glob("p*.db"))
    turn_map = mu.source_turn_map(dbs)
    persona_of = {mid: db.stem for db in dbs for mid in mu.source_turn_map([db])}
    store_size = {db.stem: sum(1 for p in persona_of.values() if p == db.stem) for db in dbs}
    answerable = [p for p in _rows(PROBES) if p["answerable"]]

    sets: dict[str, dict[str, Any]] = {}
    for p in answerable:
        gold = list(p["gold_supporting_memory_ids"])
        sets[p["probe_id"]] = {
            "needed": mu.memories_from_turns(gold, turn_map),
            "groups": [mu.memories_from_turns([g], turn_map) for g in gold],
            "counted": p["probe_id"] in sound,
        }

    def chance(lists: dict[str, list[str]], key: str, only_counted: bool) -> float:
        """Mean P(a random pick of the run's list size from the persona's store hits ``key``)."""
        probes = [p for p in answerable if sets[p["probe_id"]]["counted"] or not only_counted]
        total = 0.0
        for p in probes:
            n, k = store_size[p["persona_id"]], len(lists[p["probe_id"]])
            g = len(sets[p["probe_id"]][key])
            total += 0.0 if g == 0 else 1 - comb(n - g, k) / comb(n, k)
        return round(total / len(probes), 4)

    runs_meta = [r for r in _rows(REGISTRY) if r.get("run_kind") == "pmb_full_harness"]
    runs: dict[str, Any] = {}
    for meta in runs_meta:
        run = meta["run_path"]
        all_rows = _rows(ROOT / run / "raw.jsonl")
        every = [r for r in all_rows if r["probe"]["answerable"]]
        rows = [r for r in every if sets[r["probe"]["probe_id"]]["counted"]]
        out = [r for r in every if not sets[r["probe"]["probe_id"]]["counted"]]

        def s(r: dict[str, Any]) -> dict[str, Any]:
            return sets[r["probe"]["probe_id"]]

        ret = [r["retrieved_memory_ids"] for r in rows]
        hits = [mu.gold_retrieved(ids, s(r)["needed"], turn_map) for r, ids in zip(rows, ret)]
        ranks = [mu.gold_rank(ids, s(r)["needed"], turn_map) for r, ids in zip(rows, ret)]
        all_gold = [
            mu.all_gold_retrieved(ids, s(r)["groups"], turn_map) for r, ids in zip(rows, ret)
        ]
        any_gold = [
            mu.gold_retrieved(r["retrieved_memory_ids"], s(r)["needed"], turn_map) for r in every
        ]
        found = [k for k in ranks if k is not None]
        judge = [bool(r["lenient_correct"]) for r in rows]
        contain = [rule.contains_answer(r["response"], r["probe"]["gold_answer"]) for r in rows]
        j_ok, retrieved, j_share = mu.accuracy_when_retrieved(hits, judge)
        c_ok, _, c_share = mu.accuracy_when_retrieved(hits, contain)
        hit_ids = sorted(r["probe"]["probe_id"] for r, h in zip(rows, hits, strict=True) if h)
        uses_memory = any(r["retrieved_memory_ids"] for r in all_rows)
        lists = {r["probe"]["probe_id"]: r["retrieved_memory_ids"] for r in every}
        runs[run] = {
            "answerable": len(every),
            "counted": len(rows),
            "uses_memory": uses_memory,
            "gold_retrieved": retrieved,
            "gold_retrieval_rate": None if not rows else round(retrieved / len(rows), 4),
            "chance_rate": chance(lists, "needed", True) if uses_memory else None,
            "hit_set_sha256": hashlib.sha256(chr(10).join(hit_ids).encode()).hexdigest(),
            "prompt_builder_dropped_or_reordered": _builder_drops(run, all_rows),
            "all_gold_retrieved": sum(all_gold),
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
            "excluded_judge_correct": sum(bool(r["lenient_correct"]) for r in out),
            "original_rule": _original_rule(mu, every, turn_map),
            "any_gold_turn_all_answerable": {
                "retrieved": sum(any_gold),
                "chance_rate": chance(lists, "needed", False) if uses_memory else None,
            },
        }

    counted = [p for p in answerable if sets[p["probe_id"]]["counted"]]
    return {
        "artifact_kind": "memory-use-posthoc",
        "generated_by": "scripts/memory_use.py",
        "status": "post-hoc: computed after reading Study 001 results; not a reported metric",
        "stores": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in dbs
        },
        "memories": len(turn_map),
        "answerable_probes": len(answerable),
        "counted_probes": len(counted),
        "excluded_probes": len(answerable) - len(counted),
        "counted_with_needed_memory_extracted": sum(
            1 for p in counted if sets[p["probe_id"]]["needed"]
        ),
        "answerable_with_any_gold_turn_extracted": sum(
            1 for p in answerable if sets[p["probe_id"]]["needed"]
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
