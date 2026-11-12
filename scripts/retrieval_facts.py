"""Facts about Study 001's memory retrieval, read directly from the stores and saved runs (E42).

POST-HOC, like ``scripts/retrieval_audit.py``, but nothing here replays the retriever: every figure
is counted from committed files, so this runs in seconds with the standard library only, and
``scripts/validate.py`` recomputes it on every run.

* **stores** -- ``event_time`` values in ``data/stores/pmb_v0_full`` (opened read-only and
  immutable, so the committed files are never touched), and the memories whose recency falls back
  to the wall-clock ``created_at`` because ``event_time`` is 0 or missing (``fusion.py:70``).
* **extraction ceiling** -- answerable probes with at least one memory extracted from a gold turn.
* **saved runs** -- list sizes and gold-memory hits in each memory run's ``raw.jsonl``; gold means a
  retrieved memory whose ``provenance.turn_ids`` include one of the probe's gold supporting turns.
* **k-sweep** -- how the k=8 and k=16 runs differ: identical retrieved lists, identical responses,
  and judge verdict changes among them.

Usage::

    python scripts/retrieval_facts.py           # (re)write the output JSON
    python scripts/retrieval_facts.py --check   # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "audit" / "retrieval" / "retrieval_facts.json"
STORES = ROOT / "data" / "stores" / "pmb_v0_full"
PROBES = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
MEMORY_RUNS = (
    "results/v0.1/D_memory",
    "results/v0.1/E_sft_memory",
    "results/v1_scale/E_sft_memory",
    "results/v1_scale/E_distill",
)
KSWEEP = ROOT / "results" / "v0.1" / "ksweep"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def build() -> dict[str, Any]:
    turns: dict[str, set[str]] = {}
    event_times: list[int] = []
    fallback: dict[str, str] = {}
    for db in sorted(STORES.glob("p*.db")):
        conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            for memory_id, event_time, created_at, provenance in conn.execute(
                "SELECT id, event_time, created_at, provenance FROM memory"
            ):
                turns[memory_id] = set(json.loads(provenance or "{}").get("turn_ids") or [])
                event_times.append(event_time or 0)
                if not event_time:
                    fallback[memory_id] = f"{db.stem} {_day(created_at)}"
        finally:
            conn.close()

    answerable = [p for p in _rows(PROBES) if p["answerable"]]
    extracted = set().union(*turns.values())

    def is_gold(memory_id: str, probe: dict[str, Any]) -> bool:
        return bool(turns[memory_id] & set(probe["gold_supporting_memory_ids"]))

    runs: dict[str, Any] = {}
    for run in MEMORY_RUNS:
        rows = _rows(ROOT / run / "raw.jsonl")
        sizes = Counter(len(r["retrieved_memory_ids"]) for r in rows)
        hits = sorted(
            r["probe"]["probe_id"]
            for r in rows
            if r["probe"]["answerable"]
            and any(is_gold(m, r["probe"]) for m in r["retrieved_memory_ids"])
        )
        runs[run] = {
            "lists": len(rows),
            "list_size_min": min(sizes),
            "list_size_max": max(sizes),
            "list_sizes": {str(k): sizes[k] for k in sorted(sizes)},
            "gold_retrieved": len(hits),
            "hit_set_sha256": hashlib.sha256("\n".join(hits).encode()).hexdigest(),
        }

    k8 = {r["probe"]["probe_id"]: r for r in _rows(KSWEEP / "k8" / "raw.jsonl")}
    k16 = {r["probe"]["probe_id"]: r for r in _rows(KSWEEP / "k16" / "raw.jsonl")}

    def verdict_changes(ids: list[str]) -> dict[str, int]:
        answerable_ids = [p for p in ids if k8[p]["probe"]["answerable"]]
        lost = sum(
            1 for p in answerable_ids if k8[p]["lenient_correct"] and not k16[p]["lenient_correct"]
        )
        gained = sum(
            1 for p in answerable_ids if not k8[p]["lenient_correct"] and k16[p]["lenient_correct"]
        )
        return {
            "probes": len(ids),
            "answerable": len(answerable_ids),
            "lost": lost,
            "gained": gained,
        }

    same_list = [p for p in k8 if k8[p]["retrieved_memory_ids"] == k16[p]["retrieved_memory_ids"]]
    same_response = [p for p in k8 if k8[p]["response"] == k16[p]["response"]]

    return {
        "artifact_kind": "retrieval-facts-posthoc",
        "generated_by": "scripts/retrieval_facts.py",
        "status": "post-hoc counts from committed stores and runs; no replay, no model",
        "stores": {
            "memories": len(turns),
            "event_time_nonzero": sum(1 for t in event_times if t),
            "event_time_max": max(event_times),
            "recency_fallback_to_created_at": dict(sorted(fallback.items())),
        },
        "answerable_probes": len(answerable),
        "answerable_gold_turn_extracted": sum(
            1 for p in answerable if set(p["gold_supporting_memory_ids"]) & extracted
        ),
        "saved_runs": runs,
        "saved_runs_identical_hit_sets": len({r["hit_set_sha256"] for r in runs.values()}) == 1,
        "ksweep_k8_vs_k16": {
            "probes": len(k8),
            "identical_lists": verdict_changes(same_list),
            "identical_responses": verdict_changes(same_response),
            "all": verdict_changes(list(k8)),
        },
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
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
