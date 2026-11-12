"""Pins the pre-registered memory-use metrics by behaviour.

``docs/preregistrations/study002-memory-use.md`` fixes these before any Study 002 run. The literals
come from that document; a change here needs a dated amendment there.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from onebee.evaluation.metrics.memory_use import (
    accuracy_when_retrieved,
    gold_retrieval_rate,
    gold_retrieved,
    source_turn_map,
)


def _store(path: Path, memories: dict[str, list[str]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memory (id TEXT PRIMARY KEY, provenance TEXT)")
    conn.executemany(
        "INSERT INTO memory VALUES (?, ?)",
        [(mid, json.dumps({"turn_ids": turns})) for mid, turns in memories.items()],
    )
    conn.commit()
    conn.close()
    return path


def test_source_turn_map_reads_provenance(tmp_path: Path) -> None:
    db = _store(tmp_path / "p000.db", {"uuid-a": ["p000_s000_t002"], "uuid-b": []})
    assert source_turn_map([db]) == {
        "uuid-a": frozenset({"p000_s000_t002"}),
        "uuid-b": frozenset(),
    }


def test_worked_example() -> None:
    turn_map = {
        "u1": frozenset({"t1"}),
        "u2": frozenset({"t2"}),
        "u3": frozenset({"t9"}),
    }
    probes = [  # (retrieved, gold turns, judge correct)
        (["u1", "u3"], ["t1"], True),  # gold retrieved, correct
        (["u2"], ["t2", "t5"], False),  # gold retrieved, wrong
        (["u3"], ["t3"], True),  # gold not retrieved, correct anyway
        ([], ["t4"], False),  # nothing retrieved
    ]
    hits = [gold_retrieved(ids, gold, turn_map) for ids, gold, _ in probes]
    correct = [ok for _, _, ok in probes]
    assert hits == [True, True, False, False]
    assert gold_retrieval_rate(hits) == 0.5
    assert accuracy_when_retrieved(hits, correct) == (1, 2, 0.5)


def test_unknown_memory_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        gold_retrieved(["not-in-store"], ["t1"], {"u1": frozenset({"t1"})})


def test_empty_inputs_are_none() -> None:
    assert gold_retrieval_rate([]) is None
    assert accuracy_when_retrieved([False, False], [True, False]) == (0, 0, None)


def test_parallel_lists_required() -> None:
    with pytest.raises(ValueError):
        accuracy_when_retrieved([True], [True, False])
