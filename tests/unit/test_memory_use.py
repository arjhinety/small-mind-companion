"""Pins the pre-registered memory-use metrics by behaviour.

``docs/preregistrations/study002-memory-use.md`` fixes these before any Study 002 run (amended
2026-11-13, ERRATA E43). The literals come from that document; a change here needs a dated
amendment there.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from onebee.evaluation.metrics.memory_use import (
    accuracy_when_retrieved,
    all_gold_retrieved,
    gold_rank,
    gold_retrieval_rate,
    gold_retrieved,
    memories_from_turns,
    source_turn_map,
    stale_fact_only,
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
        "u1": frozenset({"p_s0_t1"}),
        "u2": frozenset({"p_s0_t2"}),
        "u3": frozenset({"p_s0_t9"}),
        "u4": frozenset({"p_s1_t3"}),  # an original fact
        "u5": frozenset({"p_s5_t7"}),  # its correction
        "u6": frozenset({"p_s5_t8"}),  # an unrelated correction
    }
    stale = memories_from_turns(["p_s1_t3"], turn_map)
    update = memories_from_turns(["p_s5_t7"], turn_map)

    def plain(gold: list[str]) -> tuple[frozenset[str], list[frozenset[str]]]:
        groups = [memories_from_turns([g], turn_map) for g in gold]
        return memories_from_turns(gold, turn_map), groups

    probes = [  # (retrieved, needed, groups, stale, judge correct)
        (["u3", "u1"], *plain(["p_s0_t1"]), frozenset(), True),  # hit at rank 2
        (["u2"], *plain(["p_s0_t2", "p_s0_t5"]), frozenset(), False),  # hit, not every gold turn
        (["u3"], *plain(["p_s0_t3"]), frozenset(), True),  # miss, correct anyway
        (["u4", "u6"], update, [stale, update], stale, False),  # original fact only: a miss
        (["u1", "u5"], update, [stale, update], stale, True),  # the correction at rank 2
    ]
    hits = [gold_retrieved(ids, need, turn_map) for ids, need, *_ in probes]
    ranks = [gold_rank(ids, need, turn_map) for ids, need, *_ in probes]
    all_gold = [all_gold_retrieved(ids, groups, turn_map) for ids, _, groups, *_ in probes]
    stale_only = [stale_fact_only(ids, st, need, turn_map) for ids, need, _, st, _ in probes]
    correct = [ok for *_, ok in probes]
    assert hits == [True, True, False, False, True]
    assert ranks == [2, 1, None, None, 2]
    assert all_gold == [True, False, False, False, False]
    assert stale_only == [False, False, False, True, False]
    assert gold_retrieval_rate(hits) == 0.6
    assert gold_retrieval_rate(all_gold) == 0.2
    assert accuracy_when_retrieved(hits, correct) == (2, 3, 2 / 3)
    assert sum(ok and not hit for hit, ok in zip(hits, correct, strict=True)) == 1


def test_unknown_memory_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        gold_retrieved(["not-in-store"], frozenset({"u1"}), {"u1": frozenset({"t1"})})


def test_gold_rank_validates_every_id() -> None:
    with pytest.raises(KeyError):
        gold_rank(["u1", "not-in-store"], frozenset({"u1"}), {"u1": frozenset({"t1"})})


def test_empty_inputs_are_none() -> None:
    assert gold_retrieval_rate([]) is None
    assert accuracy_when_retrieved([False, False], [True, False]) == (0, 0, None)


def test_parallel_lists_required() -> None:
    with pytest.raises(ValueError):
        accuracy_when_retrieved([True], [True, False])
