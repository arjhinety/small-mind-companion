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
    all_gold_retrieved,
    gold_rank,
    gold_retrieval_rate,
    gold_retrieved,
    needed_turns,
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
        "u4": frozenset({"p_s1_t3"}),  # a fact that is later updated
        "u5": frozenset({"p_s5_t7"}),  # its update
    }
    probes = [  # (retrieved, gold turns, category, judge correct)
        (["u3", "u1"], ["p_s0_t1"], "factual", True),  # hit at rank 2, correct
        (["u2"], ["p_s0_t2", "p_s0_t5"], "factual", False),  # hit, not every gold turn, wrong
        (["u3"], ["p_s0_t3"], "factual", True),  # miss, correct anyway
        (["u4"], ["p_s1_t3", "p_s5_t7"], "outdated_fact", False),  # stale fact only: a miss
        (["u1", "u5"], ["p_s1_t3", "p_s5_t7"], "outdated_fact", True),  # update at rank 2
    ]
    needed = [needed_turns(gold, cat) for _, gold, cat, _ in probes]
    hits = [
        gold_retrieved(ids, need, turn_map) for (ids, *_), need in zip(probes, needed, strict=True)
    ]
    ranks = [gold_rank(ids, need, turn_map) for (ids, *_), need in zip(probes, needed, strict=True)]
    all_gold = [all_gold_retrieved(ids, gold, turn_map) for ids, gold, _, _ in probes]
    stale = [stale_fact_only(ids, gold, cat, turn_map) for ids, gold, cat, _ in probes]
    correct = [ok for *_, ok in probes]
    assert hits == [True, True, False, False, True]
    assert ranks == [2, 1, None, None, 2]
    assert all_gold == [True, False, False, False, False]
    assert stale == [False, False, False, True, False]
    assert gold_retrieval_rate(hits) == 0.6
    assert gold_retrieval_rate(all_gold) == 0.2
    assert accuracy_when_retrieved(hits, correct) == (2, 3, 2 / 3)
    assert sum(ok and not hit for hit, ok in zip(hits, correct, strict=True)) == 1


def test_update_probe_needs_the_latest_gold_turn() -> None:
    assert needed_turns(["p000_s000_t008", "p000_s5_t099"], "outdated_fact") == frozenset(
        {"p000_s5_t099"}
    )
    assert needed_turns(["p000_s000_t008", "p000_s5_t099"], "factual") == frozenset(
        {"p000_s000_t008", "p000_s5_t099"}
    )


def test_gold_rank_still_validates_ids_after_a_miss() -> None:
    with pytest.raises(KeyError):
        gold_rank(["u1", "not-in-store"], ["t9"], {"u1": frozenset({"t1"})})


def test_unknown_memory_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        gold_retrieved(["not-in-store"], ["t1"], {"u1": frozenset({"t1"})})


def test_empty_inputs_are_none() -> None:
    assert gold_retrieval_rate([]) is None
    assert accuracy_when_retrieved([False, False], [True, False]) == (0, 0, None)


def test_parallel_lists_required() -> None:
    with pytest.raises(ValueError):
        accuracy_when_retrieved([True], [True, False])
