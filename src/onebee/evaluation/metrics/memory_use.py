"""Gold-memory retrieval and accuracy-when-retrieved, the replacements for ``mur``.

Pre-registered in ``docs/preregistrations/study002-memory-use.md``; change only by amendment there.

Probes name their supporting evidence as benchmark turn ids (``p000_s003_t008``). The memory store
gives each memory a random UUID, which is why ``mur`` compared ids that never matched (ERRATA E36).
Each stored memory keeps the turns it was extracted from in ``provenance.turn_ids``, so a retrieved
memory is mapped back to its source turns through the store the run used.

Standard library only, so ``scripts/validate.py`` can load this file by path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path


def source_turn_map(db_paths: Iterable[Path]) -> dict[str, frozenset[str]]:
    """Memory id -> the benchmark turn ids it was extracted from, read from the stores read-only.

    ``immutable=1`` reads the committed ``.db`` file alone, ignoring any local ``-wal`` file, so
    the result is the same on a fresh clone.
    """
    mapping: dict[str, frozenset[str]] = {}
    for path in db_paths:
        uri = f"file:{Path(path).resolve().as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            for memory_id, provenance in conn.execute("SELECT id, provenance FROM memory"):
                turns = json.loads(provenance or "{}").get("turn_ids") or []
                mapping[memory_id] = frozenset(turns)
        finally:
            conn.close()
    return mapping


_TURN_ID = re.compile(r"_s(\d+)_t(\d+)$")

# PMB category whose two gold turns are a fact and its later update; only the update answers it.
UPDATE_CATEGORY = "outdated_fact"


def _turn_order(turn_id: str) -> tuple[int, int]:
    match = _TURN_ID.search(turn_id)
    if match is None:
        raise ValueError(f"cannot read session and turn from {turn_id!r}")
    return int(match.group(1)), int(match.group(2))


def needed_turns(gold_turn_ids: Iterable[str], category: str) -> frozenset[str]:
    """The gold turns a retrieved memory must come from for the probe to count as gold-retrieved.

    For ``outdated_fact`` probes the gold turns are the original fact and its update, and only the
    update (the latest gold turn) answers the question; retrieving the stale fact alone is not a
    hit. For every other category any gold turn will do.
    """
    gold = list(gold_turn_ids)
    if category == UPDATE_CATEGORY and gold:
        return frozenset({max(gold, key=_turn_order)})
    return frozenset(gold)


def _turns_of(memory_id: str, turn_map: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """A retrieved id missing from ``turn_map`` is an error, not a miss: it means the store given
    is not the one the run used."""
    if memory_id not in turn_map:
        raise KeyError(f"retrieved memory {memory_id!r} is not in the store given")
    return turn_map[memory_id]


def gold_retrieved(
    retrieved_memory_ids: Iterable[str],
    gold_turn_ids: Iterable[str],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """Whether any retrieved memory was extracted from one of ``gold_turn_ids``.

    Pass ``needed_turns(...)`` as ``gold_turn_ids`` so update probes require the update.
    """
    return gold_rank(retrieved_memory_ids, gold_turn_ids, turn_map) is not None


def gold_rank(
    retrieved_memory_ids: Iterable[str],
    gold_turn_ids: Iterable[str],
    turn_map: Mapping[str, frozenset[str]],
) -> int | None:
    """1-based position of the first retrieved memory from a gold turn, or ``None``."""
    gold = set(gold_turn_ids)
    ids = list(retrieved_memory_ids)
    for position, memory_id in enumerate(ids, start=1):
        if _turns_of(memory_id, turn_map) & gold:
            return position
    for memory_id in ids:  # still validate every id, so a wrong store is never a silent miss
        _turns_of(memory_id, turn_map)
    return None


def stale_fact_only(
    retrieved_memory_ids: Iterable[str],
    gold_turn_ids: Iterable[str],
    category: str,
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """For an update probe: the stale fact was retrieved but the update was not."""
    if category != UPDATE_CATEGORY:
        return False
    gold = frozenset(gold_turn_ids)
    update = needed_turns(gold, category)
    ids = list(retrieved_memory_ids)
    return gold_retrieved(ids, gold - update, turn_map) and not gold_retrieved(
        ids, update, turn_map
    )


def all_gold_retrieved(
    retrieved_memory_ids: Iterable[str],
    gold_turn_ids: Iterable[str],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """Whether every gold turn is covered by some retrieved memory (all ``gold_turn_ids``)."""
    covered: set[str] = set()
    for memory_id in retrieved_memory_ids:
        covered |= _turns_of(memory_id, turn_map)
    return set(gold_turn_ids) <= covered


def gold_retrieval_rate(hits: list[bool]) -> float | None:
    """Share of answerable probes whose gold memory was retrieved. A retriever metric: it does not
    depend on the model's answer. ``None`` when there are no answerable probes."""
    return sum(hits) / len(hits) if hits else None


def accuracy_when_retrieved(hits: list[bool], correct: list[bool]) -> tuple[int, int, float | None]:
    """(correct, retrieved, share) over the answerable probes whose gold memory was retrieved.

    ``hits`` and ``correct`` are parallel, one entry per answerable probe. ``share`` is ``None``
    when nothing was retrieved.
    """
    if len(hits) != len(correct):
        raise ValueError("hits and correct must be parallel lists")
    retrieved = sum(hits)
    n_correct = sum(1 for hit, ok in zip(hits, correct, strict=True) if hit and ok)
    return n_correct, retrieved, (n_correct / retrieved if retrieved else None)
