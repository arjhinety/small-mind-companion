"""Gold-memory retrieval and accuracy-when-retrieved, the replacements for ``mur``.

Pre-registered in ``docs/preregistrations/study002-memory-use.md``; change only by amendment there
(amended 2026-11-13, ERRATA E43).

Probes name their supporting evidence as benchmark turn ids (``p000_s003_t008``). The memory store
gives each memory a random UUID, which is why ``mur`` compared ids that never matched (ERRATA E36).
Each stored memory keeps the turns it was extracted from in ``provenance.turn_ids``, so the memories
that count for a probe (every memory extracted from one of its gold turns) are found through the
store the run used. On PMB only the probes whose answer key is sound are counted (ERRATA E43):
update probes' turn ids cannot identify the correction they mean, and some gold answers are not
the value the conversation ends with.

Every function below works on sets of memory ids. Standard library only, so
``scripts/validate.py`` can load this file by path.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path


def _read_memories(db_paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in db_paths:
        uri = f"file:{Path(path).resolve().as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            for memory_id, provenance in conn.execute("SELECT id, provenance FROM memory"):
                rows[memory_id] = json.loads(provenance or "{}")
        finally:
            conn.close()
    return rows


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def source_turn_map(db_paths: Iterable[Path]) -> dict[str, frozenset[str]]:
    """Memory id -> the benchmark turn ids it was extracted from, read from the stores read-only.

    ``immutable=1`` reads the committed ``.db`` file alone, ignoring any local ``-wal`` file, so
    the result is the same on a fresh clone.
    """
    return {
        mid: frozenset(str(turn) for turn in _as_list(prov.get("turn_ids")))
        for mid, prov in _read_memories(db_paths).items()
    }


def memories_from_turns(
    turn_ids: Iterable[str], turn_map: Mapping[str, frozenset[str]]
) -> frozenset[str]:
    """Every memory extracted from at least one of ``turn_ids``."""
    turns = set(turn_ids)
    return frozenset(mid for mid, source in turn_map.items() if source & turns)


def _check_ids(retrieved: list[str], turn_map: Mapping[str, frozenset[str]]) -> None:
    """A retrieved id missing from ``turn_map`` is an error, not a miss: it means the store given
    is not the one the run used."""
    for memory_id in retrieved:
        if memory_id not in turn_map:
            raise KeyError(f"retrieved memory {memory_id!r} is not in the store given")


def gold_rank(
    retrieved_memory_ids: Iterable[str],
    needed: frozenset[str],
    turn_map: Mapping[str, frozenset[str]],
) -> int | None:
    """1-based position of the first retrieved memory in ``needed``, or ``None``."""
    ids = list(retrieved_memory_ids)
    _check_ids(ids, turn_map)
    return next((i for i, mid in enumerate(ids, start=1) if mid in needed), None)


def gold_retrieved(
    retrieved_memory_ids: Iterable[str],
    needed: frozenset[str],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """Whether any retrieved memory is one the probe needs."""
    return gold_rank(retrieved_memory_ids, needed, turn_map) is not None


def all_gold_retrieved(
    retrieved_memory_ids: Iterable[str],
    groups: Iterable[frozenset[str]],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """Whether every group of gold memories (one per gold turn) has a retrieved member."""
    ids = list(retrieved_memory_ids)
    _check_ids(ids, turn_map)
    got = set(ids)
    return all(got & group for group in groups)


def stale_fact_only(
    retrieved_memory_ids: Iterable[str],
    stale: frozenset[str],
    needed: frozenset[str],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """For an update probe: a memory of the stale fact was retrieved, the needed update was not."""
    ids = list(retrieved_memory_ids)
    _check_ids(ids, turn_map)
    got = set(ids)
    return bool(got & stale) and not got & needed


def gold_retrieval_rate(hits: list[bool]) -> float | None:
    """Share of counted answerable probes whose needed memory was retrieved. A retriever metric:
    it does not depend on the model's answer. ``None`` when there are no probes."""
    return sum(hits) / len(hits) if hits else None


def accuracy_when_retrieved(hits: list[bool], correct: list[bool]) -> tuple[int, int, float | None]:
    """(correct, retrieved, share) over the probes whose needed memory was retrieved.

    ``hits`` and ``correct`` are parallel, one entry per counted answerable probe. ``share`` is
    ``None`` when nothing was retrieved.
    """
    if len(hits) != len(correct):
        raise ValueError("hits and correct must be parallel lists")
    retrieved = sum(hits)
    n_correct = sum(1 for hit, ok in zip(hits, correct, strict=True) if hit and ok)
    return n_correct, retrieved, (n_correct / retrieved if retrieved else None)
