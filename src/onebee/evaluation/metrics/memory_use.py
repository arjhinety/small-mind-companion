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


def gold_retrieved(
    retrieved_memory_ids: Iterable[str],
    gold_turn_ids: Iterable[str],
    turn_map: Mapping[str, frozenset[str]],
) -> bool:
    """Whether any retrieved memory was extracted from a gold supporting turn.

    A retrieved id missing from ``turn_map`` is an error, not a miss: it means the store given is
    not the one the run used.
    """
    gold = set(gold_turn_ids)
    for memory_id in retrieved_memory_ids:
        if memory_id not in turn_map:
            raise KeyError(f"retrieved memory {memory_id!r} is not in the store given")
        if turn_map[memory_id] & gold:
            return True
    return False


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
