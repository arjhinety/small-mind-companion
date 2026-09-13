from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """Turn arbitrary natural-language text into a safe FTS5 MATCH expression.

    FTS5's MATCH argument is a query language, not a literal string — punctuation
    like ``?``, ``'``, ``-``, ``:`` (all common in real questions: "What's my ...?")
    is parsed as query syntax and raises ``sqlite3.OperationalError: fts5: syntax
    error`` on anything but the simplest inputs. Extract word tokens, quote each one
    (so tokens containing FTS5 operator characters are treated as literal terms, not
    operators), and OR them together for reasonable recall on natural questions.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


_FILTERABLE_COLUMNS = {
    "id",
    "tier",
    "status",
    "importance",
    "confidence",
    "supersedes",
    "superseded_by",
    "contradiction_group",
    "schema_version",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _from_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


class MemoryRecord(BaseModel):
    id: str = ""
    tier: str
    content: str
    content_struct: dict[str, Any] | None = None
    embedding_id: int | None = None
    created_at: int
    event_time: int | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    last_accessed: int | None = None
    access_count: int = 0
    importance: float
    confidence: float
    decay_rate: float
    provenance: dict[str, Any]
    emotion_valence: float | None = None
    emotion_arousal: float | None = None
    entities: list[Any] | None = None
    topics: list[Any] | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    contradiction_group: str | None = None
    status: str = "active"
    redaction: int = 0
    schema_version: int = 1

    embedding: list[float] | None = Field(default=None, exclude=True)


class TurnRecord(BaseModel):
    turn_id: str = ""
    session_id: str
    role: str
    text: str
    ts: int
    tokens: int | None = None
    affect: str | None = None
    entities: str | None = None


class SessionRecord(BaseModel):
    id: str = ""
    started: int | None = None
    ended: int | None = None
    summary: str | None = None
    turn_count: int | None = None
    affect_summary: str | None = None


class MemoryStoreProtocol(Protocol):
    def write(self, record: MemoryRecord) -> str: ...

    def search(
        self,
        query: str | None = None,
        tier: str | None = None,
        query_embedding: list[float] | None = None,
        k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]: ...

    def stats(self) -> dict[str, Any]: ...

    def write_turn(self, turn: TurnRecord) -> None: ...

    def write_session(self, session: SessionRecord) -> None: ...

    def get_by_id(self, memory_id: str) -> MemoryRecord | None: ...

    def list_by_tier(
        self, tier: str, status: str = "active", limit: int = 100
    ) -> list[MemoryRecord]: ...


class MemoryStore:
    def __init__(self, db_path: str, embedding_dim: int = 384) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        parent = Path(db_path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._vec_available = False
        try:
            import sqlite_vec

            sqlite_vec.load(self._conn)
            self._vec_available = True
        except (ImportError, Exception):
            self._vec_available = False

        self._init_schema()
        self._vec_table_created = False

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()
        self._conn.executescript(schema_sql)
        self._conn.commit()

    def _ensure_vec_table(self) -> None:
        if self._vec_table_created or not self._vec_available:
            return
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
            f"embedding_id INTEGER, embedding float[{self.embedding_dim}]"
            f")"
        )
        self._conn.commit()
        self._vec_table_created = True

    @staticmethod
    def _generate_id() -> str:
        return str(uuid.uuid4())

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            tier=row["tier"],
            content=row["content"],
            content_struct=_from_json(row["content_struct"]),
            embedding_id=row["embedding_id"],
            created_at=row["created_at"],
            event_time=row["event_time"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            importance=row["importance"],
            confidence=row["confidence"],
            decay_rate=row["decay_rate"],
            provenance=_from_json(row["provenance"]),
            emotion_valence=row["emotion_valence"],
            emotion_arousal=row["emotion_arousal"],
            entities=_from_json(row["entities"]),
            topics=_from_json(row["topics"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            contradiction_group=row["contradiction_group"],
            status=row["status"],
            redaction=row["redaction"],
            schema_version=row["schema_version"],
        )

    def write(self, record: MemoryRecord) -> str:
        if not record.id:
            record.id = self._generate_id()
        if not record.created_at:
            record.created_at = _now_ms()

        data = record.model_dump(exclude={"embedding"})
        data["content_struct"] = _to_json(record.content_struct)
        data["provenance"] = _to_json(record.provenance)
        data["entities"] = _to_json(record.entities)
        data["topics"] = _to_json(record.topics)

        columns = list(data.keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        values = [data[c] for c in columns]

        self._conn.execute(
            f"INSERT OR REPLACE INTO memory ({column_names}) VALUES ({placeholders})",
            values,
        )

        if record.embedding is not None and self._vec_available:
            self._ensure_vec_table()
            if record.embedding_id is None:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(embedding_id), 0) + 1 FROM memory_vec"
                ).fetchone()
                record.embedding_id = row[0]
                self._conn.execute(
                    "UPDATE memory SET embedding_id = ? WHERE id = ?",
                    (record.embedding_id, record.id),
                )
            else:
                self._conn.execute(
                    "DELETE FROM memory_vec WHERE embedding_id = ?",
                    (record.embedding_id,),
                )
            self._conn.execute(
                "INSERT INTO memory_vec(embedding_id, embedding) VALUES (?, ?)",
                (record.embedding_id, json.dumps(record.embedding)),
            )

        self._conn.commit()
        return record.id

    def search(
        self,
        query: str | None = None,
        tier: str | None = None,
        query_embedding: list[float] | None = None,
        k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        seen_ids: set[str] = set()
        records: list[MemoryRecord] = []

        base_where = "WHERE m.redaction = 0"
        base_params: list[Any] = []

        status_value = "active"
        if filters:
            if "status" in filters:
                status_value = filters["status"]
            else:
                extra = {k: v for k, v in filters.items() if k != "status"}
                if extra:
                    for col, val in extra.items():
                        if col not in _FILTERABLE_COLUMNS:
                            raise ValueError(f"filters: unknown or non-filterable column {col!r}")
                        base_where += f" AND m.{col} = ?"
                        base_params.append(val)

        base_where += " AND m.status = ?"
        base_params.append(status_value)

        if tier:
            base_where += " AND m.tier = ?"
            base_params.append(tier)

        self._conn.row_factory = sqlite3.Row

        fts_query = _sanitize_fts_query(query) if query else ""
        if fts_query:
            fts_where = (
                f"{base_where} AND m.rowid IN "
                f"(SELECT rowid FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank)"
            )
            fts_params = base_params + [fts_query]
            fts_limit = k * 2
            rows = self._conn.execute(
                f"SELECT m.* FROM memory m {fts_where} LIMIT {fts_limit}",
                fts_params,
            ).fetchall()
            for r in rows:
                rid = r["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    records.append(self._row_to_record(r))

        if query_embedding is not None and self._vec_available:
            self._ensure_vec_table()
            try:
                vec_where = (
                    f"{base_where} AND m.embedding_id IN "
                    f"(SELECT embedding_id FROM memory_vec "
                    f"WHERE embedding MATCH ? ORDER BY distance LIMIT ?)"
                )
                vec_params = base_params + [json.dumps(query_embedding), k * 2]
                rows = self._conn.execute(
                    f"SELECT m.* FROM memory m {vec_where}",
                    vec_params,
                ).fetchall()
                for r in rows:
                    rid = r["id"]
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        records.append(self._row_to_record(r))
            except sqlite3.OperationalError:
                pass

        if not query and query_embedding is None:
            rows = self._conn.execute(
                f"SELECT m.* FROM memory m {base_where} ORDER BY m.created_at DESC LIMIT ?",
                base_params + [k],
            ).fetchall()
            for r in rows:
                rid = r["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    records.append(self._row_to_record(r))

        if query or query_embedding is not None:
            records = records[:k]

        return records

    def stats(self) -> dict[str, Any]:
        tier_counts = {}
        rows = self._conn.execute(
            "SELECT tier, COUNT(*) as cnt FROM memory GROUP BY tier"
        ).fetchall()
        for tier, cnt in rows:
            tier_counts[tier] = cnt

        total = self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

        fts_count = 0
        try:
            fts_count = self._conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        except Exception:
            pass

        db_size = 0
        if os.path.isfile(self.db_path):
            db_size = os.path.getsize(self.db_path)

        return {
            "tier_counts": tier_counts,
            "total_rows": total,
            "db_size_bytes": db_size,
            "vec_available": self._vec_available,
            "fts_row_count": fts_count,
        }

    def write_turn(self, turn: TurnRecord) -> None:
        if not turn.turn_id:
            turn.turn_id = self._generate_id()
        data = turn.model_dump()
        columns = list(data.keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        values = [data[c] for c in columns]
        self._conn.execute(
            f"INSERT OR REPLACE INTO turn ({column_names}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def write_session(self, session: SessionRecord) -> None:
        if not session.id:
            session.id = self._generate_id()
        data = session.model_dump()
        columns = list(data.keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        values = [data[c] for c in columns]
        self._conn.execute(
            f"INSERT OR REPLACE INTO session ({column_names}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def get_by_id(self, memory_id: str) -> MemoryRecord | None:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_by_tier(
        self, tier: str, status: str = "active", limit: int = 100
    ) -> list[MemoryRecord]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE tier = ? AND status = ? AND redaction = 0 "
            "ORDER BY created_at DESC LIMIT ?",
            (tier, status, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]
