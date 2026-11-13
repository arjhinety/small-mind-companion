from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class VectorSearchUnavailableError(RuntimeError):
    """An embedding was written or queried, but sqlite-vec is not loaded into this store."""


def load_sqlite_vec(conn: sqlite3.Connection) -> str | None:
    """Load the sqlite-vec extension into ``conn``. Returns None on success, else the reason.

    Python's sqlite3 refuses ``load_extension`` ("not authorized") until
    ``enable_load_extension(True)`` is called on the connection; ``sqlite_vec.load`` does not do
    that itself. Extension loading is switched back off afterwards.
    """
    try:
        import sqlite_vec
    except ImportError as exc:
        return f"ImportError: {exc}"
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        return "this Python's sqlite3 module was built without extension loading"
    try:
        sqlite_vec.load(conn)
    except Exception as exc:  # noqa: BLE001 - the reason is reported to the caller
        return f"{type(exc).__name__}: {exc}"
    finally:
        conn.enable_load_extension(False)
    return None


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
    def __init__(
        self,
        db_path: str,
        embedding_dim: int = 384,
        allow_keyword_fallback: bool = False,
        legacy_table_order: bool = False,
    ) -> None:
        """Open (or create) a memory store.

        If sqlite-vec cannot be loaded the store warns, and any write or search that carries an
        embedding raises ``VectorSearchUnavailableError`` -- unless ``allow_keyword_fallback`` is
        set, in which case embeddings are ignored and search is keyword-only. Silent degradation
        to keyword search is how vector search could be off without anyone knowing; the mode in
        use is exposed as ``search_mode``.

        ``search`` returns keyword matches in FTS5 rank order and vector matches in distance order.
        ``legacy_table_order=True`` restores the behaviour Study 001 ran with (commit ``f0d23f8``
        and earlier): matches selected by rank but returned in table order, with the keyword row
        limit applied after that reordering (``reports/ERRATA.md`` E42). Use it only to reproduce
        Study 001.
        """
        self.db_path = db_path
        self.legacy_table_order = legacy_table_order
        self.embedding_dim = embedding_dim
        self.allow_keyword_fallback = allow_keyword_fallback
        parent = Path(db_path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self.vector_search_error = load_sqlite_vec(self._conn)
        self._vec_available = self.vector_search_error is None
        if not self._vec_available:
            warnings.warn(
                f"MemoryStore({db_path!r}): vector search unavailable "
                f"({self.vector_search_error}); "
                + (
                    "allow_keyword_fallback=True, so search is keyword-only"
                    if allow_keyword_fallback
                    else "writes and searches that carry an embedding will raise"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        self._init_schema()
        self._vec_table_created = False

    @property
    def search_mode(self) -> str:
        """``hybrid`` (keyword + vector) when sqlite-vec is loaded, else ``keyword-only``."""
        return "hybrid" if self._vec_available else "keyword-only"

    def _require_vectors(self, what: str) -> bool:
        """True when vectors can be used; raises unless keyword fallback was opted into."""
        if self._vec_available:
            return True
        if self.allow_keyword_fallback:
            return False
        raise VectorSearchUnavailableError(
            f"{what} carries an embedding, but vector search is unavailable in "
            f"{self.db_path!r} ({self.vector_search_error}). Pass allow_keyword_fallback=True "
            "to accept keyword-only search."
        )

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

        if record.embedding is not None and self._require_vectors("write()"):
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
            fts_limit = k * 2
            if self.legacy_table_order:
                fts_where = (
                    f"{base_where} AND m.rowid IN "
                    f"(SELECT rowid FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank)"
                )
                rows = self._conn.execute(
                    f"SELECT m.* FROM memory m {fts_where} LIMIT {fts_limit}",
                    base_params + [fts_query],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT m.* FROM memory_fts f JOIN memory m ON m.rowid = f.rowid "
                    f"{base_where} AND memory_fts MATCH ? ORDER BY f.rank LIMIT {fts_limit}",
                    base_params + [fts_query],
                ).fetchall()
            for r in rows:
                rid = r["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    records.append(self._row_to_record(r))

        if query_embedding is not None and self._require_vectors("search()"):
            self._ensure_vec_table()
            # No try/except: a failing vector query (wrong dimension, corrupt table) used to be
            # swallowed here, which silently turned hybrid search into keyword search.
            if self.legacy_table_order:
                vec_where = (
                    f"{base_where} AND m.embedding_id IN "
                    f"(SELECT embedding_id FROM memory_vec "
                    f"WHERE embedding MATCH ? ORDER BY distance LIMIT ?)"
                )
                rows = self._conn.execute(
                    f"SELECT m.* FROM memory m {vec_where}",
                    base_params + [json.dumps(query_embedding), k * 2],
                ).fetchall()
            else:
                # sqlite-vec's KNN selects the k*2 nearest first; the filters then apply to those,
                # as in the legacy subquery, so both paths draw from the same candidate set.
                rows = self._conn.execute(
                    f"SELECT m.* FROM memory_vec v JOIN memory m "
                    f"ON m.embedding_id = v.embedding_id {base_where} "
                    f"AND v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                    base_params + [json.dumps(query_embedding), k * 2],
                ).fetchall()
            for r in rows:
                rid = r["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    records.append(self._row_to_record(r))

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
