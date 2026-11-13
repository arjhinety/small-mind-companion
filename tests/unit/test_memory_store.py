from __future__ import annotations

import sqlite3
import time

import pytest

from onebee.memory import store as store_module
from onebee.memory.store import (
    MemoryRecord,
    MemoryStore,
    SessionRecord,
    TurnRecord,
    VectorSearchUnavailableError,
    load_sqlite_vec,
)


def _sqlite_vec_loads() -> bool:
    """Whether sqlite-vec loads into a connection, using the store's own loader.

    Probing by loading (not by `find_spec`) is the honest capability check; using
    `load_sqlite_vec` keeps the probe and the store from disagreeing about how to load it.
    """
    conn = sqlite3.connect(":memory:")
    try:
        return load_sqlite_vec(conn) is None
    finally:
        conn.close()


_HAS_SQLITE_VEC = _sqlite_vec_loads()


def _make_record(
    id: str = "",
    tier: str = "episodic",
    content: str = "test content",
    **kwargs,
) -> MemoryRecord:
    defaults: dict = {
        "id": id,
        "tier": tier,
        "content": content,
        "created_at": int(time.time() * 1000),
        "importance": 0.5,
        "confidence": 0.9,
        "decay_rate": 0.01,
        "provenance": {
            "source": "test",
            "session_id": "s1",
            "turn_ids": [],
            "extractor": "test_extractor",
            "extractor_version": "0.1",
            "verbatim_span": None,
        },
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


class TestSchema:
    def test_schema_creates_cleanly(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path)
        stats = store.stats()
        assert "tier_counts" in stats
        assert stats["total_rows"] == 0

    def test_pragma_user_version(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path)
        conn = store._conn
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 1


class TestWriteAndGetById:
    def test_write_and_get_short_term(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="short_term", content="hello short term")
        rid = store.write(rec)
        assert rid == rec.id

        fetched = store.get_by_id(rid)
        assert fetched is not None
        assert fetched.tier == "short_term"
        assert fetched.content == "hello short term"

    def test_write_and_get_episodic(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic", content="an episode")
        rid = store.write(rec)
        fetched = store.get_by_id(rid)
        assert fetched is not None
        assert fetched.tier == "episodic"
        assert fetched.content == "an episode"

    def test_write_and_get_semantic(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="semantic", content="semantic fact")
        rid = store.write(rec)
        fetched = store.get_by_id(rid)
        assert fetched is not None
        assert fetched.tier == "semantic"
        assert fetched.content == "semantic fact"

    def test_write_with_json_fields(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(
            tier="episodic",
            content="json test",
            content_struct={"key": "value", "nested": [1, 2]},
            entities=["entity1", "entity2"],
            topics=["topicA"],
            provenance={"source": "unit-test"},
        )
        rid = store.write(rec)
        fetched = store.get_by_id(rid)
        assert fetched is not None
        assert fetched.content_struct == {"key": "value", "nested": [1, 2]}
        assert fetched.entities == ["entity1", "entity2"]
        assert fetched.topics == ["topicA"]
        assert fetched.provenance == {"source": "unit-test"}

    def test_write_upserts_by_id(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(id="fixed-id", content="version 1")
        store.write(rec)

        rec2 = _make_record(id="fixed-id", content="version 2")
        store.write(rec2)

        fetched = store.get_by_id("fixed-id")
        assert fetched is not None
        assert fetched.content == "version 2"

    def test_write_generates_id(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(id="", content="auto id")
        rid = store.write(rec)
        assert rid
        assert len(rid) == 36

    def test_write_with_embedding(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"), embedding_dim=4)
        rec = _make_record(
            content="vectorized",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        rid = store.write(rec)
        fetched = store.get_by_id(rid)
        assert fetched is not None
        if _HAS_SQLITE_VEC:
            assert fetched.embedding_id is not None
        assert fetched.embedding is None

    def test_get_by_id_nonexistent(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        assert store.get_by_id("nonexistent") is None


class TestWriteTurnSession:
    def test_write_turn_roundtrip(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.write_session(SessionRecord(id="s1"))
        turn = TurnRecord(
            turn_id="t1",
            session_id="s1",
            role="user",
            text="hello",
            ts=1000,
        )
        store.write_turn(turn)

        conn = store._conn
        conn.row_factory = __import__("sqlite3").Row
        row = conn.execute("SELECT * FROM turn WHERE turn_id = ?", ("t1",)).fetchone()
        assert row is not None
        assert row["role"] == "user"
        assert row["text"] == "hello"

    def test_write_session_roundtrip(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        session = SessionRecord(
            id="s1",
            started=1000,
            ended=2000,
            summary="test session",
            turn_count=5,
        )
        store.write_session(session)

        conn = store._conn
        conn.row_factory = __import__("sqlite3").Row
        row = conn.execute("SELECT * FROM session WHERE id = ?", ("s1",)).fetchone()
        assert row is not None
        assert row["summary"] == "test session"
        assert row["turn_count"] == 5

    def test_write_turn_auto_id(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.write_session(SessionRecord(id="s1"))
        turn = TurnRecord(session_id="s1", role="assistant", text="reply", ts=2000)
        store.write_turn(turn)
        assert turn.turn_id
        assert len(turn.turn_id) == 36

    def test_write_session_auto_id(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        session = SessionRecord(summary="auto id session")
        store.write_session(session)
        assert session.id
        assert len(session.id) == 36


class TestSearch:
    def test_fts_search_finds_record(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="the quick brown fox")
        store.write(rec)
        rec2 = _make_record(content="lazy dog sleeps")
        store.write(rec2)

        results = store.search(query="quick fox", k=5)
        ids = {r.id for r in results}
        assert rec.id in ids

    def test_fts_search_returns_empty_for_no_match(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="hello world")
        store.write(rec)

        results = store.search(query="zzzznonexistent", k=5)
        assert len(results) == 0

    def test_fts_search_handles_punctuation_in_query(self, tmp_path):
        # Regression test: a real System D run crashed with
        # `sqlite3.OperationalError: fts5: syntax error near "?"` because natural
        # questions ("What's my favorite color?") contain characters FTS5's MATCH
        # parses as query syntax (?, ', -, :), not literal text.
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="my favorite color is blue")
        store.write(rec)

        for query in [
            "What's my favorite color?",
            "How-old are you: really?",
            'quote "marks" and (parens)',
            "AND OR NOT",  # FTS5 boolean operator keywords as literal query text
        ]:
            results = store.search(query=query, k=5)  # must not raise
            assert isinstance(results, list)

        results = store.search(query="What's my favorite color?", k=5)
        ids = {r.id for r in results}
        assert rec.id in ids

    def test_fts_search_empty_query_after_sanitization(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="hello world")
        store.write(rec)
        # Punctuation-only query has no word tokens after sanitization — must not
        # raise, and (with no query_embedding either) falls through to no results
        # from the FTS branch specifically (query is still truthy so the
        # recency-fallback branch does not apply either).
        results = store.search(query="???", k=5)
        assert results == []

    def test_search_rejects_unknown_filter_column(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        with pytest.raises(ValueError, match="unknown or non-filterable column"):
            store.search(query="x", filters={"content": "'; DROP TABLE memory; --"})

    def test_search_excludes_redacted(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="visible content")
        store.write(rec)
        rec2 = _make_record(content="redacted content", redaction=1)
        store.write(rec2)

        results = store.search(query="content", k=10)
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_search_excludes_inactive_by_default(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec_active = _make_record(content="active record", status="active")
        store.write(rec_active)
        rec_superseded = _make_record(content="superseded record", status="superseded")
        store.write(rec_superseded)

        results = store.search(query="record", k=10)
        ids = {r.id for r in results}
        assert rec_active.id in ids
        assert rec_superseded.id not in ids

    def test_search_includes_status_when_filtered(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec_superseded = _make_record(content="old news", status="superseded")
        store.write(rec_superseded)

        results = store.search(query="news", filters={"status": "superseded"}, k=10)
        ids = {r.id for r in results}
        assert rec_superseded.id in ids

    def test_search_by_tier(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec_st = _make_record(tier="short_term", content="st content")
        store.write(rec_st)
        rec_ep = _make_record(tier="episodic", content="ep content")
        store.write(rec_ep)

        results = store.search(query="content", tier="short_term", k=10)
        ids = {r.id for r in results}
        assert rec_st.id in ids
        assert rec_ep.id not in ids

    def test_vector_query_of_the_wrong_dimension_raises(self, tmp_path):
        """A failing vector query must surface; it used to be swallowed into keyword search."""
        if not _HAS_SQLITE_VEC:
            pytest.skip("sqlite-vec not available")
        store = MemoryStore(str(tmp_path / "test.db"))  # 384-dimension table
        store.write(_make_record(content="vector candidate"))
        with pytest.raises(sqlite3.OperationalError):
            store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], k=5)

    def test_vector_search_finds_record_when_vec_available(self, tmp_path):
        if not _HAS_SQLITE_VEC:
            pytest.skip("sqlite-vec not available")

        store = MemoryStore(str(tmp_path / "test.db"), embedding_dim=4)
        embedding = [0.1, 0.2, 0.3, 0.4]
        rec = _make_record(content="nearby", embedding=embedding)
        store.write(rec)

        rec2 = _make_record(content="far away", embedding=[0.9, 0.8, 0.7, 0.6])
        store.write(rec2)

        results = store.search(query_embedding=embedding, k=5)
        ids = {r.id for r in results}
        assert rec.id in ids

    def test_combined_fts_and_vector_search(self, tmp_path):
        if not _HAS_SQLITE_VEC:
            pytest.skip("sqlite-vec not available")

        store = MemoryStore(str(tmp_path / "test.db"), embedding_dim=4)
        rec_a = _make_record(content="alpha bravo", embedding=[0.1, 0.2, 0.3, 0.4])
        store.write(rec_a)
        rec_b = _make_record(content="delta echo", embedding=[0.15, 0.25, 0.35, 0.45])
        store.write(rec_b)

        results = store.search(query="alpha", query_embedding=[0.1, 0.2, 0.3, 0.4], k=5)
        ids = {r.id for r in results}
        assert len(ids) >= 1


class TestListByTier:
    def test_list_by_tier_filters_correctly(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec_st = _make_record(tier="short_term", content="st")
        store.write(rec_st)
        rec_ep = _make_record(tier="episodic", content="ep")
        store.write(rec_ep)

        results = store.list_by_tier("short_term")
        ids = {r.id for r in results}
        assert rec_st.id in ids
        assert rec_ep.id not in ids

    def test_list_by_tier_excludes_redacted(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic", content="visible", redaction=0)
        store.write(rec)
        rec2 = _make_record(tier="episodic", content="hidden", redaction=1)
        store.write(rec2)

        results = store.list_by_tier("episodic")
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_list_by_tier_excludes_inactive_by_default(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic", content="active", status="active")
        store.write(rec)
        rec2 = _make_record(tier="episodic", content="superseded", status="superseded")
        store.write(rec2)

        results = store.list_by_tier("episodic")
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_list_by_tier_status_override(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic", content="superseded", status="superseded")
        store.write(rec)

        results = store.list_by_tier("episodic", status="superseded")
        ids = {r.id for r in results}
        assert rec.id in ids

    def test_list_by_tier_limit(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        for i in range(10):
            store.write(_make_record(tier="episodic", content=f"item {i}"))

        results = store.list_by_tier("episodic", limit=3)
        assert len(results) == 3


class TestStats:
    def test_stats_shape(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        stats = store.stats()
        assert "tier_counts" in stats
        assert "total_rows" in stats
        assert "db_size_bytes" in stats
        assert "vec_available" in stats
        assert "fts_row_count" in stats
        assert stats["total_rows"] == 0
        assert isinstance(stats["vec_available"], bool)

    def test_stats_counts_reflect_writes(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.write(_make_record(tier="short_term"))
        store.write(_make_record(tier="episodic"))
        store.write(_make_record(tier="episodic"))
        store.write(_make_record(tier="semantic"))

        stats = store.stats()
        assert stats["total_rows"] == 4
        assert stats["tier_counts"]["short_term"] == 1
        assert stats["tier_counts"]["episodic"] == 2
        assert stats["tier_counts"]["semantic"] == 1

    def test_stats_db_size_increases(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        stats1 = store.stats()
        for _ in range(5):
            store.write(_make_record())
        stats2 = store.stats()
        assert stats2["db_size_bytes"] >= stats1["db_size_bytes"]


class TestVecUnavailable:
    """When sqlite-vec cannot load, the store says so and refuses embeddings unless told not to."""

    @pytest.fixture
    def no_vec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(store_module, "load_sqlite_vec", lambda conn: "simulated failure")

    def test_vector_search_loads_when_installed(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        store = MemoryStore(str(tmp_path / "test.db"))
        assert store.search_mode == "hybrid"
        assert store.vector_search_error is None

    def test_store_warns_and_reports_keyword_only(self, tmp_path, no_vec):
        with pytest.warns(RuntimeWarning, match="simulated failure"):
            store = MemoryStore(str(tmp_path / "test.db"))
        assert store.search_mode == "keyword-only"
        rec = _make_record(content="works without vec")
        store.write(rec)
        assert store.get_by_id(rec.id) is not None
        stats = store.stats()
        assert stats["vec_available"] is False
        assert stats["total_rows"] == 1

    def test_embedding_without_vector_search_raises(self, tmp_path, no_vec):
        with pytest.warns(RuntimeWarning):
            store = MemoryStore(str(tmp_path / "test.db"), embedding_dim=4)
        with pytest.raises(VectorSearchUnavailableError, match="write"):
            store.write(_make_record(content="x", embedding=[0.1, 0.2, 0.3, 0.4]))
        with pytest.raises(VectorSearchUnavailableError, match="search"):
            store.search(query_embedding=[0.1, 0.2, 0.3, 0.4])

    def test_keyword_fallback_is_opt_in(self, tmp_path, no_vec):
        with pytest.warns(RuntimeWarning, match="keyword-only"):
            store = MemoryStore(
                str(tmp_path / "test.db"), embedding_dim=4, allow_keyword_fallback=True
            )
        rec = _make_record(content="alpha bravo", embedding=[0.1, 0.2, 0.3, 0.4])
        store.write(rec)
        results = store.search(query="alpha", query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert [r.id for r in results] == [rec.id]


class TestRedaction:
    def test_search_excludes_redacted_rows(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(content="good and bad are both here")
        store.write(rec)
        rec2 = _make_record(content="also good and bad", redaction=1)
        store.write(rec2)

        results = store.search(query="good bad", k=10)
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_list_by_tier_excludes_redacted_rows(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic")
        store.write(rec)
        rec2 = _make_record(tier="episodic", redaction=1)
        store.write(rec2)

        results = store.list_by_tier("episodic")
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids


class TestSupersededStatus:
    def test_search_excludes_superseded_by_default(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(status="active", content="apple banana")
        store.write(rec)
        rec2 = _make_record(status="superseded", content="apple banana")
        store.write(rec2)

        results = store.search(query="apple banana", k=10)
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_search_includes_superseded_when_explicit(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec2 = _make_record(status="superseded", content="cherry pie")
        store.write(rec2)

        results = store.search(query="cherry", filters={"status": "superseded"}, k=10)
        ids = {r.id for r in results}
        assert rec2.id in ids

    def test_list_by_tier_excludes_superseded_by_default(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec = _make_record(tier="episodic", status="active", content="a")
        store.write(rec)
        rec2 = _make_record(tier="episodic", status="superseded", content="b")
        store.write(rec2)

        results = store.list_by_tier("episodic")
        ids = {r.id for r in results}
        assert rec.id in ids
        assert rec2.id not in ids

    def test_list_by_tier_includes_superseded_when_explicit(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        rec2 = _make_record(tier="episodic", status="superseded", content="b")
        store.write(rec2)

        results = store.list_by_tier("episodic", status="superseded")
        ids = {r.id for r in results}
        assert rec2.id in ids


class TestProtocolConformance:
    def test_store_implements_protocol(self):

        store = MemoryStore(":memory:")
        assert hasattr(store, "write")
        assert hasattr(store, "search")
        assert hasattr(store, "stats")
        assert hasattr(store, "write_turn")
        assert hasattr(store, "write_session")
        assert hasattr(store, "get_by_id")
        assert hasattr(store, "list_by_tier")


class TestSearchOrder:
    """Results come back in rank order, not table order (ERRATA E42). Every case inserts the best
    match last, so table order and rank order disagree; ``legacy_table_order=True`` must keep the
    table-order behaviour Study 001 ran with, including the final cut to k."""

    def _vector_store(self, tmp_path, legacy: bool) -> tuple[MemoryStore, list[str]]:
        store = MemoryStore(str(tmp_path / "v.db"), embedding_dim=4, legacy_table_order=legacy)
        ids = []
        for offset in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1):  # nearest to [1, 0, 0, 0] inserted last
            ids.append(
                store.write(_make_record(content=f"v {offset}", embedding=[1, offset, 0, 0]))
            )
        return store, ids

    def test_vector_results_nearest_first(self, tmp_path):
        if not _HAS_SQLITE_VEC:
            pytest.skip("sqlite-vec not available")
        store, ids = self._vector_store(tmp_path, legacy=False)
        results = store.search(query_embedding=[1, 0, 0, 0], k=2)
        assert [r.id for r in results] == [ids[5], ids[4]]

    def test_vector_legacy_keeps_table_order_and_cut(self, tmp_path):
        if not _HAS_SQLITE_VEC:
            pytest.skip("sqlite-vec not available")
        store, ids = self._vector_store(tmp_path, legacy=True)
        results = store.search(query_embedding=[1, 0, 0, 0], k=2)
        # the 4 (= 2k) nearest are ids[2:6]; table order, then cut to k = 2
        assert [r.id for r in results] == [ids[2], ids[3]]

    def _keyword_store(self, tmp_path, legacy: bool) -> tuple[MemoryStore, list[str]]:
        store = MemoryStore(
            str(tmp_path / "k.db"), allow_keyword_fallback=True, legacy_table_order=legacy
        )
        ids = [
            store.write(_make_record(content=f"kiwi with a long tail of filler words number {i}"))
            for i in range(5)
        ]
        ids.append(store.write(_make_record(content="kiwi kiwi kiwi kiwi")))  # best, inserted last
        return store, ids

    def test_keyword_best_match_first(self, tmp_path):
        store, ids = self._keyword_store(tmp_path, legacy=False)
        results = store.search(query="kiwi", k=2)
        assert len(results) == 2
        assert results[0].id == ids[5]

    def test_keyword_legacy_drops_best_match_beyond_limit(self, tmp_path):
        store, ids = self._keyword_store(tmp_path, legacy=True)
        results = store.search(query="kiwi", k=2)
        # the outer LIMIT 2k = 4 keeps the first 4 matches in table order, then the cut to k = 2
        assert [r.id for r in results] == [ids[0], ids[1]]
