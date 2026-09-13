from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from onebee.data.personas import (
    ConversationTurn,
    FactSheetEntry,
    Persona,
    PersonaCorpus,
    PersonaSession,
)
from onebee.data.teacher import FixtureTeacherClient
from onebee.evaluation.metrics.personalized import Probe


class TestFactSheetEntry:
    def test_valid_entry(self) -> None:
        entry = FactSheetEntry(
            fact_id="p000_f001",
            subject="p000",
            predicate="lives in",
            object="New York",
            category="factual",
        )
        assert entry.fact_id == "p000_f001"
        assert entry.category == "factual"

    def test_invalid_category(self) -> None:
        with pytest.raises(ValueError):
            FactSheetEntry(
                fact_id="x",
                subject="x",
                predicate="x",
                object="x",
                category="invalid",
            )


class TestPersona:
    def test_valid_persona(self) -> None:
        facts = [
            FactSheetEntry(
                fact_id="p000_f000",
                subject="p000",
                predicate="lives in",
                object="NYC",
                category="factual",
            )
        ]
        persona = Persona(
            persona_id="p000",
            name="Alice",
            description="Test",
            traits=["friendly"],
            fact_sheet=facts,
        )
        assert persona.persona_id == "p000"
        assert len(persona.fact_sheet) == 1


class TestConversationTurn:
    def test_valid_turn(self) -> None:
        turn = ConversationTurn(
            turn_id="p000_s000_t000",
            session_id="p000_s000",
            role="user",
            text="Hello",
            ts=0,
            revealed_fact_ids=["p000_f000"],
        )
        assert turn.turn_id == "p000_s000_t000"
        assert turn.role == "user"
        assert len(turn.revealed_fact_ids) == 1

    def test_default_revealed_fact_ids(self) -> None:
        turn = ConversationTurn(
            turn_id="t",
            session_id="s",
            role="assistant",
            text="Hi",
            ts=0,
        )
        assert turn.revealed_fact_ids == []

    def test_invalid_role(self) -> None:
        with pytest.raises(ValueError):
            ConversationTurn(
                turn_id="t",
                session_id="s",
                role="system",
                text="x",
                ts=0,
            )


class TestPersonaSession:
    def test_valid_session(self) -> None:
        session = PersonaSession(
            session_id="p000_s000",
            persona_id="p000",
            turns=[],
        )
        assert session.session_id == "p000_s000"


class TestPersonaCorpus:
    def test_valid_corpus(self) -> None:
        persona = Persona(
            persona_id="p000",
            name="Alice",
            description="Test",
            traits=["friendly"],
            fact_sheet=[],
        )
        corpus = PersonaCorpus(persona=persona, sessions=[])
        assert corpus.persona.persona_id == "p000"


class TestFixtureTeacherClient:
    def test_determinism(self) -> None:
        persona = Persona(
            persona_id="p000",
            name="Alice",
            description="Test persona",
            traits=["friendly", "organized"],
            fact_sheet=[
                FactSheetEntry(
                    fact_id="p000_f000",
                    subject="p000",
                    predicate="lives in",
                    object="NYC",
                    category="factual",
                ),
                FactSheetEntry(
                    fact_id="p000_f001",
                    subject="p000",
                    predicate="favorite food",
                    object="pizza",
                    category="preference",
                ),
            ],
        )

        teacher1 = FixtureTeacherClient(seed=42)
        teacher2 = FixtureTeacherClient(seed=42)

        turns1 = teacher1.generate_conversation(persona, 0, 4, persona.fact_sheet)
        turns2 = teacher2.generate_conversation(persona, 0, 4, persona.fact_sheet)

        assert len(turns1) == len(turns2)
        for t1, t2 in zip(turns1, turns2):
            assert t1.turn_id == t2.turn_id
            assert t1.text == t2.text
            assert t1.revealed_fact_ids == t2.revealed_fact_ids

        probe1 = teacher1.generate_probe(persona, persona.fact_sheet[0], "factual")
        probe2 = teacher2.generate_probe(persona, persona.fact_sheet[0], "factual")
        assert probe1 == probe2

    def test_generate_conversation_produces_turns(self) -> None:
        persona = Persona(
            persona_id="p000",
            name="Alice",
            description="Test",
            traits=["friendly"],
            fact_sheet=[
                FactSheetEntry(
                    fact_id="p000_f000",
                    subject="p000",
                    predicate="lives in",
                    object="NYC",
                    category="factual",
                ),
            ],
        )
        teacher = FixtureTeacherClient(seed=1)
        turns = teacher.generate_conversation(persona, 0, 6, persona.fact_sheet)
        assert len(turns) > 0
        roles = [t.role for t in turns]
        assert "user" in roles
        assert "assistant" in roles

    def test_generate_probe_per_category(self) -> None:
        persona = Persona(
            persona_id="p000",
            name="Alice",
            description="Test",
            traits=["friendly"],
            fact_sheet=[
                FactSheetEntry(
                    fact_id="p000_f000",
                    subject="p000",
                    predicate="lives in",
                    object="NYC",
                    category="factual",
                ),
            ],
        )
        fact = persona.fact_sheet[0]
        teacher = FixtureTeacherClient(seed=1)

        for cat in (
            "factual",
            "preference",
            "episodic",
            "temporal",
            "unanswerable",
            "outdated_fact",
            "distractor",
            "continuity",
        ):
            question = teacher.generate_probe(persona, fact, cat)
            assert isinstance(question, str)
            assert len(question) > 0


class TestBuildPmbMain:
    def test_end_to_end_tmp_path(self) -> None:
        from scripts import build_pmb  # type: ignore[import-not-found]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pmb_test"

            old_argv = sys.argv
            sys.argv = [
                "build_pmb.py",
                "--n-personas",
                "1",
                "--sessions-per-persona",
                "1",
                "--turns-per-session",
                "4",
                "--facts-per-persona",
                "4",
                "--out-dir",
                str(out),
                "--seed",
                "42",
            ]
            try:
                build_pmb.main()
            finally:
                sys.argv = old_argv

            assert (out / "personas").is_dir()
            persona_files = list((out / "personas").glob("*.json"))
            assert len(persona_files) == 1

            probes_path = out / "probes.jsonl"
            assert probes_path.exists()
            probes = []
            with open(probes_path) as f:
                for line in f:
                    probes.append(Probe.model_validate_json(line))
            assert len(probes) > 0

            assert (out / "DATASHEET.md").exists()
            assert (out / "hash.txt").exists()

            with open(out / "hash.txt") as f:
                stored_hash = f.read().strip()

            # Independent recomputation of the corpus digest. Kept as its own implementation rather
            # than importing _compute_hash, so a change in the script's logic is caught here. Both
            # must normalise line endings to LF, otherwise the digest depends on the platform's
            # checkout convention rather than on the content (`core.autocrlf=true` on Windows).
            def _lf(text: str) -> bytes:
                return text.replace("\r\n", "\n").encode("utf-8")

            hasher = hashlib.sha256()
            for root, dirs, files in sorted(os.walk(out)):
                for fname in sorted(files):
                    if fname == "hash.txt":
                        continue
                    fpath = Path(root) / fname
                    rel = fpath.relative_to(out)
                    hasher.update(_lf(rel.as_posix()))
                    hasher.update(_lf(fpath.read_text(encoding="utf-8")))
            computed_hash = hasher.hexdigest()
            assert stored_hash == computed_hash

            persona_ids = {p.persona_id for p in probes}
            assert persona_ids.issubset({"p000"})

    def test_reproducibility(self) -> None:
        from scripts import build_pmb  # type: ignore[import-not-found]

        out1 = Path(tempfile.mkdtemp())
        out2 = Path(tempfile.mkdtemp())

        old_argv = sys.argv
        try:
            sys.argv = [
                "build_pmb.py",
                "--n-personas",
                "1",
                "--sessions-per-persona",
                "1",
                "--turns-per-session",
                "4",
                "--facts-per-persona",
                "4",
                "--out-dir",
                str(out1),
                "--seed",
                "42",
            ]
            build_pmb.main()

            sys.argv = [
                "build_pmb.py",
                "--n-personas",
                "1",
                "--sessions-per-persona",
                "1",
                "--turns-per-session",
                "4",
                "--facts-per-persona",
                "4",
                "--out-dir",
                str(out2),
                "--seed",
                "42",
            ]
            build_pmb.main()
        finally:
            sys.argv = old_argv

        with open(out1 / "hash.txt") as f:
            h1 = f.read().strip()
        with open(out2 / "hash.txt") as f:
            h2 = f.read().strip()
        assert h1 == h2
