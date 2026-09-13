import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from onebee.data.personas import PersonaCorpus
from onebee.memory.extraction.extractor import ExtractionPipeline
from onebee.memory.extraction.openai_extractor import OpenAITeacherExtractor
from onebee.memory.extraction.scoring import compute_importance
from onebee.memory.store import MemoryRecord, MemoryStore, SessionRecord, TurnRecord

personas_dir = Path("data/benchmarks/pmb_v0_full/personas")
stores_dir = Path("data/stores/pmb_v0_full")
stores_dir.mkdir(parents=True, exist_ok=True)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
extractor = OpenAITeacherExtractor(model="gpt-5.6-luna")
pipeline = ExtractionPipeline(extractor=extractor)

persona_files = sorted(personas_dir.glob("*.json"))
print(f"Found {len(persona_files)} personas", file=sys.stderr)

total_claims = 0
total_accepted = 0
total_turns_processed = 0

for pf in persona_files:
    corpus = PersonaCorpus(**json.loads(pf.read_text()))
    persona = corpus.persona
    db_path = stores_dir / f"{persona.persona_id}.db"
    if db_path.exists():
        db_path.unlink()
    store = MemoryStore(str(db_path))

    for session in corpus.sessions:
        store.write_session(
            SessionRecord(
                id=session.session_id,
                started=session.turns[0].ts if session.turns else 0,
            )
        )

        for turn in session.turns:
            store.write_turn(
                TurnRecord(
                    turn_id=turn.turn_id,
                    session_id=session.session_id,
                    role=turn.role,
                    text=turn.text,
                    ts=turn.ts,
                )
            )
            if turn.role != "user":
                continue
            total_turns_processed += 1
            try:
                results = pipeline.process_turn(turn.text, context={"persona_name": persona.name})
            except Exception as e:
                print(f"  extraction error on {turn.turn_id}: {e}", file=sys.stderr)
                continue
            for r in results:
                total_claims += 1
                if r.rejected:
                    continue
                total_accepted += 1
                claim = r.claim
                emb = embedder.encode([claim.content])[0].tolist()
                importance = compute_importance(
                    affect_arousal=0.5,
                    novelty=0.5,
                    entity_salience=0.5 if claim.entities else 0.3,
                    user_emphasis=0.5,
                    consequence=0.5,
                )
                record = MemoryRecord(
                    tier=claim.tier,
                    content=claim.content,
                    content_struct={
                        "subject": claim.subject,
                        "predicate": claim.predicate,
                        "object": claim.object,
                    },
                    created_at=turn.ts,
                    event_time=turn.ts,
                    importance=importance,
                    confidence=r.confidence,
                    decay_rate=0.05,
                    provenance={
                        "source": "openai_extraction",
                        "session_id": session.session_id,
                        "turn_ids": [turn.turn_id],
                        "extractor": "OpenAITeacherExtractor",
                        "verbatim_span": claim.verbatim_span,
                    },
                    entities=claim.entities,
                    topics=claim.topics,
                    embedding=emb,
                )
                store.write(record)
        if total_turns_processed % 20 == 0:
            print(
                f"  ...{total_turns_processed} user turns processed, "
                f"{total_accepted}/{total_claims} claims accepted",
                file=sys.stderr,
            )

    stats = store.stats()
    print(f"{persona.persona_id} ({persona.name}): {stats}", file=sys.stderr)

print(
    f"DONE. total user turns={total_turns_processed} total_claims={total_claims} "
    f"accepted={total_accepted}"
)
