import json
import random
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from onebee.context.builder import ContextBuilder
from onebee.data.personas import PersonaCorpus
from onebee.memory.store import MemoryStore
from onebee.retrieval.router import HybridRetriever

personas_dir = Path("data/benchmarks/sft_personas_v1/personas")
stores_dir = Path("data/stores/sft_personas_v1")
out_dir = Path("data/dpo/v1_scale")
out_dir.mkdir(parents=True, exist_ok=True)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
builder = ContextBuilder()

COMPANION_PERSONA = {
    "name": "the companion",
    "description": (
        "roleplaying as a warm AI companion in an ongoing relationship with the user. "
        "You have a real memory of things the user has told you before — the memories "
        "listed above ARE things you remember about THEM, not facts about yourself. "
        "When the user asks a question addressed to you, they are usually asking you "
        "to recall something about THEM using your memory of them, not asking about "
        "yourself as an AI. Answer directly and naturally as their companion who "
        "already knows this, using the memories above when relevant. Do not say you "
        "are an AI without memory or access to personal information — you DO have the "
        "memories listed above, use them. If none of the memories actually answer the "
        "question, say you don't know rather than guessing."
    ),
    "traits": [],
}

# Rejected responses: exactly the failure mode observed on Day 3 (raw model, no SFT/DPO) —
# generic, persona-breaking, self-referential disclaimers, regardless of what memory was
# actually available. This directly targets what DPO should teach the model NOT to do.
GENERIC_DISCLAIMERS = [
    "I am a large language model, trained by Google.",
    "As an AI, I don't have access to personal information about you.",
    "I'm just a computer program and don't have memories of past conversations.",
    "I am Gemma, a large language model developed by Google DeepMind.",
    "As a language model, I don't retain information between conversations.",
]


def get_teacher_response(system_text: str, user_text: str) -> str:
    import openai

    client = openai.OpenAI()
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    completion = client.chat.completions.create(model="gpt-5.6-luna", messages=messages)
    return (completion.choices[0].message.content or "").strip()


corpora = []
for pf in sorted(personas_dir.glob("*.json")):
    corpora.append(PersonaCorpus(**json.loads(pf.read_text())))

retrievers = {}
for db_path in stores_dir.glob("*.db"):
    store = MemoryStore(str(db_path))
    retrievers[db_path.stem] = HybridRetriever(store)

print(f"Loaded {len(corpora)} personas, {len(retrievers)} stores", file=sys.stderr)

rng = random.Random(5150)
K = 8
pairs = []

for corpus in corpora:
    persona = corpus.persona
    retriever = retrievers.get(persona.persona_id)
    if retriever is None:
        continue
    for session in corpus.sessions:
        recent_turns: list[dict] = []
        for turn in session.turns:
            if turn.role != "user":
                recent_turns.append({"role": turn.role, "text": turn.text})
                continue

            query_emb = embedder.encode([turn.text])[0].tolist()
            candidates = retriever.retrieve(turn.text, query_embedding=query_emb, k=K)
            retrieved_records = [c.record for c in candidates]

            # Only build preference pairs where memory was actually retrieved — the
            # scenario where a persona-consistent vs. persona-breaking response is a
            # meaningful distinction (an empty-context turn has no "right" answer to
            # prefer over a disclaimer).
            if not retrieved_records:
                recent_turns.append({"role": turn.role, "text": turn.text})
                continue

            system_text, _ = builder.build(
                turn_id=turn.turn_id,
                persona=COMPANION_PERSONA,
                profile={},
                boundaries=[],
                retrieved_memories=retrieved_records,
                recent_turns=recent_turns[-6:],
                user_turn="",
            )
            system_text = system_text.strip()

            try:
                chosen = get_teacher_response(system_text, turn.text)
            except Exception as e:
                print(f"  teacher error on {turn.turn_id}: {e}", file=sys.stderr)
                recent_turns.append({"role": turn.role, "text": turn.text})
                continue

            rejected = rng.choice(GENERIC_DISCLAIMERS)

            if chosen and chosen != rejected:
                full_prompt = f"{system_text}\n\nUser: {turn.text}"
                pairs.append(
                    {
                        "prompt": full_prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

            recent_turns.append({"role": turn.role, "text": turn.text})

    print(f"{persona.persona_id}: {len(pairs)} pairs so far", file=sys.stderr)

print(f"Generated {len(pairs)} preference pairs", file=sys.stderr)

rng.shuffle(pairs)
split_idx = int(len(pairs) * 0.9)
train = pairs[:split_idx]
val = pairs[split_idx:]

with open(out_dir / "train.jsonl", "w") as f:
    for p in train:
        f.write(json.dumps(p) + "\n")
with open(out_dir / "val.jsonl", "w") as f:
    for p in val:
        f.write(json.dumps(p) + "\n")

datasheet = f"""# DPO v1_scale preference dataset (proper scale)

Persona-contrastive preference pairs: chosen = real teacher (gpt-5.6-luna) response using
injected memory context (same pipeline as data/sft/v1), rejected = a generic persona-breaking
AI-assistant disclaimer, matching the exact failure mode observed in Day 3's raw-model
evaluation ("I am a large language model, trained by Google."). Built from the same 40 SFT
personas as data/sft/v1/ (disjoint from the PMB-v0 eval set).

- Total pairs: {len(pairs)} ({len(train)} train / {len(val)} val)
- Only turns where memory was actually retrieved are included (a meaningful preference signal
  requires there to be a "right" character-consistent answer to prefer).
- rejected responses are a fixed pool of 5 hand-written disclaimer sentences, not
  teacher-generated — a deliberate, cheap way to encode "always prefer being in-character over
  a generic disclaimer" as the DPO signal, not a nuanced quality preference.
- Not human-reviewed.
"""
(out_dir / "DATASHEET.md").write_text(datasheet)

import hashlib

h = hashlib.sha256()
for p in sorted(out_dir.glob("*.jsonl")):
    h.update(p.read_bytes())
(out_dir / "hash.txt").write_text(h.hexdigest() + "\n")

print(f"DONE. train={len(train)} val={len(val)}")
