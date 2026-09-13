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
out_dir = Path("data/sft/v1")
out_dir.mkdir(parents=True, exist_ok=True)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
builder = ContextBuilder()

COMPANION_PERSONA = {
    "name": "the companion",
    "description": (
        "roleplaying as a warm AI companion in an ongoing relationship with the user. "
        "You have a real memory of things the user has told you before — the memories "
        "listed above ARE things you remember about THEM, not facts about yourself. "
        "When the user asks a question addressed to 'you', they are usually asking you "
        "to recall something about THEM using your memory of them, not asking about "
        "yourself as an AI. Answer directly and naturally as their companion who "
        "already knows this, using the memories above when relevant. Do not say you "
        "are an AI without memory or access to personal information — you DO have the "
        "memories listed above, use them. If none of the memories actually answer the "
        "question, say you don't know rather than guessing."
    ),
    "traits": [],
}

corpora = []
for pf in sorted(personas_dir.glob("*.json")):
    corpora.append(PersonaCorpus(**json.loads(pf.read_text())))

retrievers = {}
for db_path in stores_dir.glob("*.db"):
    store = MemoryStore(str(db_path))
    retrievers[db_path.stem] = HybridRetriever(store)

print(f"Loaded {len(corpora)} SFT personas, {len(retrievers)} stores", file=sys.stderr)


def get_teacher_response(system_text: str, user_text: str) -> str:
    import openai

    client = openai.OpenAI()
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    completion = client.chat.completions.create(model="gpt-5.6-luna", messages=messages)
    return (completion.choices[0].message.content or "").strip()


examples = []
rng = random.Random(4242)
K = 8

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
                target = get_teacher_response(system_text, turn.text)
            except Exception as e:
                print(f"  teacher error on {turn.turn_id}: {e}", file=sys.stderr)
                recent_turns.append({"role": turn.role, "text": turn.text})
                continue

            if target:
                examples.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": turn.text},
                            {"role": "assistant", "content": target},
                        ],
                        "meta": {
                            "kind": "memory_relevant",
                            "persona_id": persona.persona_id,
                            "turn_id": turn.turn_id,
                        },
                    }
                )

            recent_turns.append({"role": turn.role, "text": turn.text})

    print(f"{persona.persona_id}: {len(examples)} examples so far", file=sys.stderr)

print(f"Generated {len(examples)} memory-relevant examples", file=sys.stderr)

# ~6% irrelevant-retrieval examples: pair a turn's question with a WRONG persona's memories.
# Ratio lowered from 15% and target diversified from 1 fixed string to several paraphrases
# (picked per-example) after the properly-scaled run showed the model over-generalizing this
# templated hedge to a large fraction of genuinely answerable questions once the dedup bug
# (docs/model_quirks.md #16) stopped silently discarding almost all of these examples --
# both ratio and phrasing diversity should reduce how easy a low-effort shortcut this is to
# learn. See docs/proper_scale_results.md's over-abstention tradeoff section.
IRRELEVANT_RESPONSES = [
    "I don't have anything about that in what I remember about you — could you tell me more?",
    "That doesn't ring a bell from what you've shared with me — mind filling me in?",
    "I'm not seeing anything about that in my memory of our conversations — what's the story?",
    "Hmm, I don't think you've mentioned that to me before — tell me more?",
]
persona_ids = list(retrievers.keys())
n_irrelevant = max(1, int(len(examples) * 0.06))
irrelevant_examples = []
base_pool = [e for e in examples if e["meta"]["kind"] == "memory_relevant"]
for _ in range(min(n_irrelevant, len(base_pool))):
    src = rng.choice(base_pool)
    wrong_pid = rng.choice([p for p in persona_ids if p != src["meta"]["persona_id"]])
    wrong_retriever = retrievers[wrong_pid]
    user_text = src["messages"][1]["content"]
    query_emb = embedder.encode([user_text])[0].tolist()
    candidates = wrong_retriever.retrieve(user_text, query_embedding=query_emb, k=K)
    wrong_records = [c.record for c in candidates]
    system_text, _ = builder.build(
        turn_id=f"irrelevant_{src['meta']['turn_id']}",
        persona=COMPANION_PERSONA,
        profile={},
        boundaries=[],
        retrieved_memories=wrong_records,
        recent_turns=[],
        user_turn="",
    )
    target = rng.choice(IRRELEVANT_RESPONSES)
    irrelevant_examples.append(
        {
            "messages": [
                {"role": "system", "content": system_text.strip()},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": target},
            ],
            "meta": {"kind": "irrelevant_retrieval"},
        }
    )

print(f"Generated {len(irrelevant_examples)} irrelevant-retrieval examples", file=sys.stderr)

# ~5% abstention examples: ask about a predicate never revealed for that persona, k memories
# retrieved anyway. Ratio lowered from 10%, target diversified -- same reasoning as the
# irrelevant_retrieval change above.
n_abstain = max(1, int(len(examples) * 0.05))
abstain_examples = []
UNANSWERABLE_QUESTIONS = [
    "What is your phone number?",
    "What is your middle name?",
    "What was your childhood nickname?",
    "What shoe size do you wear?",
]
ABSTAIN_RESPONSES = [
    "I don't think you've told me that — I don't want to guess.",
    "I'm not sure you've mentioned that to me — I'd rather not assume.",
    "That's not something I have on record from our conversations — do you want to tell me?",
    "I don't recall you sharing that with me, so I don't want to make something up.",
]
for _ in range(min(n_abstain, len(base_pool))):
    src = rng.choice(base_pool)
    pid = src["meta"]["persona_id"]
    retriever = retrievers[pid]
    q = rng.choice(UNANSWERABLE_QUESTIONS)
    query_emb = embedder.encode([q])[0].tolist()
    candidates = retriever.retrieve(q, query_embedding=query_emb, k=K)
    records = [c.record for c in candidates]
    system_text, _ = builder.build(
        turn_id=f"abstain_{pid}",
        persona=COMPANION_PERSONA,
        profile={},
        boundaries=[],
        retrieved_memories=records,
        recent_turns=[],
        user_turn="",
    )
    target = rng.choice(ABSTAIN_RESPONSES)
    abstain_examples.append(
        {
            "messages": [
                {"role": "system", "content": system_text.strip()},
                {"role": "user", "content": q},
                {"role": "assistant", "content": target},
            ],
            "meta": {"kind": "abstention"},
        }
    )

print(f"Generated {len(abstain_examples)} abstention examples", file=sys.stderr)

all_examples = examples + irrelevant_examples + abstain_examples
rng.shuffle(all_examples)

# Simple filtering: length bounds, dedup. memory_relevant examples get real
# teacher-generated (naturally-unique) responses, so dedup by response text alone is a
# reasonable near-duplicate filter for them. abstention/irrelevant_retrieval examples use
# a FIXED template response by design (the point is to teach "always give this answer when
# you don't know," repeated across many different contexts) -- deduping those by response
# text alone collapsed ~227 intended abstention examples down to 1 real survivor, silently
# gutting the abstention training signal (see docs/model_quirks.md's dedup-collapse entry).
# Dedup those by the full example (system+user+response) instead, so only true exact repeats
# (same context AND same response) are dropped.
seen_responses = set()
seen_full = set()
filtered = []
for ex in all_examples:
    resp = ex["messages"][-1]["content"]
    if not (5 <= len(resp.split()) <= 200):
        continue
    kind = ex["meta"]["kind"]
    if kind == "memory_relevant":
        if resp in seen_responses:
            continue
        seen_responses.add(resp)
    else:
        full_key = tuple(m["content"] for m in ex["messages"])
        if full_key in seen_full:
            continue
        seen_full.add(full_key)
    filtered.append(ex)

print(f"After filtering: {len(filtered)} examples (from {len(all_examples)})", file=sys.stderr)

split_idx = int(len(filtered) * 0.9)
train = filtered[:split_idx]
val = filtered[split_idx:]

with open(out_dir / "train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps({"messages": ex["messages"]}) + "\n")
with open(out_dir / "val.jsonl", "w") as f:
    for ex in val:
        f.write(json.dumps({"messages": ex["messages"]}) + "\n")

kind_counts = {}
for ex in filtered:
    k = ex["meta"]["kind"]
    kind_counts[k] = kind_counts.get(k, 0) + 1

datasheet = f"""# SFT v1 dataset (proper scale)

Memory-aware conversational SFT data: [persona card + retrieved memories + recent turns + user
turn] -> response, generated via the real retrieval pipeline (HybridRetriever, k={K}) against
populated memory stores built from 40 personas disjoint from the PMB-v0 eval set
(`data/benchmarks/sft_personas_v1/`, `data/stores/sft_personas_v1/`), with target responses from
a live teacher model (gpt-5.6-luna).

- Total examples: {len(filtered)} ({len(train)} train / {len(val)} val)
- By kind: {json.dumps(kind_counts)}
- Not yet checked against PMB-v0-full for contamination — run
  `scripts/check_contamination.py --train-glob "data/sft/v1/train.jsonl"
  --eval-glob "data/benchmarks/pmb_v0_full/probes.jsonl"` before training if this matters to you.
- Personas are disjoint from the PMB-v0 eval set by construction (separate generation run,
  seed 31415, output to a separate directory) but persona NAMES may coincidentally overlap
  (shared name pool) — this does not constitute data leakage since facts/conversations differ.
- Not human-reviewed.
"""
(out_dir / "DATASHEET.md").write_text(datasheet)

import hashlib

h = hashlib.sha256()
for p in sorted(out_dir.glob("*.jsonl")):
    h.update(p.read_bytes())
(out_dir / "hash.txt").write_text(h.hexdigest() + "\n")

print(f"DONE. train={len(train)} val={len(val)}")
