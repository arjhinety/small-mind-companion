import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from onebee.context.builder import ContextBuilder
from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.harness import run_harness, save_harness_result
from onebee.evaluation.metrics.personalized import Probe
from onebee.inference.engine import GenerationConfig, HFEngine
from onebee.memory.store import MemoryStore
from onebee.retrieval.router import HybridRetriever

probes_path = Path("data/benchmarks/pmb_v0_full/probes.jsonl")
probes: list[Probe] = []
with open(probes_path) as f:
    for line in f:
        probes.append(Probe(**json.loads(line)))

print(f"Loaded {len(probes)} probes", file=sys.stderr)

stores_dir = Path("data/stores/pmb_v0_full")
stores: dict[str, MemoryStore] = {}
retrievers: dict[str, HybridRetriever] = {}
for db_path in stores_dir.glob("*.db"):
    persona_id = db_path.stem
    # Study 001 ran with search results in table order (ERRATA E42); keep that to reproduce it.
    store = MemoryStore(str(db_path), legacy_table_order=True)
    stores[persona_id] = store
    retrievers[persona_id] = HybridRetriever(store)

print(f"Loaded {len(stores)} memory stores: {list(stores.keys())}", file=sys.stderr)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
engine = HFEngine("outputs/sft/v1/merged")
engine.load()
config = GenerationConfig(max_new_tokens=64)
judge = OpenAIJudge(model="gpt-5.6-luna")
builder = ContextBuilder()

K = 8
COMPANION_PERSONA = {
    "name": "the companion",
    "description": (
        "roleplaying as a warm AI companion in an ongoing relationship with the user. "
        "You have a real memory of things the user has told you before — the memories "
        "listed above ARE things you remember about THEM, not facts about yourself. "
        "When the user asks a question addressed to 'you' (e.g. 'what company do you "
        "work for'), they are usually asking you to recall something about THEM using "
        "your memory of them, not asking about yourself as an AI. Answer directly and "
        "naturally as their companion who already knows this, using the memories above "
        "when relevant. Do not say you are an AI without memory or access to personal "
        "information — you DO have the memories listed above, use them. If none of the "
        "memories actually answer the question, say you don't know rather than guessing."
    ),
    "traits": [],
}

n_done = 0


def response_fn(probe: Probe):
    global n_done
    retriever = retrievers.get(probe.persona_id)
    retrieved_ids: list[str] = []
    system_text = ""
    if retriever is not None:
        query_emb = embedder.encode([probe.question])[0].tolist()
        candidates = retriever.retrieve(probe.question, query_embedding=query_emb, k=K)
        retrieved_records = [c.record for c in candidates]
        retrieved_ids = [c.memory_id for c in candidates]
        system_text, _trace = builder.build(
            turn_id=probe.probe_id,
            persona=COMPANION_PERSONA,
            profile={},
            boundaries=[],
            retrieved_memories=retrieved_records,
            recent_turns=[],
            user_turn="",
        )
        system_text = system_text.strip()

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": probe.question})

    result = engine.generate(messages, config)
    n_done += 1
    if n_done % 25 == 0:
        print(f"  ...{n_done}/{len(probes)}", file=sys.stderr)
    return result.text, retrieved_ids


result = run_harness(probes, response_fn, judge=judge, system_name="E_sft_memory_v1", seed=1337)
save_harness_result(result, "results/v1_scale/E_sft_memory")
print("Saved to results/v1_scale/E_sft_memory", file=sys.stderr)
print(json.dumps(result.metrics, indent=2))
print("per_category:", json.dumps(result.per_category, indent=2))
