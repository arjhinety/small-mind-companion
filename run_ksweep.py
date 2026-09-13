import json
import random
import sys
from collections import defaultdict
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
all_probes: list[Probe] = []
with open(probes_path) as f:
    for line in f:
        all_probes.append(Probe(**json.loads(line)))

# Stratified subsample: ~15 per category, fixed seed for reproducibility.
by_category: dict[str, list[Probe]] = defaultdict(list)
for p in all_probes:
    by_category[p.category].append(p)

rng = random.Random(1337)
PER_CATEGORY = 15
subsample: list[Probe] = []
for cat, plist in by_category.items():
    rng.shuffle(plist)
    subsample.extend(plist[:PER_CATEGORY])

print(
    f"Subsample: {len(subsample)} probes "
    f"({ {c: min(PER_CATEGORY, len(v)) for c, v in by_category.items()} })",
    file=sys.stderr,
)

stores_dir = Path("data/stores/pmb_v0_full")
retrievers: dict[str, HybridRetriever] = {}
for db_path in stores_dir.glob("*.db"):
    store = MemoryStore(str(db_path))
    retrievers[db_path.stem] = HybridRetriever(store)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
engine = HFEngine("google/gemma-4-E2B-it", revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7")
engine.load()
config = GenerationConfig(max_new_tokens=64)
judge = OpenAIJudge(model="gpt-5.6-luna")
builder = ContextBuilder()

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

K_VALUES = [0, 2, 4, 8, 16]
summary = {}

for k in K_VALUES:
    n_done = 0

    def response_fn(probe: Probe, k=k):
        global n_done
        retriever = retrievers.get(probe.persona_id)
        retrieved_ids: list[str] = []
        system_text = ""
        if retriever is not None and k > 0:
            query_emb = embedder.encode([probe.question])[0].tolist()
            candidates = retriever.retrieve(probe.question, query_embedding=query_emb, k=k)
            retrieved_records = [c.record for c in candidates]
            retrieved_ids = [c.memory_id for c in candidates]
            system_text, _ = builder.build(
                turn_id=probe.probe_id,
                persona=COMPANION_PERSONA,
                profile={},
                boundaries=[],
                retrieved_memories=retrieved_records,
                recent_turns=[],
                user_turn="",
            )
            system_text = system_text.strip()
        elif retriever is not None:
            # k=0: still use the companion persona/system framing, just with zero
            # injected memories, to isolate the effect of memory COUNT specifically
            # rather than conflating it with the presence/absence of the companion framing.
            system_text, _ = builder.build(
                turn_id=probe.probe_id,
                persona=COMPANION_PERSONA,
                profile={},
                boundaries=[],
                retrieved_memories=[],
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
            print(f"  k={k}: ...{n_done}/{len(subsample)}", file=sys.stderr)
        return result.text, retrieved_ids

    print(f"=== Running k={k} ===", file=sys.stderr)
    result = run_harness(subsample, response_fn, judge=judge, system_name=f"ksweep_k{k}", seed=1337)
    save_harness_result(result, f"results/v0.1/ksweep/k{k}")
    summary[k] = result.metrics
    print(f"k={k}: {json.dumps(result.metrics)}", file=sys.stderr)

Path("results/v0.1/ksweep").mkdir(parents=True, exist_ok=True)
with open("results/v0.1/ksweep/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
