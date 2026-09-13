import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from onebee.context.builder import ContextBuilder
from onebee.evaluation.graders.judge import dual_order_score
from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.metrics.personalized import Probe
from onebee.inference.engine import GenerationConfig, HFEngine
from onebee.memory.store import MemoryStore
from onebee.retrieval.router import HybridRetriever

# H23: does on-policy distillation from a larger local teacher (gemma-4-E4B-it) improve
# response quality without degrading persona consistency? Compares C (pre-distillation:
# SFT+DPO v1_scale+memory) vs F (post-distillation: SFT+DPO+distill+memory), same sampling
# parameters as the prior C_vs_E comparisons for direct comparability.
probes_path = Path("data/benchmarks/pmb_v0_full/probes.jsonl")
all_probes: list[Probe] = []
with open(probes_path) as f:
    for line in f:
        all_probes.append(Probe(**json.loads(line)))

by_category: dict[str, list[Probe]] = defaultdict(list)
for p in all_probes:
    if p.answerable:
        by_category[p.category].append(p)

rng = random.Random(2718)
PER_CATEGORY = 15
subsample: list[Probe] = []
for cat, plist in by_category.items():
    rng.shuffle(plist)
    subsample.extend(plist[:PER_CATEGORY])

print(f"Subsample: {len(subsample)} answerable probes", file=sys.stderr)

stores_dir = Path("data/stores/pmb_v0_full")
retrievers = {}
for db_path in stores_dir.glob("*.db"):
    store = MemoryStore(str(db_path))
    retrievers[db_path.stem] = HybridRetriever(store)

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
builder = ContextBuilder()
judge = OpenAIJudge(model="gpt-5.6-luna")

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

K = 8


def build_messages(probe: Probe):
    retriever = retrievers.get(probe.persona_id)
    if retriever is None:
        return [{"role": "user", "content": probe.question}]
    query_emb = embedder.encode([probe.question])[0].tolist()
    candidates = retriever.retrieve(probe.question, query_embedding=query_emb, k=K)
    records = [c.record for c in candidates]
    system_text, _ = builder.build(
        turn_id=probe.probe_id,
        persona=COMPANION_PERSONA,
        profile={},
        boundaries=[],
        retrieved_memories=records,
        recent_turns=[],
        user_turn="",
    )
    system_text = system_text.strip()
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": probe.question})
    return messages


print("Loading System C (SFT+DPO v1_scale, pre-distillation)...", file=sys.stderr)
engine_c = HFEngine("outputs/dpo/v1_scale/merged")
engine_c.load()
config = GenerationConfig(max_new_tokens=64)

c_responses = {}
for i, p in enumerate(subsample):
    r = engine_c.generate(build_messages(p), config)
    c_responses[p.probe_id] = r.text
    if (i + 1) % 25 == 0:
        print(f"  C: {i+1}/{len(subsample)}", file=sys.stderr)

del engine_c
import gc

import torch

gc.collect()
torch.cuda.empty_cache()

print("Loading System F (post-distillation, H23)...", file=sys.stderr)
engine_f = HFEngine("outputs/distill/v1/merged")
engine_f.load()

f_responses = {}
for i, p in enumerate(subsample):
    r = engine_f.generate(build_messages(p), config)
    f_responses[p.probe_id] = r.text
    if (i + 1) % 25 == 0:
        print(f"  F: {i+1}/{len(subsample)}", file=sys.stderr)

rubric = (
    "Which response is a better answer for a warm AI companion to give: more natural, "
    "more clearly in-character (not a generic AI disclaimer), and appropriately uses "
    "the companion's memory of the user when relevant? Score 0.0 if response A is "
    "better, 1.0 if response B is better, 0.5 if tied."
)

results = []
c_wins = 0
f_wins = 0
ties = 0
for p in subsample:
    resp_c = c_responses[p.probe_id]
    resp_f = f_responses[p.probe_id]
    score = dual_order_score(judge, p.question, resp_c, resp_f, rubric)
    # score near 1.0 means F (response_b) preferred, near 0.0 means C (response_a) preferred
    if score > 0.6:
        f_wins += 1
    elif score < 0.4:
        c_wins += 1
    else:
        ties += 1
    results.append(
        {
            "probe_id": p.probe_id,
            "category": p.category,
            "question": p.question,
            "response_c": resp_c,
            "response_f": resp_f,
            "dual_order_score": score,
        }
    )

print(
    f"C (pre-distill) wins: {c_wins}, F (post-distill, H23) wins: {f_wins}, ties: {ties}",
    file=sys.stderr,
)

out_dir = Path("results/v1_scale/C_vs_F_distill_pairwise")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "raw.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
summary = {
    "n": len(subsample),
    "c_wins": c_wins,
    "f_wins": f_wins,
    "ties": ties,
    "c_win_rate": c_wins / len(subsample),
    "f_win_rate": f_wins / len(subsample),
}
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
