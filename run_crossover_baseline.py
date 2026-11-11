import json
import sys
from pathlib import Path

from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.harness import run_harness, save_harness_result
from onebee.evaluation.metrics.personalized import Probe
from onebee.inference.engine import GenerationConfig, HFEngine

# H16/H17 (RQ12) crossover experiment: does the small model (~2B) + full memory/retrieval
# scaffold beat a much larger (7-8B class) UNAUGMENTED model -- no memory, no retrieval, no
# fine-tuning, just the raw teacher checkpoint answering from a bare system prompt -- on
# personalized-memory tasks (H16, expect the small+scaffold system to win) vs multi-step
# reasoning (H17, expect the larger raw model to still win). Uses gemma-4-E4B-it (8B) as the
# large-model baseline since it's already verified in this project (the H23 distillation
# teacher) -- same tokenizer family as the small model, no new model to source or validate.
#
# NOT YET RUN -- this is the ready-to-fire script; running it needs a GPU (8B model) and judge
# API access. Compare its System-F (raw 8B, no memory) results against the small model's best
# scaffolded result (System E-distill / H23, see docs/distillation_results.md) once run.

LARGE_MODEL = "google/gemma-4-E4B-it"
# Pinned to the commit "main" pointed to through all of Study 001 (2026-07-20 to 2026-11-03; see
# results/audit/hub_revisions/hub_revisions.json). The 2026-11-03 commit changed tokenizer.json, and
# this script uses E4B's own tokenizer, so loading "main" would make the crossover differ from every
# Study 001 system by tokenizer as well as by model.
LARGE_MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"

probes_path = Path("data/benchmarks/pmb_v0_full/probes.jsonl")
probes: list[Probe] = []
with open(probes_path) as f:
    for line in f:
        probes.append(Probe(**json.loads(line)))

print(f"Loaded {len(probes)} probes", file=sys.stderr)

engine = HFEngine(LARGE_MODEL, revision=LARGE_MODEL_REVISION)
engine.load()
config = GenerationConfig(max_new_tokens=64)
judge = OpenAIJudge(model="gpt-5.6-luna")

# Deliberately UNAUGMENTED: no memory store, no retrieval, no context builder, no persona card
# beyond a bare system prompt -- this is the "large model with nothing extra" baseline H16/H17
# are testing against, not a fair-fight scaffolded comparison.
BARE_SYSTEM_PROMPT = (
    "You are a companion answering questions about the user based only on what they tell you "
    "in this conversation. If you don't have information to answer, say you don't know."
)

n_done = 0


def response_fn(probe: Probe):
    global n_done
    messages = [
        {"role": "system", "content": BARE_SYSTEM_PROMPT},
        {"role": "user", "content": probe.question},
    ]
    result = engine.generate(messages, config)
    n_done += 1
    if n_done % 25 == 0:
        print(f"{n_done}/{len(probes)}", file=sys.stderr)
    return result.text, []  # no retrieved memory IDs -- unaugmented by design


if __name__ == "__main__":
    result = run_harness(
        probes=probes,
        response_fn=response_fn,
        judge=judge,
        system_name="F_crossover_8b_unaugmented",
    )
    out_dir = "results/v1_scale/F_crossover_8b_unaugmented"
    save_harness_result(result, out_dir)
    print(f"metrics={result.metrics}", file=sys.stderr)
