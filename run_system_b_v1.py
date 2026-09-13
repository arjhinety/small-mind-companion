import json
import sys
from pathlib import Path

from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.harness import run_harness, save_harness_result
from onebee.evaluation.metrics.personalized import Probe
from onebee.inference.engine import GenerationConfig, HFEngine

probes_path = Path("data/benchmarks/pmb_v0_full/probes.jsonl")
probes: list[Probe] = []
with open(probes_path) as f:
    for line in f:
        probes.append(Probe(**json.loads(line)))

print(f"Loaded {len(probes)} probes", file=sys.stderr)

engine = HFEngine("outputs/sft/v1/merged")
engine.load()
config = GenerationConfig(max_new_tokens=64)

judge = OpenAIJudge(model="gpt-5.6-luna")


def response_fn(probe: Probe):
    messages = [{"role": "user", "content": probe.question}]
    result = engine.generate(messages, config)
    return result.text, []


n_done = 0


def response_fn_logged(probe: Probe):
    global n_done
    r = response_fn(probe)
    n_done += 1
    if n_done % 25 == 0:
        print(f"  ...{n_done}/{len(probes)}", file=sys.stderr)
    return r


result = run_harness(
    probes,
    response_fn_logged,
    judge=judge,
    system_name="B_sft_v1",
    seed=1337,
)

save_harness_result(result, "results/v1_scale/B_sft")
print("Saved to results/v1_scale/B_sft", file=sys.stderr)
print(json.dumps(result.metrics, indent=2))
print("per_category:", json.dumps(result.per_category, indent=2))
