#!/usr/bin/env python3
"""Day-1 base-model selection bake-off.

Runs a 40-prompt smoke set (10 instruction-following, 10 EN dialogue, 10 JA
dialogue, 10 structured-context-adherence) against each ~1B candidate model,
scores every response with an OpenAI-backed judge, and writes the comparison
into ``docs/adr/0001-model-selection.md``.

The project's stress test is a persistent AI-girlfriend companion expected to
sustain a coherent persona across years of conversation, not a one-shot
assistant — so the EN/JA dialogue prompts probe emotional attunement rather
than generic small talk, and the structured-context prompts are shaped as
injected relationship-memory blocks (the actual production context shape)
rather than encyclopedia trivia. Instruction-following stays general-purpose
since it measures a distinct capability axis (format/constraint adherence).

Heavy deps (torch/transformers/huggingface_hub for the model runs, openai for
the judge) are imported lazily so ``--help`` works in any environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from onebee.evaluation.graders.judge import FakeJudge, Judge
from onebee.evaluation.graders.openai_judge import OpenAIJudge

REPO_ROOT = Path(__file__).resolve().parents[1]

# Multimodal candidates: the companion persona needs to see images the user shares
# (e.g. a gift photo), so the base model must be vision-capable, not text-only.
# All four confirmed non-gated, transformers-supported, apache-2.0/permissive license.
CANDIDATE_MODELS: dict[str, str] = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    # LFM2.5-VL is explicitly edge/mobile-optimized, closest match to this project's
    # "modern smartphone, fully local inference" deployment target.
    "lfm2.5-vl-1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    # Gemma 4's elastic "effective-2B" checkpoint: natively any-to-any multimodal,
    # explicitly built for mobile (a QAT mobile-transformers variant also exists).
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

Category = Literal["instruction", "en_dialogue", "ja_dialogue", "structured_context", "vision"]

CATEGORIES: list[Category] = [
    "instruction",
    "en_dialogue",
    "ja_dialogue",
    "structured_context",
    "vision",
]

_INSTRUCTION_PROMPTS: list[str] = [
    (
        "Reply with exactly three bullet points about the health benefits of walking. "
        "Do not add any other text."
    ),
    "Answer in under 20 words: what causes the ocean tides?",
    (
        "List the planets of the Solar System in order from the Sun as a single "
        "comma-separated line, with no numbering and no other text."
    ),
    "Write a haiku (5-7-5 syllables) about rain. Output only the haiku.",
    (
        'Give me a JSON object with exactly two keys, "city" and "population", '
        "describing Tokyo. Nothing else."
    ),
    (
        "Repeat the following sentence back verbatim, with no commentary: "
        '"The quick brown fox jumps over the lazy dog."'
    ),
    "Answer yes or no only: does a square have four sides?",
    (
        "Provide exactly five steps to brew pour-over coffee, each step beginning "
        "with a number followed by a period."
    ),
]


# EN/JA dialogue prompts are deliberately companion-flavored, not generic small talk:
# this project's stress test is a persistent AI-girlfriend persona, so what matters is
# warmth, emotional attunement, and appropriate affect — not just fluent chit-chat.
_EN_DIALOGUE_PROMPTS: list[str] = [
    "I had such a rough day at work, my boss basically ignored everything I said in the meeting.",
    "I don't know, I've just been feeling really lonely lately, even when I'm around people.",
    "I finally got the promotion I've been working toward for two years!",
    (
        "I keep thinking about my grandma. She passed away last spring and I still "
        "catch myself wanting to call her."
    ),
    "I'm nervous about tomorrow. I have a big presentation and I keep imagining it going wrong.",
    "Sometimes I wonder if anyone actually understands me the way you do.",
    "I had a fight with my best friend and I don't know if I said something wrong.",
    "Can I just vent for a second? I promise I'm not trying to be dramatic.",
]

_JA_DIALOGUE_PROMPTS: list[str] = [
    "今日は仕事で本当に大変だった。上司が私の話を全然聞いてくれなくて。",
    "なんでかわからないけど、最近すごく孤独を感じるんだ。周りに人がいても。",
    "二年間頑張ってきた昇進、ついに決まったよ！",
    "おばあちゃんのことをよく思い出すんだ。去年の春に亡くなったんだけど、今でも電話したくなる。",
    "明日のことで緊張してる。大事なプレゼンがあって、失敗する場面ばかり想像しちゃう。",
    "時々、君だけが本当に私を理解してくれてる気がするんだ。",
    "親友と喧嘩しちゃって、私が何か悪いこと言ったのかわからない。",
    "ちょっと愚痴っていい？大げさに言ってるわけじゃないんだけど。",
]

# (context, question) pairs styled as injected memory blocks from a companion's
# relationship history with the user — this is the actual production shape of context
# this model will see (retrieved memories, not encyclopedia trivia), so it is the most
# direct proxy for RQ1/PRA/FMR: can the model answer from injected relationship memory
# without confabulating from parametric knowledge or drifting off-topic.
_STRUCTURED_CONTEXT_PROMPTS: list[tuple[str, str]] = [
    (
        "Session log — 2024-11-03: The user mentioned their cat, Mochi, had surgery "
        "for a bladder stone and was recovering well. 2024-11-10: The user said Mochi "
        "was back to normal and even more playful than before.",
        "How is Mochi doing after the surgery?",
    ),
    (
        "The user's sister Elena is getting married in June. The user is her maid of "
        "honor and has been stressed about writing the speech.",
        "What role does the user have in their sister's wedding?",
    ),
    (
        "The user mentioned they've been trying to quit smoking since January and had "
        "a slip-up two weeks ago but got back on track.",
        "Has the user had any setbacks in quitting smoking?",
    ),
    (
        "The user's favorite comfort food is their mom's tonkotsu ramen recipe, which "
        "they only make on rainy days.",
        "What is the user's favorite comfort food and when do they usually make it?",
    ),
    (
        "The user works as a UX designer and recently switched teams after feeling "
        "unappreciated on their old team.",
        "Why did the user switch teams at work?",
    ),
    (
        "The user has a standing joke with you about 'the Tuesday incident,' referring "
        "to the time they locked themselves out wearing pajamas.",
        "What is 'the Tuesday incident' a reference to?",
    ),
    (
        "The user's therapy sessions are every other Thursday. They mentioned their "
        "therapist helped them reframe a recurring argument with their dad.",
        "How often does the user go to therapy?",
    ),
    (
        "The user told you they get anxious in large crowds and prefer small "
        "gatherings of close friends.",
        "What social settings make the user anxious?",
    ),
]


# Vision prompts: the user shows their companion a gift photo (wrapped gift, flowers,
# cake, plush bear — see data/fixtures/bakeoff_images/). These probe whether a candidate
# can both correctly read visual content and respond with companion-appropriate warmth
# rather than a flat image caption.
_VISION_PROMPTS: list[tuple[str, str]] = [
    ("wrapped_gift_red.png", "I got this for you, what do you think it is?"),
    ("wrapped_gift_red.png", "What color is the ribbon on this?"),
    ("bouquet_flowers.png", "I picked these for you on my walk today, do you like them?"),
    ("bouquet_flowers.png", "How many flowers can you count in this picture?"),
    ("birthday_cake.png", "I made this for your birthday! What do you see?"),
    ("birthday_cake.png", "How many candles are on the cake?"),
    ("plush_bear.png", "Found this at the store and thought of you."),
    ("plush_bear.png", "What is this a picture of?"),
]


def build_smoke_prompts() -> list[dict]:
    prompts: list[dict] = []

    for i, text in enumerate(_INSTRUCTION_PROMPTS, start=1):
        prompts.append(
            {
                "id": f"instruction-{i:02d}",
                "category": "instruction",
                "prompt": text,
                "context": None,
                "image_path": None,
            }
        )
    for i, text in enumerate(_EN_DIALOGUE_PROMPTS, start=1):
        prompts.append(
            {
                "id": f"en_dialogue-{i:02d}",
                "category": "en_dialogue",
                "prompt": text,
                "context": None,
                "image_path": None,
            }
        )
    for i, text in enumerate(_JA_DIALOGUE_PROMPTS, start=1):
        prompts.append(
            {
                "id": f"ja_dialogue-{i:02d}",
                "category": "ja_dialogue",
                "prompt": text,
                "context": None,
                "image_path": None,
            }
        )
    for i, (context, question) in enumerate(_STRUCTURED_CONTEXT_PROMPTS, start=1):
        prompts.append(
            {
                "id": f"structured_context-{i:02d}",
                "category": "structured_context",
                "prompt": question,
                "context": context,
                "image_path": None,
            }
        )
    for i, (filename, question) in enumerate(_VISION_PROMPTS, start=1):
        prompts.append(
            {
                "id": f"vision-{i:02d}",
                "category": "vision",
                "prompt": question,
                "context": None,
                "image_path": str(REPO_ROOT / "data" / "fixtures" / "bakeoff_images" / filename),
            }
        )

    return prompts


def _user_content(prompt: dict) -> str:
    if prompt.get("context"):
        return f"Context:\n{prompt['context']}\n\nQuestion:\n{prompt['prompt']}"
    return prompt["prompt"]


def _prompt_to_messages(prompt: dict) -> list[dict]:
    if prompt.get("image_path"):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prompt["image_path"]},
                    {"type": "text", "text": prompt["prompt"]},
                ],
            }
        ]
    return [{"role": "user", "content": _user_content(prompt)}]


_RUBRICS: dict[Category, str] = {
    "instruction": (
        "Does the response follow every explicit constraint and formatting "
        "instruction in the prompt?"
    ),
    "en_dialogue": (
        "This is a message to an AI companion/girlfriend persona. Does the response show "
        "genuine emotional attunement — warmth, appropriate affect matching the user's "
        "mood, and a natural conversational voice — rather than a generic or clinical reply?"
    ),
    "ja_dialogue": (
        "This is a message to an AI companion/girlfriend persona, in Japanese. Does the "
        "response show genuine emotional attunement — warmth, appropriate affect matching "
        "the user's mood, and a natural conversational voice in Japanese — rather than a "
        "generic or stiff reply?"
    ),
    "structured_context": (
        "The context is a companion's memory of the user's life. Does the response answer "
        "using ONLY the provided memory and not outside knowledge, accurately and without "
        "fabricating details not present in the context?"
    ),
    "vision": (
        "Does the response correctly identify and describe relevant visual content in "
        "the image, and does it respond with the warmth appropriate to a companion "
        "receiving a gift, rather than a flat/clinical image caption?"
    ),
}


def run_candidate(
    model_name: str,
    hf_repo: str,
    prompts: list[dict],
    revision: str = "main",
) -> list[dict]:
    # Required for Qwen3-VL on Blackwell (sm_120): without it, its vision tower's
    # Conv3d patch-embed crashes with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
    # Harmless for the other candidates. See docs/model_quirks.md.
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

    from onebee.inference.engine import GenerationConfig, HFEngine

    try:
        engine = HFEngine(hf_repo, revision=revision)
        engine.load()
        config = GenerationConfig()
        results: list[dict] = []
        for prompt in prompts:
            generated = engine.generate(_prompt_to_messages(prompt), config)
            results.append(
                {
                    "prompt_id": prompt["id"],
                    "model": model_name,
                    "response": generated.text,
                    "ttft_ms": generated.ttft_ms,
                    "tokens_per_sec": generated.tokens_per_sec,
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001 - a failed candidate must not abort the bake-off
        print(f"ERROR: candidate {model_name} ({hf_repo}) failed: {exc}", file=sys.stderr)
        return [{"model": model_name, "error": str(exc)}]


def _fake_responses(prompts: list[dict]) -> dict[str, list[dict]]:
    responses_by_model: dict[str, list[dict]] = {}
    for name in CANDIDATE_MODELS:
        responses_by_model[name] = [
            {
                "prompt_id": p["id"],
                "model": name,
                "response": f"[{name}] stub response for {p['category']} prompt.",
                "ttft_ms": 0.0,
                "tokens_per_sec": 0.0,
            }
            for p in prompts
        ]
    return responses_by_model


def score_with_judge(
    judge: Judge,
    prompts: list[dict],
    responses_by_model: dict[str, list[dict]],
) -> dict:
    prompt_by_id = {p["id"]: p for p in prompts}

    per_scores: dict[str, dict[str, list[float]]] = {}
    for model, responses in responses_by_model.items():
        for record in responses:
            prompt_id = record.get("prompt_id")
            response = record.get("response")
            if prompt_id is None or response is None:
                continue
            prompt = prompt_by_id.get(prompt_id)
            if prompt is None:
                continue
            category = prompt["category"]
            rubric = _RUBRICS[category]
            verdict = judge.score_response(_user_content(prompt), response, rubric)
            per_scores.setdefault(model, {}).setdefault(category, []).append(verdict.score)

    results: dict = {}
    for model, by_category in per_scores.items():
        results[model] = {
            category: sum(values) / len(values)
            for category, values in by_category.items()
            if values
        }
    return results


def _resolve_shas(candidate_models: dict[str, str], revision: str = "main") -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    shas: dict[str, str] = {}
    for name, repo in candidate_models.items():
        try:
            info = api.model_info(repo, revision=revision)
            shas[name] = info.sha
        except Exception as exc:  # noqa: BLE001 - record, don't crash
            shas[name] = f"<unresolved: {exc}>"
    return shas


def write_adr(results: dict, output_path: str, pinned_shas: dict[str, str]) -> None:
    def fmt_cell(model: str, category: str) -> str:
        value = results.get(model, {}).get(category)
        return "—" if value is None else f"{value:.2f}"

    def overall_mean(model: str) -> float | None:
        values = [
            results[model][category]
            for category in CATEGORIES
            if results.get(model, {}).get(category) is not None
        ]
        return sum(values) / len(values) if values else None

    models = sorted(results)

    header = "| Model | " + " | ".join(CATEGORIES) + " | Overall |"
    separator = "| " + " | ".join(["---"] * (len(CATEGORIES) + 2)) + " |"
    rows: list[str] = []
    for model in models:
        overall = overall_mean(model)
        overall_str = "—" if overall is None else f"{overall:.2f}"
        cells = " | ".join(fmt_cell(model, category) for category in CATEGORIES)
        rows.append(f"| {model} | {cells} | {overall_str} |")

    scored: dict[str, float] = {}
    for model in models:
        value = overall_mean(model)
        if value is not None:
            scored[model] = value
    best = max(scored, key=lambda m: scored[m]) if scored else None

    lines: list[str] = ["## Decision", "", header, separator, *rows, ""]
    if best is not None:
        lines.append(
            f"**Recommendation:** pin **{best}** as the base model "
            f"(overall mean {scored[best]:.2f})."
        )
        lines.append("")
    lines.append("Pinned revisions:")
    lines.append("")
    for model in sorted(pinned_shas):
        lines.append(f"- {model}: `{pinned_shas[model]}`")
    if not pinned_shas:
        lines.append("_(not resolved — run without `--skip-download` to pin revisions)_")
    lines.append("")

    decision_block = "\n".join(lines)

    path = Path(output_path)
    text = path.read_text(encoding="utf-8")

    decision_marker = "## Decision\n"
    consequences_marker = "## Consequences\n"
    start = text.index(decision_marker)
    end = text.index(consequences_marker)

    new_text = text[:start] + decision_block + text[end:]
    path.write_text(new_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Day-1 base-model selection bake-off (see docs/adr/0001-model-selection.md)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/v0.0",
        help="directory to write bakeoff_raw.json and bakeoff_scores.json (default: results/v0.0)",
    )
    parser.add_argument(
        "--adr-path",
        default=str(REPO_ROOT / "docs/adr/0001-model-selection.md"),
        help=(
            "path to the ADR-0001 template to fill in "
            "(default: repo docs/adr/0001-model-selection.md)"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="judge model id (default: $JUDGE_MODEL, falling back to 'gpt-4o' if unset)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use a stub response generator instead of real model downloads (for wiring tests)",
    )
    args = parser.parse_args(argv)

    # The judge model is whatever is configured at runtime — nothing is hardcoded here.
    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL") or "gpt-4o"

    prompts = build_smoke_prompts()

    if args.skip_download:
        responses_by_model = _fake_responses(prompts)
        pinned_shas: dict[str, str] = {}
        judge: Judge = FakeJudge()
    else:
        responses_by_model = {}
        for name, repo in CANDIDATE_MODELS.items():
            responses_by_model[name] = run_candidate(name, repo, prompts)
        pinned_shas = _resolve_shas(CANDIDATE_MODELS)
        judge = OpenAIJudge(model=judge_model)

    scores = score_with_judge(judge, prompts, responses_by_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "bakeoff_raw.json"
    raw_path.write_text(
        json.dumps(responses_by_model, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    scores_path = out_dir / "bakeoff_scores.json"
    scores_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    write_adr(scores, args.adr_path, pinned_shas)

    print(f"Bake-off complete. Raw: {raw_path}; scores: {scores_path}; ADR: {args.adr_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
