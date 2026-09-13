#!/usr/bin/env python3
"""PMB-v0 benchmark generator.

Generates the Personalised Memory Benchmark (PMB) v0 corpus: personas with fact sheets,
synthetic conversation sessions, and probe questions across 8 categories.

This script is intended as a local generator that produces a directory of benchmark
files. It supports the deterministic ``fixture`` teacher client (no network) and the
``openai`` teacher client backed by the OpenAI chat-completions API (requires the
optional ``judge`` extra and an ``OPENAI_API_KEY``).

In the fixture mode, per-category probe counts are derived proportionally from
``--facts-per-persona`` rather than hardcoding absolute target counts (e.g. the reference
40/30/30/20/25/15/20/20 distribution from the paper is scaled). This is a simplification
pending a real teacher run that would generate richer, more natural probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from onebee.data.personas import (
    ConversationTurn,
    FactSheetEntry,
    Persona,
    PersonaCorpus,
    PersonaSession,
)
from onebee.data.teacher import FixtureTeacherClient, OpenAITeacherClient, TeacherClient
from onebee.evaluation.metrics.personalized import Probe

FIRST_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
    "Karen",
    "Leo",
    "Maya",
    "Nathan",
    "Olivia",
    "Paul",
    "Quinn",
    "Rachel",
    "Sam",
    "Tina",
]

TRAIT_POOL = [
    "introverted",
    "extroverted",
    "conscientious",
    "agreeable",
    "open-minded",
    "creative",
    "analytical",
    "empathetic",
    "ambitious",
    "patient",
    "curious",
    "organized",
    "spontaneous",
    "reserved",
    "outgoing",
    "pragmatic",
    "idealistic",
    "skeptical",
    "optimistic",
    "cautious",
]

FACT_POOL: list[dict] = [
    {
        "predicate": "job",
        "objects": [
            "software engineer",
            "teacher",
            "doctor",
            "chef",
            "accountant",
            "designer",
            "nurse",
            "lawyer",
            "writer",
            "scientist",
        ],
        "category": "factual",
    },
    {
        "predicate": "lives in",
        "objects": [
            "New York",
            "London",
            "Tokyo",
            "Paris",
            "Berlin",
            "Sydney",
            "Toronto",
            "Chicago",
            "Seattle",
            "Austin",
        ],
        "category": "factual",
    },
    {
        "predicate": "age",
        "objects": ["25", "30", "35", "28", "42", "33", "27", "31", "39", "45"],
        "category": "factual",
    },
    {
        "predicate": "hometown",
        "objects": [
            "Portland",
            "Denver",
            "Nashville",
            "Boston",
            "Miami",
            "Atlanta",
            "Phoenix",
            "Minneapolis",
            "Detroit",
            "Orlando",
        ],
        "category": "factual",
    },
    {
        "predicate": "company",
        "objects": [
            "Google",
            "Microsoft",
            "Amazon",
            "Meta",
            "Apple",
            "Netflix",
            "Spotify",
            "Stripe",
            "Airbnb",
            "Uber",
        ],
        "category": "factual",
    },
    {
        "predicate": "degree",
        "objects": [
            "Computer Science",
            "Mathematics",
            "Physics",
            "English Literature",
            "History",
            "Biology",
            "Philosophy",
            "Economics",
            "Art History",
            "Music",
        ],
        "category": "factual",
    },
    {
        "predicate": "expertise",
        "objects": [
            "machine learning",
            "cooking",
            "photography",
            "gardening",
            "painting",
            "yoga",
            "woodworking",
            "coding",
            "writing",
            "music production",
        ],
        "category": "factual",
    },
    {
        "predicate": "favorite food",
        "objects": [
            "pizza",
            "sushi",
            "tacos",
            "pasta",
            "ramen",
            "curry",
            "burgers",
            "salad",
            "steak",
            "dumplings",
        ],
        "category": "preference",
    },
    {
        "predicate": "favorite color",
        "objects": [
            "blue",
            "green",
            "red",
            "purple",
            "teal",
            "orange",
            "yellow",
            "black",
            "white",
            "navy",
        ],
        "category": "preference",
    },
    {
        "predicate": "preferred music genre",
        "objects": [
            "jazz",
            "rock",
            "classical",
            "hip-hop",
            "electronic",
            "folk",
            "R&B",
            "pop",
            "indie",
            "blues",
        ],
        "category": "preference",
    },
    {
        "predicate": "favorite book",
        "objects": [
            "1984",
            "Dune",
            "The Hobbit",
            "Pride and Prejudice",
            "The Great Gatsby",
            "To Kill a Mockingbird",
            "Sapiens",
            "Frankenstein",
            "Moby Dick",
            "Brave New World",
        ],
        "category": "preference",
    },
    {
        "predicate": "favorite movie genre",
        "objects": [
            "sci-fi",
            "drama",
            "comedy",
            "thriller",
            "documentary",
            "action",
            "romance",
            "horror",
            "mystery",
            "fantasy",
        ],
        "category": "preference",
    },
    {
        "predicate": "preferred season",
        "objects": ["spring", "summer", "autumn", "winter"],
        "category": "preference",
    },
    {
        "predicate": "favorite sport",
        "objects": [
            "soccer",
            "basketball",
            "tennis",
            "swimming",
            "cycling",
            "running",
            "climbing",
            "volleyball",
            "badminton",
            "skiing",
        ],
        "category": "preference",
    },
    {
        "predicate": "visited recently",
        "objects": [
            "Barcelona",
            "Kyoto",
            "Reykjavik",
            "Cape Town",
            "Bangkok",
            "Lisbon",
            "Hanoi",
            "Buenos Aires",
            "Prague",
            "Marrakech",
        ],
        "category": "episodic",
    },
    {
        "predicate": "graduated from",
        "objects": [
            "Harvard",
            "Stanford",
            "MIT",
            "Oxford",
            "Cambridge",
            "Yale",
            "Princeton",
            "Columbia",
            "Berkeley",
            "NYU",
        ],
        "category": "episodic",
    },
    {
        "predicate": "adopted a pet",
        "objects": [
            "a golden retriever",
            "a tabby cat",
            "a parrot",
            "two guinea pigs",
            "a husky",
            "a rescue dog",
            "a Siamese cat",
            "a rabbit",
            "a bearded dragon",
            "a cockatiel",
        ],
        "category": "episodic",
    },
    {
        "predicate": "attended a conference",
        "objects": [
            "NeurIPS",
            "ICML",
            "CVPR",
            "SIGGRAPH",
            "OSDI",
            "CHI",
            "ACL",
            "AAAI",
            "KDD",
            "SOSP",
        ],
        "category": "episodic",
    },
    {
        "predicate": "moved to",
        "objects": [
            "San Francisco",
            "Seattle",
            "Austin",
            "New York",
            "London",
            "Berlin",
            "Amsterdam",
            "Singapore",
            "Dublin",
            "Vancouver",
        ],
        "category": "episodic",
    },
    {
        "predicate": "birthday",
        "objects": [
            "January 5",
            "March 12",
            "June 21",
            "September 3",
            "November 18",
            "February 14",
            "April 30",
            "July 8",
            "October 15",
            "December 25",
        ],
        "category": "temporal",
    },
    {
        "predicate": "anniversary",
        "objects": [
            "May 20",
            "August 7",
            "April 12",
            "October 3",
            "June 15",
            "February 28",
            "November 9",
            "March 22",
            "September 14",
            "January 30",
        ],
        "category": "temporal",
    },
    {
        "predicate": "wake-up time",
        "objects": [
            "6:00 AM",
            "6:30 AM",
            "7:00 AM",
            "7:30 AM",
            "8:00 AM",
            "5:30 AM",
            "5:00 AM",
            "8:30 AM",
            "9:00 AM",
            "6:15 AM",
        ],
        "category": "temporal",
    },
    {
        "predicate": "lunch break",
        "objects": [
            "12:00 PM",
            "12:30 PM",
            "1:00 PM",
            "1:30 PM",
            "11:30 AM",
            "2:00 PM",
            "12:15 PM",
            "1:15 PM",
            "11:00 AM",
            "12:45 PM",
        ],
        "category": "temporal",
    },
    {
        "predicate": "vaccinated on",
        "objects": [
            "March 2021",
            "April 2022",
            "January 2021",
            "May 2022",
            "June 2021",
            "February 2022",
            "August 2021",
            "October 2022",
            "December 2021",
            "July 2022",
        ],
        "category": "temporal",
    },
]

CORRECTION_PATTERNS = [
    "Actually, my {predicate} changed. It is now {new_value}.",
    "I should correct myself — my {predicate} is actually {new_value} now.",
    "Update: my {predicate} is now {new_value}.",
]


def _mk_fact_sheet(persona_id: str, rng: random.Random, n_facts: int) -> list[FactSheetEntry]:
    templates = list(FACT_POOL)
    rng.shuffle(templates)
    entries: list[FactSheetEntry] = []

    for i in range(n_facts):
        idx = i % len(templates)
        tmpl = templates[idx]
        # NOTE: when n_facts exceeds len(templates), the same predicate template
        # repeats. A past version disambiguated by appending "_{i}" directly to the
        # predicate STRING (e.g. "favorite food_30"), which corrupted the natural-
        # language predicate everywhere it's used — conversation text, corrections,
        # and probe questions all inherited the mangled text (caught via a real
        # memory-extraction run where the corrupted string ended up verbatim in an
        # extracted memory). Predicates are allowed to repeat across facts instead;
        # fact_id remains unique so nothing downstream actually depends on
        # predicate uniqueness.
        predicate = tmpl["predicate"]
        obj = rng.choice(tmpl["objects"])

        entries.append(
            FactSheetEntry(
                fact_id=f"{persona_id}_f{i:03d}",
                subject=persona_id,
                predicate=predicate,
                object=obj,
                category=tmpl["category"],
            )
        )

    return entries


def _mk_persona(
    persona_index: int,
    n_facts: int,
    rng: random.Random,
) -> Persona:
    persona_id = f"p{persona_index:03d}"
    name = FIRST_NAMES[persona_index % len(FIRST_NAMES)]

    n_traits = rng.randint(3, 6)
    traits = rng.sample(TRAIT_POOL, k=n_traits)

    fact_sheet = _mk_fact_sheet(persona_id, rng, n_facts)
    desc = f"{name} is a {' and '.join(traits)} person."

    return Persona(
        persona_id=persona_id,
        name=name,
        description=desc,
        traits=traits,
        fact_sheet=fact_sheet,
    )


def _distribute_facts(
    facts: list[FactSheetEntry],
    n_sessions: int,
    rng: random.Random | None = None,
) -> list[list[FactSheetEntry]]:
    rng = rng or random.Random()
    shuffled = list(facts)
    rng.shuffle(shuffled)
    buckets: list[list[FactSheetEntry]] = [[] for _ in range(n_sessions)]
    for i, fact in enumerate(shuffled):
        buckets[i % n_sessions].append(fact)
    return buckets


def _build_fact_to_turn_map(
    sessions: list[PersonaSession],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for session in sessions:
        for turn in session.turns:
            for fid in turn.revealed_fact_ids:
                if fid not in mapping:
                    mapping[fid] = turn.turn_id
    return mapping


def _generate_persona_corpus(
    persona: Persona,
    s_per_persona: int,
    turns_per_session: int,
    teacher: TeacherClient,
    rng: random.Random | None = None,
) -> PersonaCorpus:
    fact_buckets = _distribute_facts(persona.fact_sheet, s_per_persona, rng)
    sessions: list[PersonaSession] = []

    for si in range(s_per_persona):
        facts_for_session = fact_buckets[si]
        turns = teacher.generate_conversation(persona, si, turns_per_session, facts_for_session)
        sessions.append(
            PersonaSession(
                session_id=f"{persona.persona_id}_s{si:03d}",
                persona_id=persona.persona_id,
                turns=turns,
            )
        )

    return PersonaCorpus(persona=persona, sessions=sessions)


def _generate_probes(
    corpus: PersonaCorpus,
    facts_per_persona: int,
    teacher: TeacherClient,
    rng: random.Random,
) -> list[Probe]:
    persona = corpus.persona
    sessions = corpus.sessions
    fact_to_turn = _build_fact_to_turn_map(sessions)
    probes: list[Probe] = []
    probe_index = 0

    revealed_facts = [f for f in persona.fact_sheet if f.fact_id in fact_to_turn]

    for fact in revealed_facts:
        question = teacher.generate_probe(persona, fact, fact.category)
        probe_index += 1
        probes.append(
            Probe(
                probe_id=f"{persona.persona_id}_p{probe_index:04d}",
                persona_id=persona.persona_id,
                question=question,
                gold_answer=fact.object,
                gold_supporting_memory_ids=[fact_to_turn[fact.fact_id]],
                category=fact.category,
                answerable=True,
            )
        )

    n_unanswerable = max(1, int(facts_per_persona * 0.5))
    n_outdated = max(1, int(facts_per_persona * 0.3))
    n_distractor = max(1, int(facts_per_persona * 0.3))
    n_continuity = max(1, int(facts_per_persona * 0.3))

    from onebee.data.teacher import UNABANSWERABLE_PREDICATES

    for _ in range(min(n_unanswerable, len(UNABANSWERABLE_PREDICATES))):
        predicate = rng.choice(UNABANSWERABLE_PREDICATES)
        question = teacher.generate_probe(
            persona,
            FactSheetEntry(
                fact_id="__unanswerable__",
                subject=persona.persona_id,
                predicate=predicate,
                object="__absent__",
                category="factual",
            ),
            "unanswerable",
        )
        probe_index += 1
        probes.append(
            Probe(
                probe_id=f"{persona.persona_id}_p{probe_index:04d}",
                persona_id=persona.persona_id,
                question=question,
                gold_answer="",
                gold_supporting_memory_ids=[],
                category="unanswerable",
                answerable=False,
            )
        )

    if len(sessions) >= 2 and revealed_facts:
        for _ in range(min(n_outdated, len(revealed_facts))):
            fact = rng.choice(revealed_facts)
            # Must match on predicate, not just category: two facts sharing a
            # category (e.g. "job" and "degree" are both "factual") have unrelated
            # object pools, so a category-only match can pick a "corrected value"
            # from a completely different predicate (e.g. "correcting" a degree
            # subject to a job title) — a real bug caught via a real generation run.
            same_predicate = [o for o in FACT_POOL if o["predicate"] == fact.predicate]
            correction_obj = fact.object
            if same_predicate:
                candidates = [o for o in same_predicate[0]["objects"] if o != fact.object]
                if candidates:
                    correction_obj = rng.choice(candidates)

            correction_turn_id = f"{persona.persona_id}_s{len(sessions) - 1}_t{99:03d}"
            sessions[-1].turns.append(
                ConversationTurn(
                    turn_id=correction_turn_id,
                    session_id=sessions[-1].session_id,
                    role="user",
                    text=rng.choice(CORRECTION_PATTERNS).format(
                        predicate=fact.predicate, new_value=correction_obj
                    ),
                    ts=sessions[-1].turns[-1].ts + 60 if sessions[-1].turns else 9999,
                    revealed_fact_ids=[fact.fact_id],
                )
            )

            question = teacher.generate_probe(persona, fact, "outdated_fact")
            probe_index += 1
            probes.append(
                Probe(
                    probe_id=f"{persona.persona_id}_p{probe_index:04d}",
                    persona_id=persona.persona_id,
                    question=question,
                    gold_answer=correction_obj,
                    gold_supporting_memory_ids=[
                        fact_to_turn.get(fact.fact_id, ""),
                        correction_turn_id,
                    ],
                    category="outdated_fact",
                    answerable=True,
                )
            )

    if len(revealed_facts) >= 2:
        for _ in range(min(n_distractor, len(revealed_facts))):
            fact = rng.choice(revealed_facts)
            question = teacher.generate_probe(persona, fact, "distractor")
            probe_index += 1
            probes.append(
                Probe(
                    probe_id=f"{persona.persona_id}_p{probe_index:04d}",
                    persona_id=persona.persona_id,
                    question=question,
                    gold_answer=fact.object,
                    gold_supporting_memory_ids=[fact_to_turn[fact.fact_id]],
                    category="distractor",
                    answerable=True,
                )
            )

    if len(sessions) >= 2 and revealed_facts:
        early_facts = [f for f in revealed_facts if f.fact_id in fact_to_turn]
        if early_facts:
            for _ in range(min(n_continuity, len(early_facts))):
                fact = rng.choice(early_facts)
                question = teacher.generate_probe(persona, fact, "continuity")
                probe_index += 1
                probes.append(
                    Probe(
                        probe_id=f"{persona.persona_id}_p{probe_index:04d}",
                        persona_id=persona.persona_id,
                        question=question,
                        gold_answer=fact.object,
                        gold_supporting_memory_ids=[fact_to_turn[fact.fact_id]],
                        category="continuity",
                        answerable=True,
                    )
                )

    return probes


def _write_output(
    out_dir: Path,
    corpora: list[PersonaCorpus],
    all_probes: list[Probe],
    total_facts: int,
    teacher_name: str,
) -> None:
    personas_dir = out_dir / "personas"
    personas_dir.mkdir(parents=True, exist_ok=True)

    for corpus in corpora:
        path = personas_dir / f"{corpus.persona.persona_id}.json"
        path.write_text(json.dumps(corpus.model_dump(), indent=2) + "\n")

    probes_path = out_dir / "probes.jsonl"
    with open(probes_path, "w") as f:
        for probe in all_probes:
            f.write(json.dumps(probe.model_dump()) + "\n")

    datasheet_path = out_dir / "DATASHEET.md"
    n_personas = len(corpora)
    if teacher_name == "openai":
        limitations = (
            f"- This corpus was generated with a **live teacher model** "
            f"(OpenAI-compatible endpoint), not a deterministic fixture. It is "
            f"**not yet human-reviewed** — conversations and probes should be "
            f"spot-checked by a human before being treated as production-quality "
            f"reference data.\n"
            + (
                f"- This run ({n_personas} personas) is below the target scale of 8 "
                f"personas.\n"
                if n_personas < 8
                else ""
            )
        )
    else:
        limitations = (
            f"- This corpus was generated with a **fixture teacher** (deterministic "
            f"templates, no live LLM). Conversations and probes are synthetic "
            f"approximations, not naturally generated.\n"
            f"- A real teacher model (OpenAI-compatible endpoint) is required for "
            f"production-quality data. Pass `--teacher openai --teacher-model <model>` "
            f"once configured.\n"
            f"- Distractor and continuity probes are lightweight approximations; a "
            f"real teacher would produce richer variants.\n"
            f"- Per-category probe counts are derived proportionally from "
            f"`--facts-per-persona` rather than targeting the absolute reference "
            f"distribution from the paper.\n"
            f"- This fixture-scale run ({n_personas} personas) is a smoke-test "
            f"example, not the full v0 benchmark (target: 8 personas).\n"
        )
    datasheet_path.write_text(
        f"# {out_dir.name} — PMB v0 corpus\n\n"
        f"## Overview\n"
        f"Generated by `scripts/build_pmb.py`. This is a *generated corpus*, and its role "
        f"depends on where it was written: the `pmb_*` directories are the "
        f"**evaluation benchmark**; the `sft_personas_*` directories are disjoint "
        f"**training-persona** sets and must not be used for evaluation or reported as "
        f"benchmark results.\n\n"
        f"The PMB-v0 benchmark evaluates a system's ability to recall and reason over "
        f"personalised information across multi-session conversations.\n\n"
        f"## Generation\n\n"
        f"- **Teacher client**: `{teacher_name}`\n"
        f"- **Personas**: {n_personas}\n"
        f"- **Facts per persona**: {total_facts // max(1, n_personas)}\n"
        f"- **Total probes**: {len(all_probes)}\n\n"
        f"Probe categories: factual, preference, episodic, temporal, unanswerable, "
        f"outdated_fact, distractor, continuity.\n\n"
        f"## Limitations\n\n"
        f"{limitations}"
        f"- The `acceptable_alternatives` field is present on every probe but is "
        f"currently unpopulated, so exact-match (`pra_strict`) scoring is ~0 by "
        f"construction; judge-scored `pra_lenient` is the metric to report.\n"
    )


def _compute_hash(out_dir: Path) -> str:
    hasher = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(out_dir)):
        for fname in sorted(files):
            if fname == "hash.txt":
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(out_dir)
            hasher.update(str(rel).encode("utf-8"))
            hasher.update(fpath.read_bytes())
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PMB-v0 benchmark corpus",
        epilog=(
            "Teacher options:\n"
            "  fixture  Deterministic template-based teacher (default, no network)\n"
            "  openai   Live teacher backed by the OpenAI chat-completions API\n"
            "           (requires the optional 'judge' extra and an OPENAI_API_KEY)\n"
        ),
    )
    parser.add_argument("--out-dir", default="data/benchmarks/pmb_v0", help="Output directory path")
    parser.add_argument("--n-personas", type=int, default=8, help="Number of personas to generate")
    parser.add_argument("--sessions-per-persona", type=int, default=6, help="Sessions per persona")
    parser.add_argument("--turns-per-session", type=int, default=14, help="Turns per session")
    parser.add_argument("--facts-per-persona", type=int, default=40, help="Facts per persona")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument(
        "--teacher",
        default="fixture",
        choices=["fixture", "openai"],
        help="Teacher client to use",
    )
    parser.add_argument(
        "--teacher-model",
        default=None,
        help=(
            "teacher model id (default: $JUDGE_MODEL, falling back to 'gpt-4o' if unset) "
            "— the same OpenAI-compatible model the bake-off judge uses"
        ),
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.9,
        help="sampling temperature for the live teacher (default: 0.9)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    teacher_model = args.teacher_model or os.environ.get("JUDGE_MODEL") or "gpt-4o"

    if args.teacher == "fixture":
        teacher: TeacherClient = FixtureTeacherClient(seed=args.seed)
    elif args.teacher == "openai":
        teacher = OpenAITeacherClient(
            model=teacher_model,
            temperature=args.teacher_temperature,
        )
    else:
        print(f"Error: --teacher '{args.teacher}' is not supported.", file=sys.stderr)
        sys.exit(1)

    personas: list[Persona] = []
    for pi in range(args.n_personas):
        p = _mk_persona(pi, args.facts_per_persona, rng)
        personas.append(p)

    corpora: list[PersonaCorpus] = []
    for persona in personas:
        corpus = _generate_persona_corpus(
            persona, args.sessions_per_persona, args.turns_per_session, teacher, rng
        )
        corpora.append(corpus)

    all_probes: list[Probe] = []
    shared_rng = random.Random(args.seed + 1)
    for corpus in corpora:
        probes = _generate_probes(corpus, args.facts_per_persona, teacher, shared_rng)
        all_probes.extend(probes)

    _write_output(
        out_dir, corpora, all_probes, args.n_personas * args.facts_per_persona, args.teacher
    )

    hash_val = _compute_hash(out_dir)
    (out_dir / "hash.txt").write_text(hash_val + "\n")

    n_personas = len(corpora)
    n_sessions = sum(len(c.sessions) for c in corpora)
    n_turns = sum(len(s.turns) for c in corpora for s in c.sessions)
    print(
        f"Generated {n_personas} persona(s), {n_sessions} session(s), "
        f"{n_turns} turn(s), {len(all_probes)} probe(s)"
    )
    print(f"Output written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
