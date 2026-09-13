from __future__ import annotations

import json
import os
import random as stdlib_random
from typing import Any, Protocol, runtime_checkable

from onebee.data.personas import ConversationTurn, FactSheetEntry, Persona

# NOTE: the `openai` package is imported lazily inside method bodies. It is NOT a
# hard dependency of the package — it lives in the optional `judge` extra — so this
# module must import cleanly without it installed.


@runtime_checkable
class TeacherClient(Protocol):
    def generate_conversation(
        self,
        persona: Persona,
        session_index: int,
        target_turns: int,
        facts_to_reveal: list[FactSheetEntry],
    ) -> list[ConversationTurn]: ...

    def generate_probe(
        self,
        persona: Persona,
        fact: FactSheetEntry,
        category: str,
    ) -> str: ...


FACTUAL_TEMPLATES = [
    "What is {subject}'s {predicate}?",
    "What {predicate} does {subject} have?",
    "Tell me {subject}'s {predicate}.",
]

PREFERENCE_TEMPLATES = [
    "What is {subject}'s favorite {predicate}?",
    "What does {subject} prefer for {predicate}?",
    "Which {predicate} does {subject} like best?",
]

EPISODIC_TEMPLATES = [
    "What happened when {subject} {predicate}?",
    "Describe {subject}'s experience with {predicate}.",
    "What did {subject} do regarding {predicate}?",
]

TEMPORAL_TEMPLATES = [
    "When did {subject} {predicate}?",
    "When is {subject}'s {predicate}?",
    "At what time did {subject} {predicate}?",
]

UNABANSWERABLE_PREDICATES = [
    "pet's name",
    "sibling count",
    "middle name",
    "shoe size",
    "childhood nickname",
    "favorite movie",
    "first car model",
    "last vacation destination",
    "phone number",
    "email address",
]

ASSISTANT_REPLIES = [
    "That is interesting, {name}. Thanks for sharing!",
    "I see! Good to know, {name}.",
    "Got it, {name}. I will keep that in mind.",
    "Thanks for telling me, {name}.",
    "Noted, {name}!",
    "That makes sense, {name}.",
]

USER_FACT_STATEMENTS = {
    "factual": [
        "My {predicate} is {object}.",
        "I {predicate} {object}.",
        "I have {predicate} {object}.",
    ],
    "preference": [
        "My favorite {predicate} is {object}.",
        "I prefer {object} when it comes to {predicate}.",
        "I like {object} for {predicate}.",
    ],
    "episodic": [
        "Recently, I {predicate} {object}.",
        "I remember when I {predicate} {object}.",
        "Last time, I {predicate} {object}.",
    ],
    "temporal": [
        "My {predicate} is {object}.",
        "{predicate} happens on {object} for me.",
        "I have {predicate} on {object}.",
    ],
}


class FixtureTeacherClient:
    def __init__(self, seed: int = 1337):
        self._rng = stdlib_random.Random(seed)

    def generate_conversation(
        self,
        persona: Persona,
        session_index: int,
        target_turns: int,
        facts_to_reveal: list[FactSheetEntry],
    ) -> list[ConversationTurn]:
        session_id = f"{persona.persona_id}_s{session_index:03d}"
        base_ts = session_index * 100000
        turns: list[ConversationTurn] = []

        facts_per_user_turn = max(1, len(facts_to_reveal) // max(1, target_turns // 2))
        fact_idx = 0
        turn_idx = 0

        while fact_idx < len(facts_to_reveal) and turn_idx + 1 < target_turns:
            batch = facts_to_reveal[fact_idx : fact_idx + facts_per_user_turn]
            if not batch:
                break
            fact_idx += len(batch)

            statements = []
            for fact in batch:
                templates = USER_FACT_STATEMENTS.get(fact.category, USER_FACT_STATEMENTS["factual"])
                tmpl = self._rng.choice(templates)
                statements.append(tmpl.format(predicate=fact.predicate, object=fact.object))

            user_text = " ".join(statements)
            turn_id_user = f"{session_id}_t{turn_idx:03d}"
            ts = base_ts + turn_idx * 60
            turns.append(
                ConversationTurn(
                    turn_id=turn_id_user,
                    session_id=session_id,
                    role="user",
                    text=user_text,
                    ts=ts,
                    revealed_fact_ids=[f.fact_id for f in batch],
                )
            )
            turn_idx += 1

            reply_tmpl = self._rng.choice(ASSISTANT_REPLIES)
            assistant_text = reply_tmpl.format(name=persona.name)
            turn_id_asst = f"{session_id}_t{turn_idx:03d}"
            ts = base_ts + turn_idx * 60
            turns.append(
                ConversationTurn(
                    turn_id=turn_id_asst,
                    session_id=session_id,
                    role="assistant",
                    text=assistant_text,
                    ts=ts,
                    revealed_fact_ids=[],
                )
            )
            turn_idx += 1

        while turn_idx < target_turns:
            if turn_idx % 2 == 0:
                turn_id = f"{session_id}_t{turn_idx:03d}"
                ts = base_ts + turn_idx * 60
                turns.append(
                    ConversationTurn(
                        turn_id=turn_id,
                        session_id=session_id,
                        role="user",
                        text="How is the weather today?",
                        ts=ts,
                        revealed_fact_ids=[],
                    )
                )
            else:
                turn_id = f"{session_id}_t{turn_idx:03d}"
                ts = base_ts + turn_idx * 60
                turns.append(
                    ConversationTurn(
                        turn_id=turn_id,
                        session_id=session_id,
                        role="assistant",
                        text="The weather looks nice today!",
                        ts=ts,
                        revealed_fact_ids=[],
                    )
                )
            turn_idx += 1

        return turns

    def generate_probe(
        self,
        persona: Persona,
        fact: FactSheetEntry,
        category: str,
    ) -> str:
        if category == "unanswerable":
            predicates = self._rng.sample(
                UNABANSWERABLE_PREDICATES,
                k=min(len(UNABANSWERABLE_PREDICATES), 3),
            )
            predicate = self._rng.choice(predicates)
            return f"What is {persona.name}'s {predicate}?"

        cat_templates: dict[str, list[str]] = {
            "factual": FACTUAL_TEMPLATES,
            "preference": PREFERENCE_TEMPLATES,
            "episodic": EPISODIC_TEMPLATES,
            "temporal": TEMPORAL_TEMPLATES,
            "outdated_fact": FACTUAL_TEMPLATES,
            "distractor": FACTUAL_TEMPLATES,
            "continuity": EPISODIC_TEMPLATES,
        }

        templates = cat_templates.get(category, FACTUAL_TEMPLATES)
        tmpl = self._rng.choice(templates)
        return tmpl.format(subject=persona.name, predicate=fact.predicate, object=fact.object)


_PROBE_CATEGORY_GUIDANCE = {
    "factual": "Ask about a stable factual attribute of the persona that was shared.",
    "preference": "Ask about a preference the persona stated.",
    "episodic": "Ask about a specific event or experience the persona shared.",
    "temporal": "Ask about when something happened or is scheduled.",
    "unanswerable": (
        "The companion was never told this detail. Write a question about something "
        "NOT among the persona's revealed facts, such that a good companion must "
        "admit it does not know rather than fabricate an answer."
    ),
    "outdated_fact": (
        "Ask about a fact the user later corrected in conversation. A good companion "
        "must remember the corrected value, not the outdated one."
    ),
    "distractor": (
        "Ask about a plausible but never-mentioned detail. A good companion must not "
        "fabricate an answer for it."
    ),
    "continuity": (
        "Ask about a fact revealed in an earlier session, testing long-horizon recall "
        "across a multi-year relationship."
    ),
}


class OpenAITeacherClient:
    """A :class:`TeacherClient` backed by the OpenAI chat-completions API.

    Unlike :class:`FixtureTeacherClient`, this generates natural, varied conversations
    and probe questions via a live model. The ``openai`` client is constructed
    per-request so that construction of this object never requires the package (or an
    API key) to be present.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.9,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        # Set True after the API rejects a non-default temperature once (e.g. some
        # reasoning-style models only accept the implicit default); once learned,
        # every subsequent request for this instance skips the failing attempt.
        self._temperature_unsupported = False

    def _resolve_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        return os.environ.get("OPENAI_API_KEY")

    def _client(self) -> Any:
        import openai

        kwargs: dict[str, Any] = {}
        api_key = self._resolve_api_key()
        if api_key is not None:
            kwargs["api_key"] = api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def _request_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import openai

        client = self._client()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw: str | None = None
        for attempt in range(2):
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if not self._temperature_unsupported:
                request_kwargs["temperature"] = self.temperature
            try:
                completion = client.chat.completions.create(**request_kwargs)
            except openai.BadRequestError as exc:
                # Some models (e.g. reasoning-style models) reject any non-default
                # temperature and only accept the implicit default (1.0) — retry once
                # without the param instead of failing the whole generation run.
                if not self._temperature_unsupported and "temperature" in str(exc):
                    self._temperature_unsupported = True
                    request_kwargs.pop("temperature", None)
                    completion = client.chat.completions.create(**request_kwargs)
                else:
                    raise
            raw = completion.choices[0].message.content

            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                return parsed

            messages.append({"role": "assistant", "content": raw or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Return ONLY a single valid JSON object and nothing else."
                    ),
                }
            )

        raise RuntimeError(
            "OpenAITeacherClient: teacher model returned an unparseable response even "
            f"after a retry. Raw response: {raw!r}"
        )

    def _request_text(self, system_prompt: str, user_prompt: str) -> str:
        import openai

        client = self._client()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if not self._temperature_unsupported:
            request_kwargs["temperature"] = self.temperature
        try:
            completion = client.chat.completions.create(**request_kwargs)
        except openai.BadRequestError as exc:
            if not self._temperature_unsupported and "temperature" in str(exc):
                self._temperature_unsupported = True
                request_kwargs.pop("temperature", None)
                completion = client.chat.completions.create(**request_kwargs)
            else:
                raise
        content = completion.choices[0].message.content or ""
        return content.strip().strip('"').strip("'").strip()

    def generate_conversation(
        self,
        persona: Persona,
        session_index: int,
        target_turns: int,
        facts_to_reveal: list[FactSheetEntry],
    ) -> list[ConversationTurn]:
        system_prompt = (
            "You are writing synthetic dialogue data for a personal-memory benchmark. "
            "Write a natural multi-turn conversation between a USER (a real person, "
            "described below) and their AI COMPANION. The companion is a persistent "
            "AI companion — a companion, not a generic assistant — expected to hold "
            "its character, relationship history, and emotional continuity across "
            "years of conversation, so its replies must be warm, attentive, and "
            "personally engaged with what the user shares.\n\n"
            "The user should naturally reveal the provided facts to the companion "
            "over the course of the conversation. Do NOT dump all facts into a single "
            "turn — spread them across user turns the way a real conversation unfolds "
            "(small talk, follow-ups, and tangents are fine).\n\n"
            "You MUST respond with exactly one JSON object and no other text, of the "
            "form:\n"
            '{"turns": [{"role": "user" | "assistant", "text": "...", '
            '"revealed_fact_ids": ["fact_id", ...]}]}\n'
            "Every user turn that reveals one or more of the listed facts must list "
            'those facts\' "fact_id" values in "revealed_fact_ids"; user turns that '
            "reveal nothing get an empty list, and assistant turns always have an "
            f"empty list. Aim for approximately {target_turns} total turns (user and "
            "assistant turns combined)."
        )
        fact_lines = "\n".join(
            f'- {f.fact_id}: subject="{f.subject}" predicate="{f.predicate}" '
            f'object="{f.object}" category="{f.category}"'
            for f in facts_to_reveal
        )
        user_prompt = (
            f"Persona:\n{persona.name}. {persona.description}\n\n"
            f"Facts the user will reveal (reference them by fact_id):\n{fact_lines}"
        )

        data = self._request_json(system_prompt, user_prompt)
        raw_turns = data.get("turns")
        if not isinstance(raw_turns, list):
            raise RuntimeError(
                "OpenAITeacherClient: conversation JSON missing 'turns' array: " f"{data!r}"
            )

        session_id = f"{persona.persona_id}_s{session_index:03d}"
        base_ts = session_index * 100000
        turns: list[ConversationTurn] = []
        for idx, raw in enumerate(raw_turns):
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"OpenAITeacherClient: malformed turn in conversation JSON: {raw!r}"
                )
            role = raw.get("role")
            text = raw.get("text")
            revealed = raw.get("revealed_fact_ids", [])
            if role not in ("user", "assistant") or not isinstance(text, str):
                raise RuntimeError(
                    f"OpenAITeacherClient: malformed turn in conversation JSON: {raw!r}"
                )
            if not isinstance(revealed, list) or not all(isinstance(r, str) for r in revealed):
                raise RuntimeError(
                    "OpenAITeacherClient: malformed 'revealed_fact_ids' in "
                    f"conversation JSON: {raw!r}"
                )
            turns.append(
                ConversationTurn(
                    turn_id=f"{session_id}_t{idx:03d}",
                    session_id=session_id,
                    role=role,
                    text=text,
                    ts=base_ts + idx * 60,
                    revealed_fact_ids=revealed,
                )
            )
        return turns

    def generate_probe(
        self,
        persona: Persona,
        fact: FactSheetEntry,
        category: str,
    ) -> str:
        system_prompt = (
            "You are writing evaluation data for a personal-memory benchmark. Write "
            "ONE natural probe question (not a template-filled sentence) that tests "
            "whether an AI companion remembers a specific fact about its user. The "
            "companion is a companion, not a generic assistant: it builds a multi-year "
            "relationship and is expected to remember the personal details its user "
            "has shared.\n\n"
            f"Probe category: {category}.\n"
            f"{_PROBE_CATEGORY_GUIDANCE.get(category, 'Ask directly about the fact.')}\n\n"
            "CRITICAL: this benchmark grades answers against a fixed gold value — the "
            f"fact's object, {fact.object!r} — verbatim. The question you write MUST be "
            f"phrased so that {fact.object!r} is literally the correct, direct answer to "
            "it. Do NOT ask a 'when'/'how many'/'where' question if the object is not "
            "itself a time/count/place — match the question's expected answer type to "
            "the object's actual type. For example, if the object is a kind of pet, ask "
            "'what pet...', not 'when did...'.\n\n"
            "Respond with ONLY the question text — a single natural question, no "
            "JSON, no labels, no explanation."
        )
        user_prompt = (
            f"Persona: {persona.name}. {persona.description}\n\n"
            f"Fact: {fact.predicate} is {fact.object}.\n\n"
            "Write the probe question."
        )
        return self._request_text(system_prompt, user_prompt)
