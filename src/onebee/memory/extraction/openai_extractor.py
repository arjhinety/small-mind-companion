from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from onebee.memory.extraction.schema import ExtractedClaim

# NOTE: the `openai` package is imported lazily inside method bodies. It is NOT a
# hard dependency of the package — it lives in the optional `judge` extra — so this
# module (and everything else in `onebee`) must import cleanly without it installed.


class OpenAITeacherExtractor:
    """A :class:`TeacherExtractor` backed by the OpenAI chat-completions API in JSON mode.

    The ``openai`` client is constructed per-request so that construction of this
    object never requires the package (or an API key) to be present.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
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
                # without the param instead of failing the whole extraction run.
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
            "OpenAITeacherExtractor: extractor model returned an unparseable response "
            f"even after a retry. Raw response: {raw!r}"
        )

    def extract(self, turn_text: str, context: dict) -> list[ExtractedClaim]:
        system_prompt = (
            "You extract durable personal-memory claims from a single user turn of a "
            "conversation. The claim is stored long-term and later recalled by an AI "
            "companion, so precision matters more than recall.\n\n"
            "THE MOST IMPORTANT RULE: every claim MUST include a `verbatim_span` that "
            "is an EXACT, verbatim substring of the turn text (copy it character-for-"
            "character, including original punctuation). Do NOT paraphrase, reword, or "
            "fabricate the span. This is critical because downstream validation checks "
            "that the span literally occurs in the turn; a claim with a paraphrased or "
            "invented span is rejected and useless.\n\n"
            "Extract 0-3 claims. Most turns yield 0-1 claims, and some yield 0 — an "
            "empty list is a perfectly valid answer. Extract claims that are durable "
            "and genuinely useful to remember about the speaker, such as preferences, "
            "relationships, work, health, life events, opinions, and projects. Skip "
            "greetings, small talk, and purely conversational filler.\n\n"
            "For each claim, provide:\n"
            '- "content": a concise natural-language sentence stating the claim.\n'
            '- "tier": one of "short_term", "episodic", or "semantic". "episodic" = '
            'an event with a time anchor ("yesterday I...", "last week we..."); '
            '"semantic" = a timeless fact or preference ("I love hiking", "my brother '
            'lives in Berlin"); "short_term" = ephemeral/conversational-only content '
            "not worth remembering long-term. Prefer episodic and semantic claims; "
            "use short_term only when unsure.\n"
            '- "verbatim_span": the exact verbatim substring of the turn text that '
            "supports the claim (see the rule above).\n"
            '- "subject"/"predicate"/"object": the claim components when the claim is '
            'triple-shaped (e.g. subject "I", predicate "work as", object '
            '"engineer"); set each to null when not applicable.\n'
            '- "entities": a list of proper-noun entities mentioned in the claim '
            "(people, places, organizations, brands). Empty list when none.\n"
            '- "topics": a list of general topic keywords for the claim (e.g. '
            '"work", "health", "travel", "family"). At least one recommended.\n'
            '- "assertion_strength": one of "definite", "moderate", or "uncertain", '
            'based on lexical hedging in the turn ("I think", "maybe", "kind '
            'of" -> weaker; direct statements -> "definite").\n'
            '- "source_reliability": almost always "user_statement" here (the claim '
            'comes straight from what the user said). Use "agent_inferred" or '
            '"reflection_derived" only when the claim is inferred rather than stated.\n'
            '- "attribution": one of "user", "agent", or "third_party" — is the claim '
            "about the speaker themselves, the AI companion, or someone else they "
            "mentioned? Set it from the actual content, do not assume.\n"
            '- "sensitive": true only if the claim touches health, sexual, financial, '
            "or identity-document content; otherwise false.\n"
            '- "extractor_confidence": a float from 0.0 to 1.0 expressing your own '
            "confidence that this is a genuine, useful claim.\n\n"
            "confidence that this is a genuine, useful claim.\n\n"
            "You MUST respond with exactly one JSON object and no other text, of the "
            'form: {"claims": [{ ...fields above per claim... }]}. An empty "claims" '
            "array is valid."
        )
        context_lines: list[str] = []
        persona_name = context.get("persona_name")
        if persona_name:
            context_lines.append(f"The speaker persona is: {persona_name}.")
        recent_turns = context.get("recent_turns")
        if recent_turns:
            context_lines.append(
                "Recent conversation context (use it to resolve pronouns, but extract "
                f"claims only from the turn below):\n{recent_turns}"
            )
        context_block = "\n\n".join(context_lines)
        user_prompt = (
            f"Turn text:\n{turn_text}"
            + (f"\n\n{context_block}" if context_block else "")
            + "\n\nReturn ONLY the JSON object of extracted claims."
        )

        data = self._request_json(system_prompt, user_prompt)
        raw_claims = data.get("claims")
        if not isinstance(raw_claims, list):
            raise RuntimeError(
                "OpenAITeacherExtractor: extraction JSON missing 'claims' array: " f"{data!r}"
            )

        claims: list[ExtractedClaim] = []
        for claim_dict in raw_claims:
            if not isinstance(claim_dict, dict):
                continue
            try:
                claims.append(ExtractedClaim(**claim_dict))
            except ValidationError:
                # Drop just this one malformed claim; the rest of the turn's claims
                # are still valid.
                continue
        return claims
